#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bert_score import score as bert_score
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from ragatouille import RAGPretrainedModel
from rouge_score import rouge_scorer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Mxfp4Config,
)

# Optional imports are deferred until used

load_dotenv()

# ------------------------------
# LLM backends
# ------------------------------


class LLMBackend:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@dataclass
class OpenAIConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    api_key: str | None = None


class OpenAIBackend(LLMBackend):
    def __init__(self, cfg: OpenAIConfig):
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise RuntimeError(
                "langchain_openai が見つかりません。`pip install langchain-openai` を実行してください"
            ) from e
        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API キーが見つかりません。--openai-api-key か 環境変数 OPENAI_API_KEY を設定してください。"
            )
        self.llm = ChatOpenAI(
            model=cfg.model, temperature=cfg.temperature, openai_api_key=api_key
        )

    def generate(self, prompt: str) -> str:
        resp = self.llm.invoke(prompt)
        return getattr(resp, "content", str(resp))


import torch


@dataclass
class HFConfig:
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 10000
    device_map: str = "cuda" if torch.cuda.is_available() else "auto"
    trust_remote_code: bool = True
    reasoning_effort: str = "medium"


class HFBackend(LLMBackend):
    def __init__(self, cfg: HFConfig):
        self.cfg = cfg

        if "gpt-oss" in cfg.model:
            quantization_config = Mxfp4Config(
                quant_type="mxfp4",
                compute_dtype="bfloat16",
                use_double_quant=True,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 8bitなら load_in_8bit=True
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",  # もしくは "fp4"
                bnb_4bit_compute_dtype="bfloat16",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, trust_remote_code=cfg.trust_remote_code
        )
        # attn_impl = "kernels-community/vllm-flash-attn3"
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            # attn_implementation=attn_impl,
        ).eval()
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_new_tokens = cfg.max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer.apply_chat_template(
            [
                # {
                #     "role": "system",
                #     "content": "Finish reasoning process in 1024 tokens or less.",
                # },
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=self.cfg.reasoning_effort
            if "gpt-oss" in self.cfg.model
            else None,
        ).to(self.model.device)
        # print decoded prompt for debugging
        # print(self.tokenizer.decode(inputs["input_ids"][0]))
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            use_cache=True,
            # top_p=self.top_p,
        )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :])
        print(text)
        text = postprocess_output(text, self.cfg.model)
        print(f"Model output (postprocessed): {text}")
        return text


def postprocess_output(text: str, model: str) -> str:
    if "gpt-oss" in model.lower():
        if "<|channel|>final<|message|>" in text:
            text = text.split("<|channel|>final<|message|>")[1].split("<|return|>")[0]
    elif "deepseek" in model.lower():
        if "</think>" in text:
            text = text.split("</think>")[-1].split("<｜end▁of▁sentence｜>")[0]
    elif "llama" in model.lower():
        text = text.split("<|eot_id|>")[0].strip()

    return text


# ------------------------------
# Embeddings + FAISS loader (for RAG)
# ------------------------------


@dataclass
class EmbeddingConfig:
    backend: str = "openai"  # only 'openai' supported in this ref impl
    model: str | None = None
    api_key: str | None = None


def load_faiss(index_dir: str, emb_cfg: EmbeddingConfig):
    if emb_cfg.backend == "openai":
        api_key = emb_cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API キーが見つかりません（Embeddings）。--openai-api-key か 環境変数 OPENAI_API_KEY を設定してください。"
            )
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=emb_cfg.model)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=emb_cfg.model)

    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return db


# ------------------------------
# Prompting & parsing
# ------------------------------

CATEGORIES_TYPE = ["Change", "Incident", "Problem", "Request"]
CATEGORIES_QUEUE = [
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "IT Support",
    "Product Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Technical Support",
]
CATEGORIES_PRIORITY = ["high", "medium", "low"]


def build_prompt(similar_docs: Iterable, row: pd.Series, use_rag: bool) -> str:
    if use_rag and similar_docs:
        context = "\n\n".join(
            f"subject: {d.page_content}\ntype: {d.metadata.get('type', '')}\nqueue: {d.metadata.get('queue', '')}\npriority: {d.metadata.get('priority', '')}\nanswer: {d.metadata.get('answer', '')}"
            for d in similar_docs
        )
        context_block = f"""
--- Past Cases ---
{context}
""".strip()
    else:
        context_block = """
--- Past Cases ---
Not provided in this case.
""".strip()

    prompt = f"""
You are a skilled IT support agent. Taking into account past cases, please propose the optimal `type`, `queue`, `priority`, and `answer` for a new request, following the guidelines below:
- Choose `type` from: {" / ".join(CATEGORIES_TYPE)}
- Choose `queue` from: {" / ".join(CATEGORIES_QUEUE)}
- Choose `priority` from: {" / ".join(CATEGORIES_PRIORITY)}
- The `answer` should be a concrete and appropriate response based on the content of the body.

{context_block}

--- New Query ---
subject: {row["subject"]}
body: {row["body"]}
language: {row["language"]}
version: {row["version"]}

Please output exactly once in the following format:

type: <Change|Incident|Problem|Request>  
queue: <one of the queue categories>  
priority: <high|medium|low>  
answer: <your response>
""".strip()
    return prompt


_LINE_PAT = re.compile(r"^(type|queue|priority|answer)\s*:\s*(.*)$", re.IGNORECASE)


def parse_output(text: str) -> Tuple[str, str, str, str]:
    type_ = queue = priority = answer = ""
    answer_lines = []
    capture_answer = False

    for line in text.splitlines():
        m = _LINE_PAT.match(line.strip())
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "type":
                type_ = val
            elif key == "queue":
                queue = val
            elif key == "priority":
                priority = val
            elif key == "answer":
                # 「answer:」行に出会ったら残りはすべて回答文扱い
                answer_lines.append(val)
                capture_answer = True
        elif capture_answer:
            # answer: の後続行を全部追加
            answer_lines.append(line)
    answer = "\n".join(answer_lines).strip()
    # # 後ろに余計なゴミが付いた場合の簡易クリーニング
    # for name, allowed in (("type", CATEGORIES_TYPE), ("priority", CATEGORIES_PRIORITY)):
    #     pass
    return type_, queue, priority, answer


def rerank_docs_keep_metadata(
    reranker_id: str,
    query: str,
    docs: List,  # List[langchain.docstore.document.Document]
    top_k: int,
    shorten_chars: int = 1200,  # 長文は軽く短縮（任意）
):
    # 1) 文字列だけを作る（subjectは残しつつ短縮）
    def _shorten(d, n=shorten_chars):
        text = d.page_content
        if n and len(text) > n:
            return text[:n]
        return text

    passages = [_shorten(d) for d in docs]

    # 2) リランク（RAGatouilleは List[str] を想定）
    reranker_model = RAGPretrainedModel.from_pretrained(reranker_id)
    ranked = reranker_model.rerank(query, passages, k=top_k)

    # 3) 戻り値の形に応じて “元のインデックス” を復元
    #   例1: [{'document': '...','score': 12.3}, ...]
    #   例2: ['...', '...', ...]
    def _extract_text(item):
        return item.get("content")

    ranked_texts = [_extract_text(r) for r in ranked]

    # 同文面が重複しても安全なように、テキスト→複数indexのマップで対応
    text2idxs = {}
    for i, t in enumerate(passages):
        text2idxs.setdefault(t, []).append(i)

    chosen_idx = []
    for t in ranked_texts:
        lst = text2idxs.get(t, [])
        if lst:
            chosen_idx.append(lst.pop(0))
    # 4) 順位どおりに “メタ付き Document” を返す
    return [docs[i] for i in chosen_idx]


# ------------------------------
# Evaluation core
# ------------------------------


def evaluate(
    test_csv: str,
    backend: str,
    use_rag: bool,
    rag_k: int,
    faiss_index: str | None,
    openai_api_key: str | None,
    openai_model: str,
    hf_model: str,
    limit: int | None,
    output_prefix: str,
    embedding_backend: str,
    embedding_model: str | None,
    reranker: str | None = None,
):
    # Prepare LLM backend
    if backend == "openai":
        llm = OpenAIBackend(
            OpenAIConfig(model=openai_model, temperature=0.0, api_key=openai_api_key)
        )
    elif backend == "hf":
        llm = HFBackend(HFConfig(model=hf_model, temperature=0.0))
    else:
        raise ValueError("--backend は 'openai' か 'hf' を指定してください")

    # Optional FAISS
    db = None
    if use_rag:
        if not faiss_index:
            raise ValueError("--use-rag を使う場合は --faiss-index を指定してください")
        db = load_faiss(
            faiss_index,
            EmbeddingConfig(
                backend=embedding_backend, model=embedding_model, api_key=openai_api_key
            ),
        )

    test_df = pd.read_csv(test_csv)
    if limit is not None:
        test_df = test_df.head(limit)

    results: List[dict] = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for idx, row in test_df.iterrows():
        print(f"Processing row {idx + 1}/{len(test_df)}...")
        similar_docs = []
        if use_rag and db is not None:
            query_text = f"subject: {row['subject']}\nbody: {row['body']}\nlanguage: {row['language']}\nversion: {row['version']}"

            try:
                similar_docs = db.similarity_search(
                    query_text, k=rag_k * 5 if reranker else rag_k
                )
                if reranker:
                    similar_docs = rerank_docs_keep_metadata(
                        reranker_id=reranker,
                        query=query_text,
                        docs=similar_docs,
                        top_k=rag_k,
                    )
            except Exception as e:
                print(f"FAISS similarity search failed: {e}")
                similar_docs = []

        prompt = build_prompt(similar_docs, row, use_rag)
        print("Prompt:")
        print(prompt)
        pred_text = llm.generate(prompt)

        type_, queue, priority, answer = parse_output(pred_text)
        print(
            f"Parsed: type={type_}, queue={queue}, priority={priority}, answer={answer}"
        )

        rougeL = scorer.score(row["answer"], answer)["rougeL"].fmeasure
        results.append(
            {
                "true_type": row["type"],
                "pred_type": type_,
                "true_queue": row["queue"],
                "pred_queue": queue,
                "true_priority": row["priority"],
                "pred_priority": priority,
                "true_answer": row["answer"],
                "pred_answer": answer,
                "rougeL": rougeL,
            }
        )

    df_results = pd.DataFrame(results)

    # BERTScore
    P, R, F1 = bert_score(
        df_results["pred_answer"].tolist(),
        df_results["true_answer"].tolist(),
        lang="en",
    )
    df_results["BERTScore"] = F1.numpy()

    out_csv = f"{output_prefix}evaluation_results.csv"

    df_results.to_csv(out_csv, index=False)
    print(f"📄 評価結果を {out_csv} に保存しました")

    # Classification metrics + Confusion matrices
    for col in ["type", "queue", "priority"]:
        y_true = df_results[f"true_{col}"]
        y_pred = df_results[f"pred_{col}"]

        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        present_labels = sorted(set(y_true))
        macro_f1 = f1_score(
            y_true, y_pred, labels=present_labels, average="macro", zero_division=0
        )
        print(f"\n{col.upper()} Macro F1: {macro_f1:.4f}")
        print(classification_report(y_true, y_pred))

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(f"{col.upper()} Confusion Matrix (Macro F1={macro_f1:.4f})")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        fig_path = f"{output_prefix}confusion_matrix_{col}.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"🖼 Confusion matrix saved: {fig_path}")

    print("✅ 全ての評価が完了しました")


# ------------------------------
# CLI
# ------------------------------


def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(
        description="Evaluate IT support ticket classification + generation"
    )
    p.add_argument("--test-csv", required=True, help="評価用 CSV ファイルのパス")
    p.add_argument(
        "--backend",
        choices=["openai", "hf"],
        default="openai",
        help="LLM バックエンドの選択",
    )

    # OpenAI
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"))

    # HF
    p.add_argument("--hf-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # RAG / FAISS
    p.add_argument(
        "--use-rag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="RAG を使う/使わない",
    )
    p.add_argument("--rag-k", type=int, default=3, help="RAG で使う類似事例数")
    p.add_argument(
        "--faiss-index",
        default="./faiss_index",
        help="FAISS インデックスのディレクトリ",
    )
    p.add_argument("--embedding-backend", choices=["openai", "hf"], default="openai")
    p.add_argument(
        "--embedding-model", default=None, help="埋め込みモデル名（OpenAI or HF）"
    )

    # Others
    p.add_argument(
        "--limit", type=int, default=None, help="評価する最大件数（None で全件）"
    )
    p.add_argument(
        "--reranker", type=str, default=None, help="RAG 用の再ランキングモデル"
    )
    p.add_argument("--output-prefix", default="", help="出力ファイル名の接頭辞")

    args = p.parse_args(argv)
    return args


def main(argv: List[str] | None = None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    evaluate(
        test_csv=args.test_csv,
        backend=args.backend,
        use_rag=args.use_rag,
        rag_k=args.rag_k,
        faiss_index=args.faiss_index,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        hf_model=args.hf_model,
        limit=args.limit if isinstance(args.limit, int) and args.limit >= 0 else None,
        output_prefix=args.output_prefix,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        reranker=args.reranker,
    )


if __name__ == "__main__":
    main()
