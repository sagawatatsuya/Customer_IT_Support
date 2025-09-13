#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv

dotenv_path = Path.home() / ".env"
load_dotenv(dotenv_path=dotenv_path)

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
    api_key: Optional[str] = None


class OpenAIBackend(LLMBackend):
    def __init__(self, cfg: OpenAIConfig):
        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API キーが見つかりません。--openai-api-key か 環境変数 OPENAI_API_KEY を設定してください。"
            )
        try:
            from langchain_openai import ChatOpenAI  # lazy
        except Exception as e:
            raise RuntimeError("langchain_openai の読み込みに失敗しました") from e

        self.llm = ChatOpenAI(
            model=cfg.model, temperature=cfg.temperature, openai_api_key=api_key
        )

    def generate(self, prompt: str) -> str:
        resp = self.llm.invoke(prompt)
        return getattr(resp, "content", str(resp))


@dataclass
class HFConfig:
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 10000
    device_map: str = "auto"
    trust_remote_code: bool = True
    reasoning_effort: str = "medium"


class HFBackend(LLMBackend):
    def __init__(self, cfg: HFConfig):
        self.cfg = cfg

        try:
            import torch  # lazy
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            # Mxfp4Config は一部のモデルでのみ
            try:
                from transformers import Mxfp4Config  # type: ignore
            except Exception:
                Mxfp4Config = None  # noqa: N806
        except Exception as e:
            raise RuntimeError("transformers/torch の読み込みに失敗しました") from e

        # 量子化設定
        if "gpt-oss" in cfg.model.lower() and Mxfp4Config:
            quantization_config = Mxfp4Config(
                quant_type="mxfp4",
                compute_dtype="bfloat16",
                use_double_quant=True,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
            )

        # Tokenizer / Processor
        if "gemma-3" in cfg.model.lower():
            self.tokenizer = AutoProcessor.from_pretrained(cfg.model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.model, trust_remote_code=cfg.trust_remote_code
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            dtype=torch.bfloat16,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        ).eval()

        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_new_tokens = cfg.max_new_tokens

    def generate(self, prompt: Any) -> str:
        kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        if "gpt-oss" in self.cfg.model.lower():
            kwargs["reasoning_effort"] = getattr(self.cfg, "reasoning_effort", None)

        inputs = self.tokenizer.apply_chat_template(
            prompt,
            **kwargs,
        ).to(self.model.device)
        # print decoded prompt for debugging
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            use_cache=True,
            # top_p=self.top_p,
        )

        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :])
        print(f"Raw generation:\n{text}")
        text = postprocess_output(text, self.cfg.model, self.tokenizer)
        return text


def postprocess_output(text: str, model: str, tokenizer) -> str:
    low = model.lower()
    if "gpt-oss" in low:
        if "<|channel|>final<|message|>" in text:
            text = text.split("<|channel|>final<|message|>")[1]
    elif "deepseek" in low:
        if "</think>" in text:
            text = text.split("</think>")[-1]
    # remove tokenizer special tokens
    special_tokens = tokenizer.all_special_tokens
    for tok in special_tokens:
        text = text.replace(tok, "")

    return text.strip()


# ------------------------------
# Embeddings + FAISS loader (for RAG)
# ------------------------------


@dataclass
class EmbeddingConfig:
    backend: str = "openai"  # 'openai' or 'hf'
    model: Optional[str] = None
    api_key: Optional[str] = None


def load_faiss(index_dir: str, emb_cfg: EmbeddingConfig):
    try:
        from langchain_community.vectorstores import FAISS  # lazy
        from langchain_huggingface import HuggingFaceEmbeddings  # lazy
        from langchain_openai import OpenAIEmbeddings  # lazy
    except Exception as e:
        raise RuntimeError("FAISS/Embeddings の読み込みに失敗しました") from e

    if emb_cfg.backend == "openai":
        api_key = emb_cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API キーが見つかりません（Embeddings）。--openai-api-key か 環境変数 OPENAI_API_KEY を設定してください。"
            )
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=emb_cfg.model)
    else:
        if not emb_cfg.model:
            raise ValueError(
                "HuggingFace 埋め込みを使う場合は --embedding-model を指定してください。"
            )
        embeddings = HuggingFaceEmbeddings(model_name=emb_cfg.model)

    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return db


# ------------------------------
# Prompting & parsing
# ------------------------------


CATEGORIES_TYPE: Sequence[str] = ("Change", "Incident", "Problem", "Request")
CATEGORIES_QUEUE: Sequence[str] = (
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
)
CATEGORIES_PRIORITY: Sequence[str] = ("high", "medium", "low")


def _build_context_block(similar_docs: Iterable) -> str:
    if similar_docs:
        ctx = "\n\n".join(
            "subject: {pc}\n"
            "type: {ty}\nqueue: {qu}\npriority: {pr}\nanswer: {ans}".format(
                pc=getattr(d, "page_content", ""),
                ty=d.metadata.get("type", ""),
                qu=d.metadata.get("queue", ""),
                pr=d.metadata.get("priority", ""),
                ans=d.metadata.get("answer", ""),
            )
            for d in similar_docs
        )
    else:
        ctx = "Not provided in this case."
    return f"--- Past Cases ---\n{ctx}"


def build_prompt(
    similar_docs: Iterable, row: pd.Series, use_rag: bool, backend: str, model: str
) -> str:
    context_block = (
        _build_context_block(similar_docs)
        if use_rag
        else "--- Past Cases ---\nNot provided in this case."
    )
    system_prompt = "You are a skilled IT support agent."

    core = f"""
Taking into account past cases, please propose the optimal `type`, `queue`, `priority`, and `answer` for a new request, following the guidelines below:
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

    if backend == "openai":
        return f"{system_prompt}\n\n{core}"

    low_model = model.lower()

    if "gpt-oss" in low_model:
        prompt = [
            {"role": "user", "content": f"{system_prompt}\n\n{core}"},
        ]
    elif "gemma-3" in low_model:
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": core},
                ],
            },
        ]
    else:
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": core},
        ]

    return prompt


_LINE_PAT = re.compile(r"^(type|queue|priority|answer)\s*:\s*(.*)$", re.IGNORECASE)


def parse_output(text: str) -> Tuple[str, str, str, str]:
    type_ = queue = priority = ""
    answer_lines: List[str] = []
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
    return type_, queue, priority, answer


# ------------------------------
# Reranking
# ------------------------------


def _load_reranker(reranker_id: str):
    """Cache RAGatouille reranker to avoid per-row reload."""
    try:
        from ragatouille import RAGPretrainedModel  # lazy
    except Exception as e:
        raise RuntimeError("ragatouille の読み込みに失敗しました") from e
    return RAGPretrainedModel.from_pretrained(reranker_id)


def rerank_docs_keep_metadata(
    reranker_id: str,
    query: str,
    docs: List,  # List[langchain.docstore.document.Document]
    top_k: int,
    shorten_chars: int = 1200,
):
    def _shorten(d, n=shorten_chars):
        text = d.page_content
        return text[:n] if (n and isinstance(text, str) and len(text) > n) else text

    passages = [_shorten(d) for d in docs]

    reranker_model = _load_reranker(reranker_id)
    ranked = reranker_model.rerank(query, passages, k=top_k)

    def _extract_text(item):
        return item.get("content")

    ranked_texts = [_extract_text(r) for r in ranked]

    # 同文面が重複しても安全なように、テキスト→複数indexのマップで対応
    text2idxs: dict[str, List[int]] = {}
    for i, t in enumerate(passages):
        text2idxs.setdefault(t, []).append(i)

    chosen_idx: List[int] = []
    for t in ranked_texts:
        idxs = text2idxs.get(t, [])
        if idxs:
            chosen_idx.append(idxs.pop(0))
    # 順位どおりに “メタ付き Document” を返す
    return [docs[i] for i in chosen_idx]


# ------------------------------
# Evaluation core
# ------------------------------


@dataclass
class EvalArgs:
    test_csv: str
    backend: str
    use_rag: bool
    rag_k: int
    faiss_index: Optional[str]
    openai_api_key: Optional[str]
    openai_model: str
    hf_model: str
    limit: Optional[int]
    output_prefix: str
    embedding_backend: str
    embedding_model: Optional[str]
    reranker: Optional[str]


def _ensure_output_prefix_dir(prefix: str) -> Path:
    p = Path(prefix)
    # prefix がディレクトリで終わる場合も考慮
    if p.suffix:
        # 例えば "runs/exp_" のような接頭辞（拡張子なし）なら親ディレクトリを作成
        dirpath = p.parent if p.name else Path(".")
    else:
        dirpath = p
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def evaluate(cfg: EvalArgs) -> None:
    model_name = cfg.openai_model if cfg.backend == "openai" else cfg.hf_model
    # Prepare LLM backend
    if cfg.backend == "openai":
        llm = OpenAIBackend(
            OpenAIConfig(model=model_name, temperature=0.0, api_key=cfg.openai_api_key)
        )
    elif cfg.backend == "hf":
        llm = HFBackend(HFConfig(model=model_name, temperature=0.0))
    else:
        raise ValueError("--backend は 'openai' か 'hf' を指定してください")

    # Vector store (optional)
    db = None
    if cfg.use_rag:
        if not cfg.faiss_index:
            raise ValueError("--use-rag を使う場合は --faiss-index を指定してください")
        db = load_faiss(
            cfg.faiss_index,
            EmbeddingConfig(
                backend=cfg.embedding_backend,
                model=cfg.embedding_model,
                api_key=cfg.openai_api_key,
            ),
        )

    test_df = pd.read_csv(cfg.test_csv)
    if isinstance(cfg.limit, int) and cfg.limit >= 0:
        test_df = test_df.head(cfg.limit)

    results: List[dict] = []
    try:
        from rouge_score import rouge_scorer  # lazy

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except Exception:
        rouge = None

    reranker_id = cfg.reranker
    for idx, row in test_df.iterrows():
        print(f"Processing row {idx + 1}/{len(test_df)}...")

        similar_docs = []
        if cfg.use_rag and db is not None:
            query_text = f"subject: {row['subject']}\nbody: {row['body']}\nlanguage: {row['language']}\nversion: {row['version']}"

            try:
                base_k = cfg.rag_k * 5 if reranker_id else cfg.rag_k
                similar_docs = db.similarity_search(query_text, k=base_k)
                if reranker_id:
                    similar_docs = rerank_docs_keep_metadata(
                        reranker_id=reranker_id,
                        query=query_text,
                        docs=similar_docs,
                        top_k=cfg.rag_k,
                    )
            except Exception as e:
                print(f"FAISS similarity search failed: {e}")
                similar_docs = []

        prompt = build_prompt(similar_docs, row, cfg.use_rag, cfg.backend, model_name)
        print("Prompt:")
        print(prompt)
        pred_text = llm.generate(prompt)
        type_, queue, priority, answer = parse_output(pred_text)
        print(
            f"Parsed: type={type_}, queue={queue}, priority={priority}, answer={answer}"
        )
        if rouge:
            rougeL = rouge.score(row["answer"], answer)["rougeL"].fmeasure
        else:
            rougeL = 0.0
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
    bert_ok = False
    try:
        from bert_score import score as bert_score  # lazy

        P, R, F1 = bert_score(
            df_results["pred_answer"].tolist(),
            df_results["true_answer"].tolist(),
            lang="en",
        )
        df_results["BERTScore"] = F1.numpy()
        bert_ok = True
    except Exception as e:
        print(f"BERTScore の計算に失敗しました: {e}")
        df_results["BERTScore"] = 0.0

    out_prefix = cfg.output_prefix or ""
    _ensure_output_prefix_dir(out_prefix)
    out_csv = f"{out_prefix}evaluation_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"📄 評価結果を {out_csv} に保存しました")

    _save_classification_artifacts(df_results, out_prefix)

    avg_rouge = float(df_results["rougeL"].mean()) if not df_results.empty else 0.0
    avg_bert = float(df_results["BERTScore"].mean()) if bert_ok else 0.0
    print(f"Average ROUGE-L: {avg_rouge:.4f}, Average BERTScore: {avg_bert:.4f}")
    # Save summary JSON
    summary_path = f"{out_prefix}summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": cfg.backend,
                "model": model_name,
                "use_rag": cfg.use_rag,
                "rag_k": cfg.rag_k,
                "embedding_backend": cfg.embedding_backend,
                "embedding_model": cfg.embedding_model,
                "reranker": cfg.reranker,
                "size": len(df_results),
                "avg_rougeL": avg_rouge,
                "avg_bertscore": avg_bert,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("✅ 全ての評価が完了しました")


def _save_classification_artifacts(df_results: pd.DataFrame, out_prefix: str) -> None:
    """Save classification reports & confusion matrices."""
    try:
        import matplotlib.pyplot as plt  # lazy
        import seaborn as sns  # lazy
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
    except Exception as e:
        print(f"可視化/分類メトリクス依存の読み込みに失敗: {e}")
        return

    reports = {}

    for col in ("type", "queue", "priority"):
        y_true = df_results[f"true_{col}"].fillna("N/A")
        y_pred = df_results[f"pred_{col}"].fillna("N/A")

        labels = sorted(list(set(y_true) | set(y_pred)))
        try:
            cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        except Exception:
            cm = confusion_matrix(y_true, y_pred, labels=labels)
        present_labels = sorted(set(y_true))
        macro_f1 = f1_score(
            y_true, y_pred, labels=present_labels, average="macro", zero_division=0
        )

        rep = classification_report(y_true, y_pred, zero_division=0)
        reports[col] = {"macro_f1": macro_f1, "report_text": rep}

        # Save heatmap
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if cm.dtype.kind in "fc" else "d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(f"{col.upper()} Confusion Matrix (Macro F1={macro_f1:.4f})")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        fig_path = f"{out_prefix}confusion_matrix_{col}.png"
        Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()
        print(f"🖼 Confusion matrix saved: {fig_path}")

        # Save report text
        rpt_path = f"{out_prefix}classification_report_{col}.txt"
        with open(rpt_path, "w", encoding="utf-8") as f:
            f.write(rep)
        print(f"📝 Classification report saved: {rpt_path}")


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

    return EvalArgs(
        test_csv=args.test_csv,
        backend=args.backend,
        use_rag=args.use_rag,
        rag_k=args.rag_k,
        faiss_index=args.faiss_index,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        hf_model=args.hf_model,
        limit=args.limit,
        output_prefix=args.output_prefix,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        reranker=args.reranker,
    )


def main(argv: List[str] | None = None):
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    evaluate(cfg)


if __name__ == "__main__":
    main()
