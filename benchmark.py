# evaluate.py

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bert_score import score as bert_score
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rouge_score import rouge_scorer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(openai_api_key=api_key)
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)

db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

test_df = pd.read_csv("./Customer_IT_Support/test.csv")

# 保存用データ
results = []

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

use_rag = True


def build_prompt(similar_docs, query_row):
    context = "\n\n".join(
        f"subject: {doc.page_content}\ntype: {doc.metadata['type']}\nqueue: {doc.metadata['queue']}\npriority: {doc.metadata['priority']}\nanswer: {doc.metadata['answer']}"
        for doc in similar_docs
    )

    if use_rag:
        context_block = f"""
    --- 過去の事例 ---
    {context}
    """
    else:
        context_block = """
    --- 過去の事例 ---
    （今回のケースでは過去事例は提供されていません。）
    """

    prompt = f"""
    過去の事例を考慮して、以下の条件に従い、新しい問い合わせに対して最適な type, queue, priority, answer を提案してください。

    - type は以下のいずれかから選んでください：
    Change / Incident / Problem / Request

    - queue は以下のいずれかから選んでください：
    Billing and Payments /
    Customer Service /
    General Inquiry /
    Human Resources /
    IT Support /
    Product Support /
    Returns and Exchanges /
    Sales and Pre-Sales /
    Service Outages and Maintenance /
    Technical Support

    - priority は以下のいずれかから選んでください：
    high / medium / low

    - answer は、body に書かれた内容に対する具体的で適切な回答案を生成してください。

    {context_block}

    --- 新しい問い合わせ ---
    subject: {query_row['subject']}
    body: {query_row['body']}
    language: {query_row['language']}
    version: {query_row['version']}

    出力フォーマットは以下の通りです：
    type: …
    queue: …
    priority: …
    answer: …
    """
    return prompt


def parse_output(text):
    # 正規表現で抽出
    type_ = queue = priority = answer = ""
    for line in text.splitlines():
        if line.lower().startswith("type:"):
            type_ = line.split(":", 1)[1].strip()
        elif line.lower().startswith("queue:"):
            queue = line.split(":", 1)[1].strip()
        elif line.lower().startswith("priority:"):
            priority = line.split(":", 1)[1].strip()
        elif line.lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
    return type_, queue, priority, answer

for idx, row in test_df.iterrows():
    print(f"Processing row {idx+1}/{len(test_df)}...")
    query_text = f"subject: {row['subject']}\nbody: {row['body']}\nlanguage: {row['language']}\nversion: {row['version']}"
    similar_docs = db.similarity_search(query_text, k=3)
    prompt = build_prompt(similar_docs, row)
    pred = llm.invoke(prompt)
    type_, queue, priority, answer = parse_output(pred.content)

    rougeL = scorer.score(row['answer'], answer)['rougeL'].fmeasure
    results.append({
        'true_type': row['type'], 'pred_type': type_,
        'true_queue': row['queue'], 'pred_queue': queue,
        'true_priority': row['priority'], 'pred_priority': priority,
        'true_answer': row['answer'], 'pred_answer': answer,
        'rougeL': rougeL
    })

df_results = pd.DataFrame(results)

# BERTScore 計算
P, R, F1 = bert_score(df_results['pred_answer'].tolist(), df_results['true_answer'].tolist(), lang='en')
df_results['BERTScore'] = F1.numpy()

df_results.to_csv("evaluation_results.csv", index=False)
print("📄 評価結果を evaluation_results.csv に保存しました")

# 分類タスク評価
for col in ['type', 'queue', 'priority']:
    y_true = df_results[f'true_{col}']
    y_pred = df_results[f'pred_{col}']

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    present_labels = sorted(set(y_true))
    macro_f1 = f1_score(y_true, y_pred, labels=present_labels, average='macro', zero_division=0)
    print(f"\n{col.upper()} Macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred))

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"{col.upper()} Confusion Matrix (Macro F1={macro_f1:.4f})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{col}.png")
    plt.close()
    print(f"🖼 Confusion matrix saved: confusion_matrix_{col}.png")

print("✅ 全ての評価が完了しました")
