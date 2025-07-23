# infer.py

import os

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def build_prompt(similar_docs, query_row):
    context = "\n\n".join(
        f"subject: {doc.page_content}\ntype: {doc.metadata['type']}\nqueue: {doc.metadata['queue']}\npriority: {doc.metadata['priority']}\nanswer: {doc.metadata['answer']}"
        for doc in similar_docs
    )

    prompt = f"""
以下の過去事例を参考にして、新しい問い合わせに対して最適なtype, queue, priority, answerを提案してください。

--- 過去の事例 ---
{context}

--- 新しい問い合わせ ---
subject: {query_row['subject']}
body: {query_row['body']}
language: {query_row['language']}
version: {query_row['version']}

出力フォーマット:
type: …
queue: …
priority: …
answer: …
"""
    return prompt

def main():
    # 環境変数読み込み
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY が .env に設定されていません")

    # 埋め込み & LLM
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=api_key)

    # ベクトルDBをロード
    db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

    # test.csvを読み込み
    test_df = pd.read_csv("./Customer_IT_Support/test.csv")

    # 行番号を指定
    row_idx = int(input(f"test.csv の行番号を入力してください（0〜{len(test_df)-1}）： "))
    query_row = test_df.iloc[row_idx]

    query_text = (
        f"subject: {query_row['subject']}\n"
        f"body: {query_row['body']}\n"
        f"language: {query_row['language']}\n"
        f"version: {query_row['version']}"
    )

    # 類似事例を検索
    similar_docs = db.similarity_search(query_text, k=3)

    # プロンプトを構築
    prompt = build_prompt(similar_docs, query_row)

    # LLM呼び出し
    result = llm.invoke(prompt)

    print("\n===== 予測結果 =====")
    print(result.content)


if __name__ == "__main__":
    main()
