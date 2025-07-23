# build_db.py

import os

import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def main():
    # 環境変数読み込み
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY が .env に設定されていません")

    # 埋め込みインスタンス
    embedding = OpenAIEmbeddings(openai_api_key=api_key)

    # train.csvを読み込み
    train_df = pd.read_csv("./Customer_IT_Support/train.csv")

    docs = []
    for _, row in train_df.iterrows():
        metadata = {
            "type": row['type'],
            "queue": row['queue'],
            "priority": row['priority'],
            "answer": row['answer']
        }
        content = (
            f"subject: {row['subject']}\n"
            f"body: {row['body']}\n"
            f"language: {row['language']}\n"
            f"version: {row['version']}"
        )
        docs.append(Document(page_content=content, metadata=metadata))

    # ベクトルDBを構築
    db = FAISS.from_documents(docs, embedding)

    # 保存
    db.save_local("faiss_index")
    print("✅ ベクトルDBを faiss_index に保存しました。")

if __name__ == "__main__":
    main()
