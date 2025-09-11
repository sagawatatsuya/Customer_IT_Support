import numpy as np
import pandas as pd

df = pd.read_csv(
    "/home/sagawa/Customer_IT_Support/gpt-oss-20b-rag-k5-Qwen3-Embedding-0.6B-reranker-colbertv2.0evaluation_results.csv"
)

print(np.mean(df["rougeL"]), np.mean(df["BERTScore"]))
