import numpy as np
import pandas as pd

df = pd.read_csv(
    "/data2/sagawatatsuya/Customer_IT_Support/results/llama-3.1-8b-instruct-rag-k5-Qwen3-Embedding-0.6B-reranker-colbertv2.0-english-promptevaluation_results.csv"
)

print(np.mean(df["rougeL"]), np.mean(df["BERTScore"]))
