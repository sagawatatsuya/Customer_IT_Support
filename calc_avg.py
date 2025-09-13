import numpy as np
import pandas as pd

df = pd.read_csv(
    "/home/sagawa/Customer_IT_Support/deepseek-r1-distill-qwen-32b-rag-k5-Qwen3-Embedding-0.6B-reranker-colbertv2.0-english-promptevaluation_results.csv"
)

print(np.mean(df["rougeL"]), np.mean(df["BERTScore"]))
