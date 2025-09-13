# Customer_IT_Support
Kaggleデータを使ってRAGを試してみた。


# install
```
conda create -n cis python=3.11
conda activate cis
pip install --upgrade pip
pip install --upgrade transformers datasets
pip install --upgrade jupyter
pip install python-dotenv
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install faiss-cpu
pip install matplotlib seaborn
pip install bert-score rouge-score
pip install langchain-huggingface
pip install sentence-transformers
pip install accelerate bitsandbytes
pip install --upgrade kernels "triton>=3.4"
pip install RAGatouille
```

## 使い方
```
python build_db.py

CUDA_VISIBLE_DEVICES=0 python benchmark.py \
  --backend hf \
  --hf-model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --test-csv ./Customer_IT_Support/test.csv \
  --use-rag \
  --rag-k 5 \
  --embedding-backend hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --faiss-index ./faiss_index_Qwen_Qwen3-Embedding-0.6B \
  --reranker colbert-ir/colbertv2.0 \
  --output-prefix ./results/deepseek-r1-distill-qwen-32b-rag-k5-Qwen3-Embedding-0.6B-reranker-colbertv2.0-english-prompt

CUDA_VISIBLE_DEVICES=1 python benchmark.py \
  --backend openai \
  --openai-model gpt-4o \
  --test-csv ./Customer_IT_Support/test.csv \
  --use-rag \
  --rag-k 5 \
  --embedding-backend hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --faiss-index ./faiss_index_Qwen_Qwen3-Embedding-0.6B \
  --reranker colbert-ir/colbertv2.0 \
  --output-prefix ./results/gpt-4o-rag-k5-Qwen3-Embedding-0.6B-reranker-colbertv2.0-english-prompt


```

# Todo
- RAGのembedding modelの選択(https://huggingface.co/spaces/mteb/leaderboard)
- RAGのdb作成時のチャンク生成方法の考察
- RAG + reranking, reranking modelの選択
