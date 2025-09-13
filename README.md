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
pip install matplotlib
pip install bert-score
pip install rouge-score
pip install langchain-huggingface
pip install sentence-transformers
pip install accelerate bitsandbytes
pip install --upgrade kernels "triton>=3.4"
pip install RAGatouille
```

## 使い方
```
pyton build_db.py
CUDA_VISIBLE_DEVICES=1 python benchmark.py \
  --backend hf \
  --hf-model Qwen/Qwen2.5-32B-Instruct \
  --test-csv ./Customer_IT_Support/test.csv \
  --use-rag \
  --rag-k 5 \
  --embedding-backend hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --faiss-index /home/sagawa/Customer_IT_Support/faiss_index_Qwen_Qwen3-Embedding-0.6B \
  --reranker colbert-ir/colbertv2.0 \
  --output-prefix qwen2.5-32B-instruct-rag-k5-Qwen3-Embedding-0.6B-reranker-colbertv2.0

CUDA_VISIBLE_DEVICES=0 python benchmark.py \
  --backend hf \
  --hf-model openai/gpt-oss-20b \
  --test-csv ./Customer_IT_Support/test.csv \
  --use-rag \
  --rag-k 5 \
  --embedding-backend hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --faiss-index /home/sagawa/Customer_IT_Support/faiss_index_Qwen_Qwen3-Embedding-0.6B \
  --reranker colbert-ir/colbertv2.0 \
  --output-prefix gpt-oss-20b-rag-k5-Qwen3-Embedding-0.6B-reranker-colbertv2.0-temperature1.0-topp1.0

HF_HUB_DISABLE_XET=1 \
CUDA_VISIBLE_DEVICES=0 python benchmark.py \
  --backend hf \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --test-csv ./Customer_IT_Support/test.csv \
  --use-rag \
  --rag-k 5 \
  --embedding-backend hf \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --faiss-index /home/sagawa/Customer_IT_Support/faiss_index_Qwen_Qwen3-Embedding-0.6B \
  --reranker colbert-ir/colbertv2.0 \
  --output-prefix llama-3.1-8b-instruct-rag-k5-Qwen3-Embedding-0.6B-reranker-colbertv2.0-english-prompt

HF_HUB_DISABLE_XET=1 python download.py
```

# Todo
- RAGのembedding modelの選択(https://huggingface.co/spaces/mteb/leaderboard)
- RAGのdb作成時のチャンク生成方法の考察
- RAG + reranking, reranking modelの選択
