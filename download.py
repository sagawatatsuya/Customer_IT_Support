# from huggingface_hub import snapshot_download

# path = snapshot_download(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     local_dir="/mnt/d/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct",
#     resume_download=True,
#     max_workers=8,  # 回線に合わせて
# )
# print("downloaded to:", path)

from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModel.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
)
print("model loaded")
