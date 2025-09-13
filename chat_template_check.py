from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)

model_id = "google/gemma-3-27b-it"  # 例
tokenizer = AutoProcessor.from_pretrained(model_id)
prompt = "Hello, how are you?"
inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
            ],
            return_tensors="pt",
            # tokenize=True,
            add_generation_prompt=True,
            return_dict=True,)
print(inputs)
inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": [{"type": "text", "text": prompt},]},
            ],
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,)
print(inputs)


# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
#             {"type": "text", "text": "Describe this image in detail."}
#         ]
#     }
# ]

# inputs = processor.apply_chat_template(
#     messages, add_generation_prompt=True, tokenize=True,
#     return_dict=True, return_tensors="pt"
# ).to(model.device, dtype=torch.bfloat16)