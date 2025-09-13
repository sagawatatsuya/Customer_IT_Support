from transformers import (
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
    return_dict=True,
)
print(inputs)
inputs = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ],
    return_tensors="pt",
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
)
print(inputs)
