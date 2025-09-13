from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-32B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(response)

def postprocess_output(text: str, model: str, tokenizer) -> str:
    if "gpt-oss" in model.lower():
        if "<|channel|>final<|message|>" in text:
            text = text.split("<|channel|>final<|message|>")[1].split("<|return|>")[0]
    elif "deepseek" in model.lower():
        if "</think>" in text:
            text = text.split("</think>")[-1].split("<｜end▁of▁sentence｜>")[0]
    # elif "llama" in model.lower():
    #     text = text.split("<|eot_id|>")[0].strip()
    else:
        # remove special tokens
        special_tokens = tokenizer.all_special_tokens
        for tok in special_tokens:
            text = text.replace(tok, "")
        text = text.strip()

    return text

response = postprocess_output(response, model_name, tokenizer)
print("Postprocessed response:")
print(response)