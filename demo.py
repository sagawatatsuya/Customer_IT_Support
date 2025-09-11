from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # 例
tokenizer = AutoTokenizer.from_pretrained(model_id)
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,  # 8bitなら load_in_8bit=True
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # もしくは "fp4"
    bnb_4bit_compute_dtype="bfloat16",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_cfg,
    device_map="auto",
)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# prompt = "You are a helpful assistant.\nUser: Explain RAG.\nAssistant:"
# out = pipe(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
# print(out)

# # use Mxfp4Config
# from transformers import Mxfp4Config

# quantization_config = Mxfp4Config(
#     quant_type="mxfp4",
#     compute_dtype="bfloat16",
#     use_double_quant=True,
# )

# model_id = "openai/gpt-oss-20b"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     quantization_config=quantization_config,
# )

messages = [
    {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=10000)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :]))
