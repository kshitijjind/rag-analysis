from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token="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")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", token="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")


def generate_text(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids)
    return tokenizer.decode(outputs[0])
