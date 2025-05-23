from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/DialoGPT-medium"
# Alternative: "microsoft/phi-2" or "huggingface/CodeBertaline-small"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded! Ready for inference.")

# Simple chat function
def chat_with_model(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Demo
prompt = "Hello, how are you today? Tell me a story"
response = chat_with_model(prompt)
print(f"Input: {prompt}")
print(f"Output: {response}")