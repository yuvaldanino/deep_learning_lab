from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"  # You can also try "t5-base" or "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define input text
input_text = "Generate a structured bullet-point outline for: an app that tracks how many made bsketabll shots you have with some kind of open ai model and it then makes stats on the made vs missed shots you have "
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate output
output_ids = model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print output
print("Generated Output:", output_text)
