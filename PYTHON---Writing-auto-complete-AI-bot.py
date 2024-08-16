import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode (not training)
model.eval()

def complete_text(prompt, max_length=50):
    # Tokenize the input text and convert to tensors
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode the generated text to readable format
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    user_input = input("Start typing: ")
    completion = complete_text(user_input)
    print(f"Completed text:\n{completion}")
