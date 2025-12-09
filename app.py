import os
import sys
import torch
from flask import Flask, render_template, request
from transformers import AutoTokenizer

# Add current directory to path so we can import src.model
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.model import GPT2WithPIIFiltering

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model_logic(model_path):
    global model, tokenizer, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model configuration...")
    model = GPT2WithPIIFiltering.from_pretrained("gpt2")
    
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Model file not found at {model_path}")
        # In a real app we might want to handle this better, but for now print error
        return False
    
    model.to(device)
    model.eval()
    return True

def redact_text_logic(text, threshold=0.9):
    if not text:
        return ""
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        inputs_embeds = model.transformer.wte(inputs.input_ids)
        
        filter_features = model.pii_filter_layer(inputs_embeds)
        pii_logits = model.pii_classifier(filter_features)
        pii_probs = torch.sigmoid(pii_logits)
        
    input_ids = inputs.input_ids[0].cpu().numpy()
    probs = pii_probs[0, :, 0].cpu().numpy()
    
    output_tokens = []
    for token_id, prob in zip(input_ids, probs):
        token_str = tokenizer.decode([token_id])
        # Only redact if probability is high AND it's not a common stopword/punctuation
        # This is a heuristic to reduce false positives on common words
        if prob > threshold and len(token_str.strip()) > 1:
            output_tokens.append("[REDACTED]")
        else:
            output_tokens.append(token_str)
    
    redacted_text = "".join(output_tokens)
    return redacted_text

def generate_response_logic(text, max_new_tokens=150):
    if not text:
        return ""
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    
    # Decode only the newly generated tokens, excluding the input prompt
    new_tokens = output_sequences[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text

# Initialize model on startup
model_path = os.path.join(current_dir, "models", "pii_filter_model.pt")
if not load_model_logic(model_path):
    print("WARNING: Failed to load model. App may not function correctly.")

@app.route('/', methods=['GET', 'POST'])
def index():
    redacted_text = ""
    input_text = ""
    gpt_response = ""
    
    if request.method == 'POST':
        input_text = request.form.get('input_text', '')
        if input_text:
            redacted_text = redact_text_logic(input_text)
            # Generate response based on the redacted text
            gpt_response = generate_response_logic(redacted_text)
    
    return render_template('index.html', input_text=input_text, redacted_text=redacted_text, gpt_response=gpt_response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
