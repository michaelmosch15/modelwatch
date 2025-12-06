import os
import sys
import glob
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Add app directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, "..", "app")
sys.path.insert(0, app_dir)

from src.transformer_filter_layer import GPT2WithPIIFiltering

# Configuration
EMAIL_DIR = os.path.join(current_dir, "..", "email_generation", "output_emails")
OUTPUT_DIR = os.path.join(current_dir, "..", "email_generation", "output_emails_sanitized")
MODEL_PATH = os.path.join(current_dir, "..", "models", "pii_filter_model.pt")

def sanitize_emails():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    print("Loading model...")
    model = GPT2WithPIIFiltering.from_pretrained("gpt2")
    
    # Load trained weights
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained weights from {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: No trained model found. Running with random filter weights (will likely produce garbage).")
    
    model.to(device)
    model.eval()

    files = glob.glob(os.path.join(EMAIL_DIR, "*.txt"))
    print(f"Processing {len(files)} emails...")

    for file_path in tqdm(files):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            # 1. Get Embeddings
            inputs_embeds = model.transformer.wte(inputs.input_ids)
            
            # 2. Run Filter
            filter_features = model.pii_filter_layer(inputs_embeds)
            pii_logits = model.pii_classifier(filter_features)
            pii_probs = torch.sigmoid(pii_logits) # (1, seq_len, 1)
            
        # 3. Reconstruct Text with Redaction
        # We iterate through tokens and check their PII probability
        input_ids = inputs.input_ids[0].cpu().numpy()
        probs = pii_probs[0, :, 0].cpu().numpy()
        
        output_tokens = []
        for token_id, prob in zip(input_ids, probs):
            if prob > 0.5: # Threshold for PII
                output_tokens.append("[REDACTED]")
            else:
                output_tokens.append(tokenizer.decode([token_id]))
        
        # Join and clean up
        # Simple join since we decoded individually. 
        # Note: This is a rough reconstruction. 
        sanitized_text = "".join(output_tokens)
        
        # Save
        filename = os.path.basename(file_path)
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(sanitized_text)

    print(f"Done! Sanitized emails saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    sanitize_emails()
