import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

# Ensure we can import from the app directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.model import GPT2WithPIIFiltering

def load_model(model_path, device):
    """Loads the trained PII filtering model."""
    print("Loading model configuration...")
    # Initialize model structure
    model = GPT2WithPIIFiltering.from_pretrained("gpt2")
    
    # Load trained weights
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    return model

def redact_text(text, model, tokenizer, device, threshold=0.5):
    """
    Takes raw text and returns the redacted version using the model.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        # 1. Get Embeddings
        inputs_embeds = model.transformer.wte(inputs.input_ids)
        
        # 2. Run Filter
        filter_features = model.pii_filter_layer(inputs_embeds)
        pii_logits = model.pii_classifier(filter_features)
        pii_probs = torch.sigmoid(pii_logits) # (1, seq_len, 1)
        
    # 3. Reconstruct Text with Redaction
    input_ids = inputs.input_ids[0].cpu().numpy()
    probs = pii_probs[0, :, 0].cpu().numpy()
    
    output_tokens = []
    for token_id, prob in zip(input_ids, probs):
        if prob > threshold: # Threshold for PII
            output_tokens.append("[REDACTED]")
        else:
            output_tokens.append(tokenizer.decode([token_id]))
    
    # Join and clean up
    redacted_text = "".join(output_tokens)
    return redacted_text

def main():
    parser = argparse.ArgumentParser(description="Redact PII from an email text file using the trained ModelWatch model.")
    parser.add_argument("input_file", help="Path to the text file containing the email to redact.")
    parser.add_argument("--output", "-o", help="Path to save the redacted output. If not provided, prints to console.")
    parser.add_argument("--model_path", default="models/pii_filter_model.pt", help="Path to the trained model file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for PII detection (default: 0.5).")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model
    model = load_model(args.model_path, device)
    
    # Read Input
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
        
    print(f"Reading from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Redact
    print("Processing...")
    redacted_text = redact_text(text, model, tokenizer, device, args.threshold)
    
    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(redacted_text)
        print(f"Redacted text saved to {args.output}")
    else:
        print("\n--- Redacted Output ---\n")
        print(redacted_text)
        print("\n-----------------------\n")

if __name__ == "__main__":
    main()
