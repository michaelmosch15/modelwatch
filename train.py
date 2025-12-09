import os
import sys
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.model import GPT2WithPIIFiltering
from src.labeling import detect_pii

EMAIL_DIR = os.path.join(current_dir, "data", "synthetic_emails")
MODEL_SAVE_PATH = os.path.join(current_dir, "models", "pii_filter_model.pt")
BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 1e-4
MAX_LEN = 512

class PIIDataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_len=512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], "r", encoding="utf-8") as f:
            text = f.read()

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        offset_mapping = encoding["offset_mapping"].squeeze()

        labels = torch.zeros_like(input_ids, dtype=torch.float)
        
        pii_items = detect_pii(text)
        
        for item in pii_items:
            start, end = item["start"], item["end"]
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == 0 and tok_end == 0: continue
                
                if tok_start < end and tok_end > start:
                    labels[i] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2WithPIIFiltering.from_pretrained("gpt2")
    model.to(device)
    
    for param in model.transformer.parameters():
        param.requires_grad = False
    
    for param in model.pii_filter_layer.parameters():
        param.requires_grad = True
    for param in model.pii_classifier.parameters():
        param.requires_grad = True
    model.mask_embedding.requires_grad = True

    files = glob.glob(os.path.join(EMAIL_DIR, "*.txt"))
    if not files:
        print(f"No training data found in {EMAIL_DIR}")
        return

    dataset = PIIDataset(files, tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).unsqueeze(-1)

            optimizer.zero_grad()
            
            inputs_embeds = model.transformer.wte(input_ids)
            filter_features = model.pii_filter_layer(inputs_embeds)
            pii_logits = model.pii_classifier(filter_features)
            
            active_loss = attention_mask.view(-1) == 1
            active_logits = pii_logits.view(-1)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            loss = criterion(active_logits, active_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader)}")

    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()
