# ModelWatch: Deep Learning PII Redaction

ModelWatch is a sophisticated privacy-preserving tool that uses a custom **Soft-Masking Transformer Architecture** to detect and redact Personally Identifiable Information (PII) from emails and text documents.

Unlike traditional regex or rule-based systems, ModelWatch employs a trained 6-layer Transformer Encoder inserted *before* a GPT-2 backbone to semantically understand context and identify sensitive entities like Names, Emails, Phone Numbers, and IDs.

## Key Features

-   **Custom Architecture**: Extends `GPT2LMHeadModel` with a dedicated PII Filter Stack.
-   **Soft Masking**: Uses a learnable `[REDACTED]` embedding vector that is mathematically blended with original token embeddings based on a predicted PII probability score.
-   **Context-Aware**: Distinguishes between sensitive numbers (e.g., SSN, Phone) and non-sensitive numbers (e.g., indices, quantities) based on sentence context.
-   **End-to-End Training**: The filter layer is trained on synthetic email data to maximize PII detection accuracy.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/michaelmosch15/modelwatch.git
    cd modelwatch
    ```

2.  **Create a virtual environment**:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: This project requires PyTorch and Hugging Face Transformers.*

## Usage

### 1. Redact a New Email (Inference)

To redact PII from a text file using the trained model, use the `redact_email.py` script.

**Command:**
```bash
python redact_email.py <path_to_email_file>
```

**Example:**
```bash
python redact_email.py email_generation/output_emails/email_chain_1.txt --output redacted_email.txt
```

**Options:**
-   `--output`, `-o`: Save the redacted text to a file. If omitted, prints to console.
-   `--threshold`: Probability threshold for redaction (default: 0.5). Lower values make the model more aggressive.

### 2. Training the Model

If you wish to retrain the PII filter on new data:

1.  **Generate Data**: Use the OpenAI-based generator (requires API key).
    ```bash
    python email_generation/generate_emails.py
    ```
2.  **Train**: Run the training loop.
    ```bash
    python scripts/train_pii_filter.py
    ```
    This will save the model weights to `models/pii_filter_model.pt`.

## Project Structure

```
modelwatch/
├── redact_email.py             # Main entry point for redaction
├── app/
│   ├── src/
│   │   └── transformer_filter_layer.py  # Custom Model Architecture
│   └── ...
├── email_generation/
│   ├── generate_emails.py      # Synthetic data generator
│   └── output_emails/          # Generated training data
├── models/
│   └── pii_filter_model.pt     # Trained model weights (Git LFS)
├── scripts/
│   ├── train_pii_filter.py     # Training script
│   └── run_pii_filter.py       # Batch processing script
└── requirements.txt
```
