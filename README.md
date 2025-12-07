# ModelWatch: Deep Learning PII Redaction

ModelWatch is a privacy-preserving tool that utilizes a Soft-Masking Transformer Architecture to detect and redact Personally Identifiable Information (PII) from text documents.

It employs a trained 6-layer Transformer Encoder inserted before a GPT-2 backbone to identify sensitive entities such as Names, Emails, Phone Numbers, and IDs.

## Key Features

-   **Soft-Masking Architecture**: Extends `GPT2LMHeadModel` with a dedicated PII Filter Stack.
-   **Learnable Masking**: Utilizes a learnable `[REDACTED]` embedding vector blended with original token embeddings based on predicted PII probability.
-   **Context-Aware Detection**: Distinguishes between sensitive and non-sensitive numerical data based on sentence context.
-   **End-to-End Training**: The filter layer is trained on synthetic email data to optimize PII detection.

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

## Usage

### Inference

To redact PII from a text file using the trained model:

```bash
python redact.py --input_file data/synthetic_emails/email_chain_1.txt --output outputs/redacted_email.txt
```

**Arguments:**
-   `--input_file`: Path to the input text file.
-   `--output`, `-o`: Path to save the redacted output.
-   `--threshold`: Probability threshold for redaction (default: 0.5).

### Training

To retrain the PII filter on new data:

1.  **Generate Data**:
    ```bash
    python generate_data.py
    ```
2.  **Train**:
    ```bash
    python train.py
    ```
    This saves the model weights to `models/pii_filter_model.pt`.

## Project Structure

```
modelwatch/
├── src/
│   ├── model.py                # Transformer Architecture definition
│   └── labeling.py             # NER and Regex logic for labeling
├── data/
│   ├── source/                 # Input data
│   └── synthetic_emails/       # Training data
├── models/
│   └── pii_filter_model.pt     # Model weights
├── generate_data.py            # Data generation script
├── train.py                    # Training script
├── redact.py                   # Inference script
└── requirements.txt
```
