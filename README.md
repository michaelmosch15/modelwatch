# modelwatch

Local GPT-2 rewrite + PII detection/anonymization toolkit, organized under `app/` with a single CLI entrypoint.

## Quick start

1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
# Optional: spaCy English model for Presidio
python -m spacy download en_core_web_sm
```

3) Run the CLI (from repo root):

```powershell
python app\main.py --help
```

Common examples:

- Detect PII in sample emails:

```powershell
python app\main.py detect --input app\data\sample_inputs.csv --output app\data\sample_detect.jsonl --engine auto
```

- Anonymize emails and write minimal CSV with body only:

```powershell
python app\main.py anonymize --input app\data\sample_inputs.csv --output app\data\sample_outputs_anonymized_simple.csv --simple-output app\data\sample_outputs_body_only.csv --include-subject --engine auto
```

- Rewrite email bodies with GPT-2 and block PII tokens from generation:

```powershell
python app\main.py rewrite --input app\data\sample_inputs.csv --output app\data\sample_rewrites.csv --block-pii --engine auto
```

## Project layout

- `app/main.py` — CLI entrypoint with subcommands: `rewrite`, `detect`, `anonymize`, `apply-pii`.
- `app/src/pii_layer.py` — Hybrid PII detector (HF NER + regex + heuristics) and text anonymizer.
- `app/src/presidio_layer.py` — Presidio-based detector mapped to PERSON/EMAIL/PHONE.
- `app/src/token_layer.py` — TokenReplacementEmbeddingLayer for token-level masking of PII spans.
- `app/scripts/` — Small runnable examples and CSV utilities.
- `app/data/` — Sample inputs and example outputs. Sample outputs are gitignored; `sample_inputs.csv` is tracked.

## Notes

- Set `--engine presidio` to use only Presidio, `--engine hybrid` to use the in-repo detector, or `--engine auto` to union both.
- First Presidio run will load `en_core_web_sm` via spaCy. Install it if you haven't already.
- CPU-only runs are supported; no GPU is required.
