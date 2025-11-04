"""
Detect personal identifiers (names, emails, phone numbers) in email data without changing the text.

Outputs a CSV that includes:
- The original email body (and can include subject if desired)
- A `pii` column containing a JSON list of detected items with their type and text.

Configure row range and columns at the top of main().
Runs fully on CPU.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any

from transformers import pipeline
import re


def build_detectors(use_ner: bool = True):
    """Build detection functions returning list of dicts: {label, text, start, end}."""
    detectors = []

    if use_ner:
        ner = pipeline(
            "token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1,  # CPU
        )

        def detect_ner(text: str) -> List[Dict[str, Any]]:
            out = ner(text)
            items: List[Dict[str, Any]] = []
            for ent in out:
                label = ent.get("entity_group", "").upper()
                if label in {"PER", "PERSON"}:
                    items.append(
                        {
                            "label": "PERSON",
                            "text": text[ent["start"] : ent["end"]],
                            "start": int(ent["start"]),
                            "end": int(ent["end"]),
                        }
                    )
            return items

        detectors.append(detect_ner)

    # Email regex
    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    def detect_email(text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for m in EMAIL_RE.finditer(text):
            items.append(
                {
                    "label": "EMAIL",
                    "text": m.group(0),
                    "start": int(m.start()),
                    "end": int(m.end()),
                }
            )
        return items

    detectors.append(detect_email)

    # Phone regex (US-leaning)
    PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}")

    def detect_phone(text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for m in PHONE_RE.finditer(text):
            items.append(
                {
                    "label": "PHONE",
                    "text": m.group(0),
                    "start": int(m.start()),
                    "end": int(m.end()),
                }
            )
        return items

    detectors.append(detect_phone)

    return detectors


def detect_pii(text: str, detectors) -> List[Dict[str, Any]]:
    """Run all detectors and merge results, de-duplicating by (label, text, start, end)."""
    seen = set()
    results: List[Dict[str, Any]] = []
    for det in detectors:
        for item in det(text or ""):
            key = (item["label"], item["text"], item["start"], item["end"])
            if key not in seen:
                seen.add(key)
                results.append(item)
    # Sort by start index for readability
    results.sort(key=lambda x: (x["start"], x["end"]))
    return results


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")

    # Configuration
    INPUT_CSV = os.path.join(data_dir, "sample_inputs.csv")
    OUTPUT_CSV = os.path.join(data_dir, "sample_outputs_pii.csv")
    SUBJECT_COL = "subject"  # can include in PII detection if desired
    BODY_COL = "body"
    INCLUDE_SUBJECT = False   # set True if you want to scan subject as well
    ROW_START = 0
    ROW_END = None  # None = process to end

    df = pd.read_csv(INPUT_CSV)
    if ROW_START is None:
        ROW_START = 0
    if ROW_END is None or ROW_END > len(df):
        ROW_END = len(df)
    df_slice = df.iloc[ROW_START:ROW_END].copy()

    detectors = build_detectors(use_ner=True)

    pii_col = []
    body_original_col = []
    for _, row in df_slice.iterrows():
        subj = str(row.get(SUBJECT_COL, ""))
        body = str(row.get(BODY_COL, ""))

        body_original_col.append(body)

        text_to_scan = body
        results = detect_pii(text_to_scan, detectors)

        if INCLUDE_SUBJECT and subj:
            results += detect_pii(subj, detectors)
            # de-dup again after combining
            seen = set()
            dedup = []
            for item in results:
                key = (item["label"], item["text"], item["start"], item["end"])
                if key not in seen:
                    seen.add(key)
                    dedup.append(item)
            results = dedup

        pii_col.append(json.dumps(results, ensure_ascii=False))

    # Attach columns
    df_slice["body_original"] = body_original_col
    df_slice["pii"] = pii_col

    # Merge into full df at indices and save
    out_df = df.copy()
    out_df.loc[df_slice.index, "body_original"] = df_slice["body_original"]
    out_df.loc[df_slice.index, "pii"] = df_slice["pii"]

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote PII detections to {OUTPUT_CSV} (rows {ROW_START}:{ROW_END})")


if __name__ == "__main__":
    main()
