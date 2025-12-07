"""
PII Transform Layer: detect personal identifiers (names, emails, phones) and anonymize text.

Exports:
- detect_pii(text) -> list of dicts {label, text, start, end}
- anonymize(text) -> (text_with_tags, mapping, items)
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple
import re

from transformers import pipeline


def _build_detectors():
    detectors = []

    # HF NER for PERSON
    ner = pipeline(
        "token-classification",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=-1,  # CPU
    )

    def detect_ner(text: str) -> List[Dict[str, Any]]:
        out = ner(text or "")
        items: List[Dict[str, Any]] = []
        for ent in out:
            label = ent.get("entity_group", "").upper()
            if label in {"PER", "PERSON"}:
                start = int(ent["start"])
                end = int(ent["end"])
                span_text = (text or "")[start:end]
                items.append({"label": "PERSON", "text": span_text, "start": start, "end": end})
        return items

    detectors.append(detect_ner)

    # Email
    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    def detect_email(text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for m in EMAIL_RE.finditer(text or ""):
            items.append({
                "label": "EMAIL",
                "text": m.group(0),
                "start": int(m.start()),
                "end": int(m.end()),
            })
        return items

    detectors.append(detect_email)

    # Phone
    PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}")

    def detect_phone(text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for m in PHONE_RE.finditer(text or ""):
            items.append({
                "label": "PHONE",
                "text": m.group(0),
                "start": int(m.start()),
                "end": int(m.end()),
            })
        return items

    detectors.append(detect_phone)

    # Date of Birth
    DOB_RE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")

    def detect_dob(text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for m in DOB_RE.finditer(text or ""):
            items.append({
                "label": "DOB",
                "text": m.group(0),
                "start": int(m.start()),
                "end": int(m.end()),
            })
        return items

    detectors.append(detect_dob)

    # Unique Identifier
    ID_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    def detect_id(text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for m in ID_RE.finditer(text or ""):
            items.append({
                "label": "ID",
                "text": m.group(0),
                "start": int(m.start()),
                "end": int(m.end()),
            })
        return items

    detectors.append(detect_id)

    return detectors


_DETECTORS = _build_detectors()


def detect_pii(text: str) -> List[Dict[str, Any]]:
    """
    Run all detectors on text and return a merged list of PII items.
    """
    all_items = []
    for det in _DETECTORS:
        all_items.extend(det(text))

    # Sort by start position
    all_items.sort(key=lambda x: x["start"])
    
    # Merge overlapping spans
    merged = []
    if not all_items:
        return merged
        
    curr = all_items[0]
    for next_item in all_items[1:]:
        if next_item["start"] < curr["end"]:
            # Overlap: take the longer one
            if next_item["end"] > curr["end"]:
                curr = next_item
        else:
            merged.append(curr)
            curr = next_item
    merged.append(curr)
    
    return merged


def anonymize(text: str) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    """
    Replace PII with tags like [PERSON_1], [EMAIL_2].
    Returns:
      - anonymized text
      - mapping {tag: original_value}
      - list of PII items found
    """
    items = detect_pii(text)
    
    # Create mapping
    mapping = {}
    
    # First pass: assign tags
    tagged_items = []
    counters = {}
    
    for item in items:
        label = item["label"]
        counters[label] = counters.get(label, 0) + 1
        tag = f"[{label}_{counters[label]}]"
        mapping[tag] = item["text"]
        
        tagged_items.append({
            **item,
            "tag": tag
        })
        
    # Second pass: replace in text (reverse order)
    result_text = list(text)
    for item in reversed(tagged_items):
        start, end = item["start"], item["end"]
        tag = item["tag"]
        result_text[start:end] = list(tag)
        
    return "".join(result_text), mapping, items
