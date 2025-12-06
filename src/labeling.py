"""
PII Transform Layer: detect personal identifiers (names, emails, phones) and anonymize text.

Exports:
- detect_pii(text) -> list of dicts {label, text, start, end}
- anonymize(text) -> (text_with_tags, mapping, items)

Notes:
- Uses a Hugging Face NER model for PERSON names (dslim/bert-base-NER) on CPU.
- Uses regex for EMAIL and PHONE.
- Mapping is stable per-call in order of first appearance, separately by label type.
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

    # Phone (US-leaning)
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

    # Date of Birth (DD-MM-YY based on example 17-08-21)
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

    # Unique Identifier (XXX-XX-XXXX based on example 089-45-9486)
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

    # Sex (Male/Female)
    SEX_RE = re.compile(r"\b(Male|Female)\b", re.IGNORECASE)

    def detect_sex(text: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for m in SEX_RE.finditer(text or ""):
            items.append({
                "label": "SEX",
                "text": m.group(0),
                "start": int(m.start()),
                "end": int(m.end()),
            })
        return items

    detectors.append(detect_sex)

    # Signature name detector (e.g., "Thanks,\nAlice" or "Best,\nJohn Doe")
    SALUTATIONS = {"thanks", "regards", "cheers", "best", "sincerely", "kind regards", "warm regards", "best regards"}
    SIG_NAME_RE = re.compile(r"(?im)^[ \t]*([A-Z][a-z]+(?:[ \t][A-Z][a-z]+){0,2})[ \t]*$")

    def detect_signature(text: str):
        t = text or ""
        candidates = list(SIG_NAME_RE.finditer(t))
        if not candidates:
            return []
        # choose last candidate whose previous non-empty line is a salutation
        for m in reversed(candidates):
            line_start = t.rfind("\n", 0, m.start()) + 1
            # find previous line content
            prev_end = line_start - 1
            if prev_end < 0:
                continue
            prev_start = t.rfind("\n", 0, prev_end) + 1
            prev_line = t[prev_start:prev_end].strip().lower().rstrip(',')
            if prev_line in SALUTATIONS:
                name = m.group(1)
                return [{
                    "label": "PERSON",
                    "text": name,
                    "start": int(m.start(1)),
                    "end": int(m.end(1)),
                }]
        return []

    detectors.append(detect_signature)

    # Trailing single-line name (common in signatures)
    TRAIL_RE = re.compile(r"(?m)^\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\s*$")

    def detect_trailing_name(text: str):
        t = text or ""
        matches = list(TRAIL_RE.finditer(t))
        if not matches:
            return []
        m = matches[-1]
        return [{
            "label": "PERSON",
            "text": m.group(1),
            "start": int(m.start(1)),
            "end": int(m.end(1)),
        }]

    detectors.append(detect_trailing_name)

    return detectors


_DETECTORS = None


def _get_detectors():
    global _DETECTORS
    if _DETECTORS is None:
        _DETECTORS = _build_detectors()
    return _DETECTORS


def detect_pii(text: str) -> List[Dict[str, Any]]:
    """Run all detectors and merge results, sorted by start index."""
    results: List[Dict[str, Any]] = []
    seen = set()
    for det in _get_detectors():
        for item in det(text or ""):
            key = (item["label"], item["text"], item["start"], item["end"])
            if key not in seen:
                seen.add(key)
                results.append(item)
    results.sort(key=lambda x: (x["start"], x["end"]))
    return results


def anonymize(text: str) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    """
    Replace detected spans with stable tags per label type.
    Example tags: [PERSON_1], [EMAIL_1], [PHONE_1]
    Returns: (anonymized_text, mapping, items)
    mapping maps original string values to their tag.
    """
    items = detect_pii(text)
    # Build separate counters so PERSON_1 and EMAIL_1 are independent
    counters = {"PERSON": 1, "EMAIL": 1, "PHONE": 1}
    mapping: Dict[str, str] = {}

    # For overlapping spans, replace from right to left by start offset
    out = text or ""
    for it in sorted(items, key=lambda x: x["start"], reverse=True):
        label = it["label"]
        span_text = it["text"]
        if span_text not in mapping:
            tag = f"[{label}_{counters.get(label, 1)}]"
            mapping[span_text] = tag
            counters[label] = counters.get(label, 1) + 1
        out = out[: it["start"]] + mapping[span_text] + out[it["end"] :]
    return out, mapping, items
