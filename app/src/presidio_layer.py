"""
Presidio-based PII detection using the same rules as microsoft/presidio.

Exports:
- detect_pii(text) -> list of {label, text, start, end}

Notes:
- Requires presidio-analyzer and a compatible NLP engine (spacy with en_core_web_sm).
- If dependencies are missing, importing this module will raise ImportError.
"""

from __future__ import annotations

from typing import List, Dict, Any


def _build_analyzer():
    # Lazy import to avoid hard dependency at import time
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Configure spaCy small English model; caller should have it installed
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    return analyzer


_ANALYZER = None


def _get_analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = _build_analyzer()
    return _ANALYZER


def detect_pii(text: str) -> List[Dict[str, Any]]:
    analyzer = _get_analyzer()
    res = analyzer.analyze(text=text or "", language="en")
    items: List[Dict[str, Any]] = []
    for ent in res:
        etype = ent.entity_type.upper()
        # Map Presidio entity types to our labels
        if etype == "EMAIL_ADDRESS":
            label = "EMAIL"
        elif etype == "PHONE_NUMBER":
            label = "PHONE"
        else:
            label = etype
        # Only keep relevant common types
        if label in {"PERSON", "EMAIL", "PHONE"}:
            span_text = (text or "")[ent.start : ent.end]
            items.append({
                "label": label,
                "text": span_text,
                "start": int(ent.start),
                "end": int(ent.end),
            })
    # Sort by start
    items.sort(key=lambda x: (x["start"], x["end"]))
    return items
