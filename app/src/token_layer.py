from __future__ import annotations

from typing import Dict, List, Tuple, Callable

import torch


class TokenReplacementEmbeddingLayer:
    """
    Replaces tokens overlapping detected PII spans with special tokens
    before feeding inputs to the language model.

    - Adds special tokens [PERSON], [EMAIL], [PHONE] to the tokenizer and
      resizes model embeddings.
    - Uses character-offset alignment (offset_mapping) to map detected spans
      to token indices, then substitutes token ids.
    """

    def __init__(
        self,
        tokenizer,
        model,
        detect_fn: Callable[[str], List[Dict]],
        labels: Tuple[str, ...] = ("PERSON", "EMAIL", "PHONE"),
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.detect_fn = detect_fn
        self.labels = labels

        # Ensure special tokens exist in the tokenizer
        add_tokens = [f"[{l}]" for l in labels]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
        if added:
            # Resize model embeddings to accommodate new tokens
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Cache ids for quick lookup
        self.label_token_ids: Dict[str, int] = {}
        for l in labels:
            tok = f"[{l}]"
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            self.label_token_ids[l] = tok_id

    def _spans_from_detector(self, text: str) -> List[Tuple[int, int, str]]:
        items = self.detect_fn(text or "") or []
        spans: List[Tuple[int, int, str]] = []
        for it in items:
            try:
                s = int(it.get("start", 0))
                e = int(it.get("end", 0))
                lbl = str(it.get("label", ""))
                if s < e and lbl in self.labels:
                    spans.append((s, e, lbl))
            except Exception:
                continue
        spans.sort(key=lambda x: (x[0], x[1]))
        return spans

    @staticmethod
    def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return max(a_start, b_start) < min(a_end, b_end)

    def encode_with_replacements(self, text: str, truncation: bool = True, max_length: int | None = None):
        # Get offsets mapping from fast tokenizer
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=truncation,
            max_length=max_length,
        )
        input_ids: List[int] = enc["input_ids"]
        attention_mask: List[int] = enc.get("attention_mask", [1] * len(input_ids))
        offsets = enc.get("offset_mapping", None)

        if offsets is None:
            # Fallback: no offsets, return plain encoding
            return {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }

        spans = self._spans_from_detector(text)
        if spans:
            # For each token, if overlaps any span, replace with the label token id
            for i, (s, e) in enumerate(offsets):
                # Special tokens sometimes have (0,0) offsets; skip them
                if s == e:
                    continue
                for (ps, pe, lbl) in spans:
                    if self._overlap(s, e, ps, pe):
                        rep_id = self.label_token_ids.get(lbl)
                        if rep_id is not None:
                            input_ids[i] = rep_id
                        break  # first matching span wins

        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }
