from __future__ import annotations

import argparse
import os
import sys
import json
import pandas as pd
import torch

# Resolve local modules within app/
THIS_DIR = os.path.dirname(__file__)  # .../modelwatch/app
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "src"))
SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, "scripts"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, NoBadWordsLogitsProcessor
from src.pii_layer import detect_pii as hybrid_detect_pii, anonymize
from src.token_layer import TokenReplacementEmbeddingLayer


def build_rewrite_prompt(subject: str, body: str) -> str:
    subject = subject or ""
    body = body or ""
    return (
        "Rewrite the following email in a more formal tone while keeping the meaning the same. "
        "Only output the rewritten email content. Do NOT include the words 'Subject:' or 'Body:' "
        "and do NOT restate the original text.\n\n"
        f"Subject: {subject}\n\n"
        f"Body:\n{body}\n"
    )


def generate_completion(model, tokenizer, device, prompt: str, max_new_tokens: int = 160) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = output_ids[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def cmd_rewrite(args: argparse.Namespace) -> None:
    need_detector = bool(getattr(args, "block_pii", False) or getattr(args, "use_token_layer", False))
    detect_fn, _engine_used = _choose_detector(getattr(args, "engine", "auto")) if need_detector else (None, None)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(args.input)
    start = args.start or 0
    end = args.end if args.end is not None else len(df)
    end = min(end, len(df))
    df_slice = df.iloc[start:end].copy()

    # Optional token replacement layer
    token_layer = None
    if getattr(args, "use_token_layer", False) and detect_fn:
        token_layer = TokenReplacementEmbeddingLayer(tokenizer, model, detect_fn)

    subj_out, body_out = [], []
    for _, row in df_slice.iterrows():
        subject = str(row.get(args.subject_col, ""))
        body = str(row.get(args.body_col, ""))
        prompt = build_rewrite_prompt(subject, body)
        # Optional: block detected PII terms during generation
        if args.block_pii:
            # Detect on original text (subject + body)
            items = detect_fn(f"{subject}\n\n{body}") if detect_fn else []
            # Build bad words token ids
            bad_words_ids = []
            seen = set()
            for it in items:
                text_piece = (it.get("text") or "").strip()
                if not text_piece or text_piece.lower() in seen:
                    continue
                token_ids = tokenizer(text_piece, add_special_tokens=False).input_ids
                if token_ids and len(token_ids) < 10:  # avoid extremely long spans
                    bad_words_ids.append(token_ids)
                    seen.add(text_piece.lower())
            logits_processor = LogitsProcessorList()
            if bad_words_ids:
                logits_processor.append(NoBadWordsLogitsProcessor(bad_words_ids=bad_words_ids, eos_token_id=tokenizer.eos_token_id))

            if token_layer is not None:
                inputs = token_layer.encode_with_replacements(prompt)
            else:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=logits_processor,
                )
            gen_ids = output_ids[0, input_len:]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        else:
            if token_layer is not None:
                inputs = token_layer.encode_with_replacements(prompt)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                gen_ids = output_ids[0, input_len:]
                completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            else:
                completion = generate_completion(model, tokenizer, device, prompt, max_new_tokens=args.max_new_tokens)
        subj_out.append(subject if not args.rewrite_subject else completion)
        body_out.append(completion)

    out_df = df.copy()
    out_df.loc[df_slice.index, "subject_formal"] = subj_out
    out_df.loc[df_slice.index, "body_formal"] = body_out
    out_df.to_csv(args.output, index=False)
    print(f"[rewrite] wrote {args.output} (rows {start}:{end})")


def cmd_detect(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    start = args.start or 0
    end = args.end if args.end is not None else len(df)
    end = min(end, len(df))
    df_slice = df.iloc[start:end].copy()

    body_original, pii_json = [], []
    for _, row in df_slice.iterrows():
        body = str(row.get(args.body_col, ""))
        items = hybrid_detect_pii(body)
        if args.include_subject:
            subj = str(row.get(args.subject_col, ""))
            items += hybrid_detect_pii(subj)
            # de-dup
            seen = set()
            dedup = []
            for it in items:
                key = (it["label"], it["text"], it["start"], it["end"])
                if key not in seen:
                    seen.add(key)
                    dedup.append(it)
            items = dedup
        body_original.append(body)
        pii_json.append(json.dumps(items, ensure_ascii=False))

    out_df = df.copy()
    out_df.loc[df_slice.index, "body_original"] = body_original
    out_df.loc[df_slice.index, "pii"] = pii_json
    out_df.to_csv(args.output, index=False)
    print(f"[detect] wrote {args.output} (rows {start}:{end})")


def _choose_detector(engine_choice: str):
    """Pick a detect_pii(text) callable based on engine choice.
    engine_choice: 'presidio' | 'hybrid' | 'auto'
    'auto' tries presidio, falls back to hybrid.
    """
    choice = (engine_choice or "auto").lower()
    if choice == "hybrid":
        return hybrid_detect_pii, "hybrid"
    if choice in {"presidio", "auto"}:
        try:
            from src.presidio_layer import detect_pii as presidio_detect_pii  # type: ignore
            if choice == "presidio":
                return presidio_detect_pii, "presidio"
            # auto: union of presidio and hybrid results
            def auto_detect(text: str):
                results = []
                try:
                    results.extend(presidio_detect_pii(text))
                except Exception:
                    pass
                try:
                    results.extend(hybrid_detect_pii(text))
                except Exception:
                    pass
                # de-dup
                seen = set()
                dedup = []
                for it in results:
                    key = (it.get("label"), it.get("text"), int(it.get("start", 0)), int(it.get("end", 0)))
                    if key not in seen:
                        seen.add(key)
                        dedup.append(it)
                dedup.sort(key=lambda x: (int(x.get("start", 0)), int(x.get("end", 0))))
                return dedup
            return auto_detect, "auto"
        except Exception:
            if choice == "presidio":
                raise
            return hybrid_detect_pii, "hybrid"
    return hybrid_detect_pii, "hybrid"


def cmd_anonymize(args: argparse.Namespace) -> None:
    detect_fn, engine_used = _choose_detector(getattr(args, "engine", "auto"))
    df = pd.read_csv(args.input)
    start = args.start or 0
    end = args.end if args.end is not None else len(df)
    end = min(end, len(df))
    df_slice = df.iloc[start:end].copy()

    body_original, body_anon, body_pii_json = [], [], []
    subj_original, subj_anon, subj_pii_json = [], [], []
    from_original, from_anon, from_pii_json = [], [], []
    to_original, to_anon, to_pii_json = [], [], []
    pii_map_json = []
    for _, row in df_slice.iterrows():
        # Shared mapping per row so tags are consistent across columns
        counters = {"PERSON": 0, "EMAIL": 0, "PHONE": 0}
        mapping = {}

        def apply(text: str):
            # detect with chosen engine
            items = detect_fn(text)
            # build/extend mapping by first appearance per label
            for it in items:
                lbl = it["label"]
                orig = it["text"]
                if orig not in mapping:
                    counters[lbl] = counters.get(lbl, 0) + 1
                    mapping[orig] = f"[{lbl}_{counters[lbl]}]"
            # replace right-to-left
            out = text
            for it in sorted(items, key=lambda x: x["start"], reverse=True):
                s, e = int(it["start"]), int(it["end"])
                orig = it["text"]
                out = out[:s] + mapping.get(orig, orig) + out[e:]
            return out, items

        # body
        body = str(row.get(args.body_col, ""))
        body_out, body_items = apply(body)
        body_original.append(body)
        body_anon.append(body_out)
        body_pii_json.append(json.dumps(body_items, ensure_ascii=False))

        # subject
        if getattr(args, "include_subject", False):
            subj = str(row.get(args.subject_col, ""))
            subj_out, subj_items = apply(subj)
            subj_original.append(subj)
            subj_anon.append(subj_out)
            subj_pii_json.append(json.dumps(subj_items, ensure_ascii=False))

        # from/to
        if getattr(args, "include_from_to", False):
            fval = str(row.get(args.from_col, row.get("from", "")))
            fout, f_items = apply(fval)
            from_original.append(fval)
            from_anon.append(fout)
            from_pii_json.append(json.dumps(f_items, ensure_ascii=False))

            tval = str(row.get(args.to_col, row.get("to", "")))
            tout, t_items = apply(tval)
            to_original.append(tval)
            to_anon.append(tout)
            to_pii_json.append(json.dumps(t_items, ensure_ascii=False))
        # store unified mapping used for this row
        pii_map_json.append(json.dumps(mapping, ensure_ascii=False))

    out_df = df.copy()
    out_df.loc[df_slice.index, "body_original"] = body_original
    out_df.loc[df_slice.index, "body_anonymized"] = body_anon
    out_df.loc[df_slice.index, "body_pii"] = body_pii_json
    # ensure alignment with row indexes
    import pandas as _pd
    out_df.loc[df_slice.index, "pii_map"] = _pd.Series(pii_map_json, index=df_slice.index)
    out_df.loc[df_slice.index, "pii_engine"] = engine_used

    if getattr(args, "include_subject", False):
        out_df.loc[df_slice.index, "subject_original"] = subj_original
        out_df.loc[df_slice.index, "subject_anonymized"] = subj_anon
        out_df.loc[df_slice.index, "subject_pii"] = subj_pii_json

    if getattr(args, "include_from_to", False):
        out_df.loc[df_slice.index, "from_original"] = from_original
        out_df.loc[df_slice.index, "from_anonymized"] = from_anon
        out_df.loc[df_slice.index, "from_pii"] = from_pii_json
        out_df.loc[df_slice.index, "to_original"] = to_original
        out_df.loc[df_slice.index, "to_anonymized"] = to_anon
        out_df.loc[df_slice.index, "to_pii"] = to_pii_json

    out_df.to_csv(args.output, index=False)
    print(f"[anonymize] wrote {args.output} (rows {start}:{end})")

    # Optional minimal CSV with just key anonymized fields
    if getattr(args, "simple_output", None):
        simple_cols = []
        if "id" in out_df.columns:
            simple_cols.append("id")
        if getattr(args, "include_subject", False) and "subject_anonymized" in out_df.columns:
            simple_cols.append("subject_anonymized")
        simple_cols.append("body_anonymized")
        simple_df = out_df.loc[df_slice.index, simple_cols]
        simple_df.to_csv(args.simple_output, index=False)
        print(f"[anonymize] wrote simple output {args.simple_output} (rows {start}:{end})")


def _anonymize_from_items(text: str, items: list[dict]) -> tuple[str, dict]:
    """Apply anonymization to text using provided PII items only.
    Returns (anonymized_text, mapping) where mapping is original->tag.
    """
    if not text:
        return "", {}
    items_sorted = sorted(items or [], key=lambda x: (x.get("start", 0), x.get("end", 0)))
    # Build mapping per label by first appearance
    counters = {}
    mapping = {}
    for it in items_sorted:
        label = str(it.get("label", ""))
        original = str(it.get("text", ""))
        if original not in mapping:
            counters[label] = counters.get(label, 0) + 1
            mapping[original] = f"[{label}_{counters[label]}]"
    # Replace right-to-left to keep offsets valid
    out = text
    for it in sorted(items_sorted, key=lambda x: x.get("start", 0), reverse=True):
        s = int(it.get("start", 0))
        e = int(it.get("end", s))
        original = str(it.get("text", ""))
        tag = mapping.get(original, original)
        out = out[:s] + tag + out[e:]
    return out, mapping


def cmd_apply_pii(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    start = args.start or 0
    end = args.end if args.end is not None else len(df)
    end = min(end, len(df))
    df_slice = df.iloc[start:end].copy()

    body_original, body_anon, body_mapping_json = [], [], []
    subj_original, subj_anon, subj_mapping_json = [], [], []

    for _, row in df_slice.iterrows():
        body = str(row.get(args.body_col, ""))
        body_pii_raw = row.get(args.pii_col, "[]")
        try:
            body_items = json.loads(body_pii_raw) if isinstance(body_pii_raw, str) else (body_pii_raw or [])
        except Exception:
            body_items = []
        anon_body, body_map = _anonymize_from_items(body, body_items)
        body_original.append(body)
        body_anon.append(anon_body)
        body_mapping_json.append(json.dumps(body_map, ensure_ascii=False))

        if args.include_subject:
            subj = str(row.get(args.subject_col, ""))
            subj_pii_raw = row.get(args.subject_pii_col, "[]")
            try:
                subj_items = json.loads(subj_pii_raw) if isinstance(subj_pii_raw, str) else (subj_pii_raw or [])
            except Exception:
                subj_items = []
            anon_subj, subj_map = _anonymize_from_items(subj, subj_items)
            subj_original.append(subj)
            subj_anon.append(anon_subj)
            subj_mapping_json.append(json.dumps(subj_map, ensure_ascii=False))

    out_df = df.copy()
    out_df.loc[df_slice.index, "body_original"] = body_original
    out_df.loc[df_slice.index, "body_anonymized"] = body_anon
    if args.include_subject:
        out_df.loc[df_slice.index, "subject_original"] = subj_original
        out_df.loc[df_slice.index, "subject_anonymized"] = subj_anon
        out_df.loc[df_slice.index, "subject_mapping"] = subj_mapping_json
    out_df.loc[df_slice.index, "body_mapping"] = body_mapping_json
    out_df.to_csv(args.output, index=False)
    print(f"[apply-pii] wrote {args.output} (rows {start}:{end})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Email workflows: rewrite, detect PII, anonymize.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # rewrite
    pr = sub.add_parser("rewrite", help="Rewrite emails to a more formal tone using local GPT-2")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--subject-col", default="subject")
    pr.add_argument("--body-col", default="body")
    pr.add_argument("--start", type=int, default=0)
    pr.add_argument("--end", type=int)
    pr.add_argument("--model-id", default="gpt2")
    pr.add_argument("--max-new-tokens", type=int, default=160)
    pr.add_argument("--rewrite-subject", action="store_true", help="Also rewrite subject (default: keep subject)")
    pr.add_argument("--block-pii", action="store_true", help="Block detected PII terms during generation")
    pr.add_argument("--engine", choices=["auto", "presidio", "hybrid"], default="auto", help="Detector used when --block-pii is set")
    pr.add_argument("--use-token-layer", action="store_true", help="Replace PII tokens with special token embeddings before generation")
    pr.set_defaults(func=cmd_rewrite)

    # detect
    pdp = sub.add_parser("detect", help="Detect personal identifiers without modifying text")
    pdp.add_argument("--input", required=True)
    pdp.add_argument("--output", required=True)
    pdp.add_argument("--subject-col", default="subject")
    pdp.add_argument("--body-col", default="body")
    pdp.add_argument("--include-subject", action="store_true")
    pdp.add_argument("--start", type=int, default=0)
    pdp.add_argument("--end", type=int)
    pdp.set_defaults(func=cmd_detect)

    # anonymize
    pa = sub.add_parser("anonymize", help="Anonymize personal identifiers with tags")
    pa.add_argument("--input", required=True)
    pa.add_argument("--output", required=True)
    pa.add_argument("--simple-output", help="Optional minimal CSV with anonymized fields only")
    pa.add_argument("--subject-col", default="subject")
    pa.add_argument("--body-col", default="body")
    pa.add_argument("--include-subject", action="store_true")
    pa.add_argument("--include-from-to", action="store_true", help="Also anonymize 'from' and 'to' columns")
    pa.add_argument("--from-col", default="from")
    pa.add_argument("--to-col", default="to")
    pa.add_argument("--engine", choices=["auto", "presidio", "hybrid"], default="auto", help="Detection engine")
    pa.add_argument("--start", type=int, default=0)
    pa.add_argument("--end", type=int)
    pa.set_defaults(func=cmd_anonymize)

    # apply-pii: read PII spans from a JSON column and apply anonymization deterministically
    pap = sub.add_parser("apply-pii", help="Apply anonymization using an existing PII JSON column (no re-detection)")
    pap.add_argument("--input", required=True)
    pap.add_argument("--output", required=True)
    pap.add_argument("--pii-col", default="pii", help="Column containing PII JSON for body")
    pap.add_argument("--body-col", default="body")
    pap.add_argument("--subject-col", default="subject")
    pap.add_argument("--subject-pii-col", default="subject_pii")
    pap.add_argument("--include-subject", action="store_true")
    pap.add_argument("--start", type=int, default=0)
    pap.add_argument("--end", type=int)
    pap.set_defaults(func=cmd_apply_pii)

    return p


def main():
    # If invoked without arguments, run a friendly default: anonymize the sample inputs
    if len(sys.argv) == 1:
        default_input = os.path.join(THIS_DIR, "data", "sample_inputs.csv")
        default_output = os.path.join(THIS_DIR, "data", "sample_outputs_anonymized.csv")
        print("No arguments provided. Running default: anonymize sample_inputs.csv -> sample_outputs_anonymized.csv")
        default_args = argparse.Namespace(
            input=default_input,
            output=default_output,
            subject_col="subject",
            body_col="body",
            include_subject=True,
            include_from_to=True,
            from_col="from",
            to_col="to",
            engine="auto",
            start=0,
            end=None,
        )
        cmd_anonymize(default_args)
        return

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
