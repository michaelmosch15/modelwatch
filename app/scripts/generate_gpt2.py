import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt(subject: str, body: str) -> str:
    subject = subject or ""
    body = body or ""
    return (
       ""
       f"Subject: {subject}\n\n"
       f"Body:\n{body}\n"
    )


def generate_completion(model, tokenizer, device, prompt: str, max_new_tokens: int = 120) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding is faster and more stable on CPU
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Slice off the prompt tokens to get only the completion
    gen_ids = output_ids[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")

    # Configuration: set input/output and row range here
    INPUT_CSV = os.path.join(data_dir, "sample_inputs.csv")
    OUTPUT_CSV = os.path.join(data_dir, "sample_outputs_formal.csv")
    SUBJECT_COL = "subject"
    BODY_COL = "body"
    # Rows to process (0-based, end exclusive). Set to None to process all.
    ROW_START = 0
    ROW_END = 5  # e.g., process rows [0, 5)
    MAX_NEW_TOKENS = 240  # keep tight for CPU speed; raise if you need longer outputs

    # Load model once
    model_id = "gpt2"  # small and fast on CPU
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Avoid repeated warnings; GPT-2 has no pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # Prefer eager attention on CPU to avoid SDPA overheads
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Read CSV
    df = pd.read_csv(INPUT_CSV)

    # Determine range
    if ROW_START is None:
        ROW_START = 0
    if ROW_END is None or ROW_END > len(df):
        ROW_END = len(df)

    df_slice = df.iloc[ROW_START:ROW_END].copy()

    outputs_subject = []
    outputs_body = []

    for _, row in df_slice.iterrows():
        subject = str(row.get(SUBJECT_COL, ""))
        body = str(row.get(BODY_COL, ""))
        prompt = build_prompt(subject, body)
        completion = generate_completion(model, tokenizer, device, prompt, max_new_tokens=MAX_NEW_TOKENS)

        # Try to split out a subject/body if the model formats it; otherwise, treat whole as body
        subj_out = subject
        body_out = completion
        outputs_subject.append(subj_out)
        outputs_body.append(body_out)

    # Attach outputs back to the slice
    df_slice["subject_formal"] = outputs_subject
    df_slice["body_formal"] = outputs_body

    # Merge back into original DataFrame or just write the slice
    out_df = df.copy()
    out_df.loc[df_slice.index, "subject_formal"] = df_slice["subject_formal"]
    out_df.loc[df_slice.index, "body_formal"] = df_slice["body_formal"]

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote formalized emails to {OUTPUT_CSV} (rows {ROW_START}:{ROW_END})")


if __name__ == "__main__":
    main()
