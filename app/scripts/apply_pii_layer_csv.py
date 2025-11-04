import os
import sys
import json
import pandas as pd

this_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(this_dir, ".."))

from app.src.pii_layer import anonymize 

def main():
    data_dir = os.path.join(base_dir, "data")
    # Config
    INPUT_CSV = os.path.join(data_dir, "sample_inputs.csv")
    OUTPUT_CSV = os.path.join(data_dir, "sample_outputs_pii_layer.csv")
    SUBJECT_COL = "subject"
    BODY_COL = "body"
    ROW_START = 0
    ROW_END = None

    df = pd.read_csv(INPUT_CSV)
    if ROW_START is None:
        ROW_START = 0
    if ROW_END is None or ROW_END > len(df):
        ROW_END = len(df)
    df_slice = df.iloc[ROW_START:ROW_END].copy()

    body_original = []
    body_anon = []
    pii_col = []

    for _, row in df_slice.iterrows():
        body = str(row.get(BODY_COL, ""))
        body_original.append(body)
        anon_text, mapping, items = anonymize(body)
        body_anon.append(anon_text)
        pii_col.append(json.dumps(items, ensure_ascii=False))

    df_slice["body_original"] = body_original
    df_slice["body_anonymized"] = body_anon
    df_slice["pii"] = pii_col

    out_df = df.copy()
    out_df.loc[df_slice.index, ["body_original", "body_anonymized", "pii"]] = df_slice[
        ["body_original", "body_anonymized", "pii"]
    ]

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote PII-layer outputs to {OUTPUT_CSV} (rows {ROW_START}:{ROW_END})")


if __name__ == "__main__":
    main()
