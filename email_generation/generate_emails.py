import os
import pandas as pd
import random
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

INPUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "data", "Employee DATA(in).csv"))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_emails")
NUM_EMAILS = 500
COMPANY_NAME = "Merck"

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI()

def load_employees(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find employee data at {filepath}")
    return pd.read_csv(filepath)

def generate_email_chain(sender, receiver):
    prompt = (
        f"Generate a realistic corporate email chain (2-3 emails) between two employees at {COMPANY_NAME}.\n\n"
        f"Sender: {sender['First Name']} {sender['Last Name']}, Job Title: {sender['Job Title']}, Email: {sender['Email']}\n"
        f"Receiver: {receiver['First Name']} {receiver['Last Name']}, Job Title: {receiver['Job Title']}, Email: {receiver['Email']}\n\n"
        "Context: The emails should be relevant to their job titles and the pharmaceutical industry context of Merck. "
        "Include realistic corporate jargon, project references, and naturally occurring personal information "
        "(like a phone number, a specific date, an office location, or a mention of an upcoming leave).\n\n"
        "Format the output as a plain text email chain.\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a corporate email generator assistant. Generate realistic business emails."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating email: {e}")
        return None

def current_count(output_dir):
    # Count existing files that match the naming pattern
    count = 0
    for name in os.listdir(output_dir):
        if name.startswith("email_chain_") and name.endswith(".txt"):
            try:
                idx = int(name.replace("email_chain_", "").replace(".txt", ""))
                count = max(count, idx)
            except ValueError:
                continue
    return count

def main():
    print(f"Loading employee data from {INPUT_CSV}...")
    try:
        df = load_employees(INPUT_CSV)
    except Exception as e:
        print(e)
        return

    # Determine how many are already generated and resume
    already = current_count(OUTPUT_DIR)
    if already >= NUM_EMAILS:
        print(f"Already generated {already} emails. Nothing to do.")
        return

    remaining = NUM_EMAILS - already
    print(f"Resuming generation: {already} existing. Generating {remaining} more email chains...")

    for i in tqdm(range(already, NUM_EMAILS)):
        sample = df.sample(2)
        sender = sample.iloc[0]
        receiver = sample.iloc[1]

        email_content = generate_email_chain(sender, receiver)

        if email_content:
            filename = os.path.join(OUTPUT_DIR, f"email_chain_{i+1}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(email_content)

    print(f"Done! Generated emails are in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
