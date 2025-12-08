from pathlib import Path
import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import Trainer, TrainingArguments
import torch
import warnings
from typing import List, Dict, Tuple, Optional
import re
from tqdm import tqdm
import phonenumbers
from phonenumbers import NumberParseException
warnings.filterwarnings('ignore')

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

def main():
    detector = PIIDetector()
    input_csv_path = "./data/source/Employee DATA(in).csv"
    try:
        input_df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"File not found: {input_csv_path}")
        return
    output_df, detections = detector.process_data(input_df)
    print(output_df.head())

class PIIDetector:
    '''
    PII Detector with two components (for now):
    - 1. BERT NERT Model
      Contains the following 4 categories:
      - PER (Person)
      - ORG (Organization)
      - LOC (Location)
      - MISC (Miscellaneous)
    - 2. Regex Matching for Edge Cases
      - Email
      - Phone
      - SSN
      - Zip Code
      - DOB
    '''

    def __init__(self, model_name="dslim/bert-base-NER"):
        # Initialize BERT Model:
        print(f"Initializing model: {model_name}")
        # Pretrained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple", # This combines subwords
            batch_size=32
        )
        # Initialize common Regex patterns:
        self.regex_patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'''
                \b
                (?:\+?1[\s\-\.]?)?        # optional country code
                (?:\(?\d{3}\)?[\s\-\.]?)  # area code
                \d{3}[\s\-\.]?\d{4}        # main number
                (?:\s*(?:x|ext\.?|\#)\s*\d+)? # optional extension
                \b
            ''',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'ZIP_CODE': r'\b\d{5}(?:-\d{4})?\b',
            'DATE_OF_BIRTH': r'\b\d{2}-\d{2}-\d{2}\b'
        }
        print("Model loaded!")

    # Normalize phone numbers
    def normalize_phone(self, phone: str) -> str:
        if not phone or not isinstance(phone, str):
            return phone

        # Remove common text labels
        phone = phone.lower().replace("ext", "x").replace("extension", "x")

        # Extract extension if present
        ext_match = re.search(r'(x\d+)$', phone)
        extension = ext_match.group(1) if ext_match else ''

        # Remove everything except digits and +
        base = re.sub(r'[^\d+]', '', phone.replace(extension, ''))

        try:
            # Handle 001 prefix
            if base.startswith("001"):
                base = "+" + base[3:]

            # Add US default if missing country code
            if not base.startswith('+'):
                base = "+1" + base

            number = phonenumbers.parse(base, None)

            if phonenumbers.is_valid_number(number):
                formatted = phonenumbers.format_number(
                    number,
                    phonenumbers.PhoneNumberFormat.E164
                )
                return formatted + extension
            else:
                return phone

        except NumberParseException:
            return phone


    # Detect with BERT
    def bert_ner(self, input: str) -> List[Dict]:
        if not input or not isinstance(input, str):
            return []
        pii_entities = []
        try:
            entities = self.ner_pipeline(input)
            for ent in entities:
                # Append only PII entities
                if ent['entity_group'] in ['PER', 'PERSON', 'ORG', 'MISC']:
                    pii_entities.append({
                        'entity': ent['word'],
                        'type': ent['entity_group'],
                        'score': ent['score'],
                        'start': ent['start'],
                        'end': ent['end'],
                        'method': 'NER'
                    })
            return pii_entities
        except Exception as e:
            print(f'Error processing entities: {e}')
            return []

    # Detect with Regex
    def regex(self, input: str) -> List[Dict]:
        if not input or not isinstance(input, str):
            return []
        pii_entities = []
        for category, pattern in self.regex_patterns.items():
            matches = re.finditer(pattern, input, re.VERBOSE | re.IGNORECASE)
            for match in matches:
                original_value = match.group()
                entity_value = original_value
                if category == "PHONE":
                    entity_value = self.normalize_phone(original_value)
                pii_entities.append({
                    'entity': original_value,
                    'type': category,
                    'score': 1.0,
                    'start': match.start(),
                    'end': match.end(),
                    'method': 'Regex'
                })
        return pii_entities

    # Combine the results
    def combine_results(self, input: str) -> List[Dict]:
        bert_res = self.bert_ner(input)
        regex_res = self.regex(input)
        results = bert_res + regex_res
        # Sort the results by start position
        results.sort(key=lambda x: x['start'])
        return results

    # Redact the resulting entities
    def redact_input(self, input: str, entities: List[Dict]) -> str:
        if not input or not entities:
            return input
        # Sorted to maintain order by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        redacted = input
        for ent in sorted_entities:
            start, end = ent['start'], ent['end']
            # Make sure to get the full word
            if ent['type'] == "PHONE":
                # Replace all occurrences of the original number
                redacted = redacted[:start] + "[PHONE]" + redacted[end:]
            else:
              while start > 0 and input[start-1].isalnum():
                  start -= 1
              while end < len(input) and input[end].isalnum():
                  end += 1
              # Mask it with the type (PER, LOC, etc.)
              mask = f"[{ent['type']}]"
              redacted = redacted[:ent['start']] + mask + redacted[ent['end']:]
        return redacted

    # Data Processing Pipeline
    def process_data(self, input_df: pd.DataFrame, text_columns: Optional[List[str]] = None, redact: bool = True):
        '''
        Args:
        - input_df: Input dataframe (CSV, SQL table)
        - text_columns: List of columns to scan
        - redact: Redacts text columns if true
        '''
        # If columns aren't specified, choose those containing strings (objects)
        if text_columns is None:
            text_columns = input_df.select_dtypes(include=['object']).columns.tolist()

        print(f'Searching columns: {text_columns}')

        pii_detections = []
        for i, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing rows"):
            row_pii = {
                'row_idx': i,
                'pii_detected': False,
                'pii_types': set(),
                'pii_count': 0,
                'details': []
            }
            for col in text_columns:
                # Get the data from each column we want to detect
                text = str(row[col]) if pd.notna(row[col]) else ""
                if not text or text.lower() == 'nan':
                    continue
                
                # Force full redaction of names
                if col in ("First Name", "Last Name"):
                  redacted_col_name = f'{col}_redacted'
                  input_df.at[i, redacted_col_name] = "[PER]"
                  row_pii['pii_detected'] = True
                  row_pii['pii_types'].add('PER')
                  row_pii['pii_count'] += 1

                  row_pii['details'].append({
                      'column': col,
                      'entity': text,
                      'type': 'PER',
                      'method': 'ColumnRule',
                      'score': 1.0
                  })

                  continue
                # Detect the entities, and update the pii for each row
                entities = self.combine_results(text)
                if entities:
                    row_pii['pii_detected'] = True
                    row_pii['pii_count'] += len(entities)
                    for ent in entities:
                        # Update the row_pii with information
                        row_pii['pii_types'].add(ent['type'])
                        row_pii['details'].append({
                            'column': col,
                            'entity': ent['entity'],
                            'type': ent['type'],
                            'method': ent['method'],
                            'score': ent['score']
                        })
                    # Redact if necessary
                    if redact:
                        redacted_col_name = f'{col}_redacted'
                        input_df.at[i, redacted_col_name] = self.redact_input(text, entities)
            # Convert from set to list after
            row_pii['pii_types'] = list(row_pii['pii_types'])
            pii_detections.append(row_pii)
        # Add detection results to dataframe
        input_df['pii_detected'] = [d['pii_detected'] for d in pii_detections]
        input_df['pii_types'] = [', '.join(d['pii_types']) for d in pii_detections]
        input_df['pii_count'] = [d['pii_count'] for d in pii_detections]
        # Normalize phone redactions
        if 'Phone_redacted' in input_df.columns:
            input_df['Phone_redacted'] = input_df['Phone_redacted'].apply(
                lambda x: '[PHONE]' if isinstance(x, str) and '[PHONE]' in x else x
            )
        # Print summary
        self.print_summary(pii_detections)
        # Save as a CSV
        input_df.to_csv("./pii_results.csv", index=False)
        print("Saved results to ./pii_results.csv")
        return input_df, pii_detections

    # Print summary stats
    def print_summary(self, pii_detections: List[Dict]):
        total_rows = len(pii_detections)
        rows_with_pii = sum(1 for d in pii_detections if d['pii_detected'])
        total_pii = sum(d['pii_count'] for d in pii_detections)

        # Count by type 
        type_counts = {}
        for detection in pii_detections:
            for pii_type in detection['pii_types']:
                type_counts[pii_type] = type_counts.get(pii_type, 0) + 1

        print("\n" + "="*60)
        print("PII DETECTION SUMMARY")
        print("="*60)
        print(f"Total rows processed: {total_rows}")
        print(f"Rows with PII: {rows_with_pii} ({rows_with_pii/total_rows*100:.1f}%)")
        print(f"Total PII entities found: {total_pii}")
        print("\nPII by type:")
        for pii_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pii_type}: {count}")
        print("="*60)

if __name__ == "__main__":
    main()