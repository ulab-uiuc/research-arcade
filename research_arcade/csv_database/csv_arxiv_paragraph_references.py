import pandas as pd
import os
from pathlib import Path


class CSVArxivParagraphReference:
    def __init__(self, csv_dir):
        csv_path = f"{csv_dir}/paragraph_references.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paragraph_references_table()

    def create_paragraph_references_table(self):
        df = pd.DataFrame(columns=[
            'id', 'paragraph_id', 'paper_section',
            'paper_arxiv_id', 'reference_label', 'reference_type'
        ])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paragraph_references CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_paragraph_reference(self, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type):
        df = self._load_data()
        new_id = df['id'].max() + 1 if not df.empty else 1
        new_row = pd.DataFrame([{
            'id': new_id, 'paragraph_id': paragraph_id,
            'paper_section': paper_section, 'paper_arxiv_id': paper_arxiv_id,
            'reference_label': reference_label, 'reference_type': reference_type
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id



