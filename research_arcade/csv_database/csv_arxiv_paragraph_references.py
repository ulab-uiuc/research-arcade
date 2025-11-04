import pandas as pd
import os
from pathlib import Path
from typing import Optional

class CSVArxivParagraphReference:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paragraph_references.csv"
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

    def get_all_paragraph_references(self):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()


    def get_paragraph_neighboring_references(self, paragraph_id: int) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paragraph_id'] == paragraph_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def get_reference_neighboring_paragraphs(self, reference_id: int) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['id'] == reference_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def delete_paragraph_reference_by_id(self, paragraph_id: int, reference_id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = (df['paragraph_id'] == paragraph_id) & (df['id'] == reference_id)
        
        if not mask.any():
            return False
        
        df = df[~mask]
        self._save_data(df)
        
        return True


    def delete_paragraph_reference_by_paragraph_id(self, paragraph_id: int) -> int:

        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['paragraph_id'] == paragraph_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count

    def delete_paragraph_reference_by_reference_id(self, reference_id: int) -> int:

        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['id'] == reference_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count
