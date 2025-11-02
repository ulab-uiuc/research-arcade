import pandas as pd
import os
from pathlib import Path
from typing import Optional

class CSVArxivPaperCategory:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paper_category.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paper_category_table()

    def create_paper_category_table(self):
        df = pd.DataFrame(columns=['paper_arxiv_id', 'category_id'])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paper_category CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_paper_category(self, paper_arxiv_id, category_id):
        df = self._load_data()
        conflict = df[
            (df['paper_arxiv_id'] == paper_arxiv_id) &
            (df['category_id'] == category_id)
        ]
        if not conflict.empty:
            return False
        new_row = pd.DataFrame([{
            'paper_arxiv_id': paper_arxiv_id,
            'category_id': category_id
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True

    def get_all_categories(self):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()
