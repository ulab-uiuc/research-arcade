import pandas as pd
import os
from pathlib import Path
from typing import Optional


class CSVArxivPaperAuthor:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paper_authors.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paper_authors_table()

    def create_paper_authors_table(self):
        df = pd.DataFrame(columns=['paper_arxiv_id', 'author_id', 'author_sequence'])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paper_authors CSV at {self.csv_path}")

    def _load_data(self):
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()

    def _save_data(self, df):
        df.to_csv(self.csv_path, index=False)

    def insert_paper_author(self, paper_arxiv_id, author_id, author_sequence):
        df = self._load_data()
        conflict = df[
            (df['paper_arxiv_id'] == paper_arxiv_id) &
            (df['author_id'] == author_id)
        ]
        if not conflict.empty:
            return False
        new_row = pd.DataFrame([{
            'paper_arxiv_id': paper_arxiv_id,
            'author_id': author_id,
            'author_sequence': author_sequence
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True

