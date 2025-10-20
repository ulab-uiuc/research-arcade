import pandas as pd
import os
from pathlib import Path
import json


class CSVArxivCitation:
    def __init__(self, csv_dir):
        csv_path = f"{csv_dir}/citations.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_citations_table()

    def create_citations_table(self):
        df = pd.DataFrame(columns=[
            'id', 'citing_arxiv_id', 'cited_arxiv_id',
            'bib_title', 'bib_key', 'author_cited_paper',
            'citing_sections', 'citing_paragraphs'
        ])
        df.to_csv(self.csv_path, index=False)
        print(f"Created citations CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_citation(self, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, author_cited_paper, citing_sections=None):
        if citing_arxiv_id == cited_arxiv_id:
            return False
        df = self._load_data()
        conflict = df[
            (df['citing_arxiv_id'] == citing_arxiv_id) &
            (df['cited_arxiv_id'] == cited_arxiv_id)
        ]
        if not conflict.empty:
            return False
        new_id = df['id'].max() + 1 if not df.empty else 1
        citing_sections_str = json.dumps(citing_sections) if citing_sections else '[]'
        new_row = pd.DataFrame([{
            'id': new_id, 'citing_arxiv_id': citing_arxiv_id,
            'cited_arxiv_id': cited_arxiv_id, 'bib_title': bib_title,
            'bib_key': bib_key, 'author_cited_paper': author_cited_paper,
            'citing_sections': citing_sections_str, 'citing_paragraphs': '[]'
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True

