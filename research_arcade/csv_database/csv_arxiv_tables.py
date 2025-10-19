import pandas as pd
import os
from typing import List, Optional, Tuple
from pathlib import Path
import json


class CSVArxivTable:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_tables_table()
        # self.arxiv_crawler = ArxivCrawler()
    
    
    def create_tables_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'paper_arxiv_id', 'path', 'caption', 'label', 'table_text'
            ])
            df.to_csv(self.csv_path, index=False)
            print(f"Created empty CSV file at {self.csv_path}")

    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            return df
        return pd.DataFrame()
    
    def _save_data(self, df: pd.DataFrame):
        df.to_csv(self.csv_path, index=False)


    def insert_table(self, paper_arxiv_id, path=None, caption=None, label=None, table_text=None):
        df = self._load_data()
        
        if name in df['name'].values:
            return None
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'paper_arxiv_id': paper_arxiv_id,
            'path': path,
            'caption': caption,
            'label': label,
            'table_text': table_text
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def delete_table_by_id(self, id):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True
    
    
    def update_table(self, id, paper_arxiv_id, path=None, caption=None, label=None, table_text=None):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        mask = df['id'] == id
        
        if paper_arxiv_id is not None:
            df.loc[mask, 'paper_arxiv_id'] = paper_arxiv_id
        if path is not None:
            df.loc[mask, 'path'] = path
        if caption is not None:
            df.loc[mask, 'caption'] = caption
        if label is not None:
            df.loc[mask, 'label'] = label
        if table_text is not None:
            df.loc[mask, 'table_text'] = table_text
        
        self._save_data(df)
        return True
    
    def get_table_by_id(self, id: int) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        table = df[df['id'] == id]
        return table
    
    def check_table_exists(self, id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        return id in df['id'].values

    def construct_tables_table_from_csv(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()

        required_cols = ['paper_arxiv_id', 'path','caption','label', 'table_text']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} tables from {csv_file}")
        return True