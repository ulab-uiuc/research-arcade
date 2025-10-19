import pandas as pd
import os
from typing import List, Optional, Tuple
from pathlib import Path
import json


class CSVArxivParagraphs:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paragraphs_table()
        # self.arxiv_crawler = ArxivCrawler()
    

    def create_paragraphs_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'paragraph_id', 'content', 'paper_arxiv_id', 'paper_section'
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


    def insert_paragraph(self, paragraph_id, content, paper_arxiv_id, paper_section):
        df = self._load_data()
        
        conflict = df[
            (df['paragraph_id'] == paragraph_id) & 
            (df['paper_arxiv_id'] == paper_arxiv_id) & 
            (df['paper_section'] == paper_section)
        ]
        if not conflict.empty:
            return None
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'paragraph_id': paragraph_id,
            'content': content,
            'paper_arxiv_id': paper_arxiv_id,
            'paper_section': paper_section
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def delete_paragraph_by_id(self, id):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True
    
    def delete_paragraph_by_paper_arxiv_id(self, paper_arxiv_id):
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns:
            return 0
        
        initial_count = len(df)
        
        df = df[df['paper_arxiv_id'] != paper_arxiv_id]
        deleted_count = initial_count - len(df)
        
        self._save_data(df)
        return deleted_count
    
    def delete_paragraph_by_paper_section(self, paper_arxiv_id, paper_section):
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns or 'paper_section' not in df.columns:
            return 0
        
        initial_count = len(df)
        
        df = df[~((df['paper_arxiv_id'] == paper_arxiv_id) & (df['paper_section'] == paper_section))]
        deleted_count = initial_count - len(df)
        
        self._save_data(df)
        return deleted_count

    
    def update_paragraph(self, id, paragraph_id=None, content=None, paper_arxiv_id=None, paper_section=None):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        mask = df['id'] == id
        
        if paragraph_id is not None:
            df.loc[mask, 'paragraph_id'] = paragraph_id
        if content is not None:
            df.loc[mask, 'content'] = content
        if paper_arxiv_id is not None:
            df.loc[mask, 'paper_arxiv_id'] = paper_arxiv_id
        if paper_section is not None:
            df.loc[mask, 'paper_section'] = paper_section
        
        self._save_data(df)
        return True
    
    def get_paragraph_by_id(self, id: int) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        paragraph = df[df['id'] == id]
        return paragraph


    def get_paragraphs_by_arxiv_id(self, arxiv_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns:
            return None
        
        paragraphs = df[df['paper_arxiv_id'] == arxiv_id]
        
        if paragraphs.empty:
            return None
        
        return paragraphs
    
    def get_paragraphs_by_paper_section(self, paper_arxiv_id, paper_section):
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns or 'paper_section' not in df.columns:
            return None
        
        paragraphs = df[(df['paper_arxiv_id'] == paper_arxiv_id) & (df['paper_section'] == paper_section)]
        
        if paragraphs.empty:
            return None
        
        return paragraphs

    
    def check_paragraph_exists(self, id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        return id in df['id'].values
    
    def construct_paragraph_table_from_csv(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()
        
        required_cols = ['paragraph_id', 'content', 'paper_arxiv_id', 'paper_section']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} paragraphs from {csv_file}")
        return True