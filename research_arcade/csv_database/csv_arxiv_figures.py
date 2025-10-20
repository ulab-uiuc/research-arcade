import pandas as pd
import os
from typing import Optional
from pathlib import Path


class CSVArxivFigure:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_figures.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_figures_table()
        # self.arxiv_crawler = ArxivCrawler()
    
    
    def create_figures_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'paper_arxiv_id', 'path', 'caption', 'label', 'name'
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


    def insert_figure(self, paper_arxiv_id, path, caption=None, label=None, name=None):
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
            'name': name
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def delete_figure_by_id(self, id):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True
    
    
    def update_figure(self, id, paper_arxiv_id, path, caption=None, label=None, name=None):
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
        if name is not None:
            df.loc[mask, 'name'] = name
        
        self._save_data(df)
        return True
    
    def get_figure_by_id(self, id: int) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        figure = df[df['id'] == id]
        return figure
    
    def check_figure_exists(self, id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        return id in df['id'].values

    def construct_figure_table_from_csv(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()

        required_cols = ['paper_arxiv_id', 'path','caption','label', 'name']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} figures from {csv_file}")
        return True