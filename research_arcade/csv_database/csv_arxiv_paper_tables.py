import pandas as pd
import os
from pathlib import Path
from typing import Optional

class CSVArxivPaperTable:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paper_tables.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paper_tables_table()

    def create_paper_tables_table(self):
        df = pd.DataFrame(columns=['paper_arxiv_id', 'table_id'])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paper_tables CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_paper_table(self, paper_arxiv_id, table_id):
        df = self._load_data()
        conflict = df[
            (df['paper_arxiv_id'] == paper_arxiv_id) &
            (df['table_id'] == table_id)
        ]
        if not conflict.empty:
            return False
        new_row = pd.DataFrame([{
            'paper_arxiv_id': paper_arxiv_id,
            'table_id': table_id
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True

import pandas as pd
import os
from pathlib import Path
from typing import Optional

class CSVArxivPaperFigure:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paper_figures.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paper_figures_table()
    
    def create_paper_figures_table(self):
        df = pd.DataFrame(columns=['paper_arxiv_id', 'figure_id'])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paper_figures CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_paper_figure(self, paper_arxiv_id, figure_id):
        df = self._load_data()
        conflict = df[
            (df['paper_arxiv_id'] == paper_arxiv_id) &
            (df['figure_id'] == figure_id)
        ]
        if not conflict.empty:
            return False
        new_row = pd.DataFrame([{
            'paper_arxiv_id': paper_arxiv_id,
            'figure_id': figure_id
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True

    def get_all_paper_figures(self):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()
    
    def get_all_paper_tables(self):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()
    

    def get_paper_neighboring_tables(self, paper_arxiv_id: str) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        # Filter for the specific paper
        result = df[df['paper_arxiv_id'] == paper_arxiv_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def get_table_neighboring_papers(self, table_id: int) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['table_id'] == table_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def delete_paper_table_by_id(self, paper_arxiv_id: str, table_id: int) -> bool:

        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = (df['paper_arxiv_id'] == paper_arxiv_id) & (df['table_id'] == table_id)
        
        if not mask.any():
            return False
        
        df = df[~mask]
        self._save_data(df)
        
        return True


    def delete_paper_table_by_paper_id(self, paper_arxiv_id: str) -> int:

        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['paper_arxiv_id'] == paper_arxiv_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count


    def delete_paper_table_by_table_id(self, table_id: int) -> int:

        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['table_id'] == table_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count
