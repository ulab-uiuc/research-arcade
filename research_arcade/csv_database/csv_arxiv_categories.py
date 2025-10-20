import pandas as pd
import os
from typing import Optional
from pathlib import Path


class CSVArxivCategory:
    def __init__(self, csv_dir):
        csv_path = f"{csv_dir}/categories.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_categories_table()
        # self.arxiv_crawler = ArxivCrawler()
    
    
    def create_categories_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'name', 'description'
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


    def insert_category(self, name, description=None):
        df = self._load_data()
        
        if name in df['name'].values:
            return None
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'name': name,
            'description': description
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def delete_category_by_id(self, id):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True
    
    
    def update_category(self, id, name, description=None):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        mask = df['id'] == id
        
        if name is not None:
            df.loc[mask, 'name'] = name
        if description is not None:
            df.loc[mask, 'description'] = description
        
        self._save_data(df)
        return True
    
    def get_category_by_id(self, id: int) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        category = df[df['id'] == id]
        return category
    
    def check_category_exists(self, id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        return id in df['id'].values



    def construct_category_table_from_csv(self, csv_file: str):
        """
        Construct the categories table from an external CSV file.
        Assumes the CSV has compatible columns or can be mapped to the categories schema.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()
        
        required_cols = ['name']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        # Add optional columns if they don't exist
        if 'description' not in external_df.columns:
            external_df['description'] = None
        
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        if not current_df.empty:
            existing_names = set(current_df['name'].values)
            external_df = external_df[~external_df['name'].isin(existing_names)]
        
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} categories from {csv_file}")
        return True