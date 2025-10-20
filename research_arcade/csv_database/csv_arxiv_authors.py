import pandas as pd
import os
from typing import Optional
from pathlib import Path

class CSVArxivAuthors:
    def __init__(self, csv_dir):
        csv_path = f"{csv_dir}/authors.csv"
        self.csv_path = csv_path
        # Set up the target directory
        # Automatically create the csv path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_authors_table()

    def create_authors_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'semantic_scholar_id', 'name', 'homepage'
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

    def insert_author(self, semantic_scholar_id, name, homepage=None):
        """Insert an author. Returns the generated author id."""
        df = self._load_data()
        
        if semantic_scholar_id in df['semantic_scholar_id'].values:
            return None
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'semantic_scholar_id': semantic_scholar_id,
            'name': name,
            'homepage': homepage
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def delete_author_by_id(self, id):
        """Delete an author by its id. Returns True if deleted, False if not found."""
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True

    def update_author(self, id, semantic_scholar_id=None, name=None, homepage=None):
        """Update an author by id. Returns True if updated, False if not found."""
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        mask = df['id'] == id
        
        if semantic_scholar_id is not None:
            df.loc[mask, 'semantic_scholar_id'] = semantic_scholar_id
        if name is not None:
            df.loc[mask, 'name'] = name
        if homepage is not None:
            df.loc[mask, 'homepage'] = homepage
        
        self._save_data(df)
        return True

    def get_author_by_id(self, id: int) -> Optional[pd.DataFrame]:
        """Get an author by its id. Returns a DataFrame with the author or None if not found."""
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        author = df[df['id'] == id]
        return author

    def check_author_exists(self, id: int) -> bool:
        """Check if an author exists by its id."""
        df = self._load_data()

        if df.empty:
            return False

        return id in df['id'].values

    def construct_author_table_from_csv(self, csv_file: str):
        """
        Construct the authors table from an external CSV file.
        Assumes the CSV has compatible columns or can be mapped to the authors schema.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()

        required_cols = ['semantic_scholar_id', 'name']
        missing_cols = [col for col in required_cols if col not in external_df.columns]

        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False

        # Add optional columns if they don't exist
        if 'homepage' not in external_df.columns:
            external_df['homepage'] = None

        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))

        # Filter out authors that already exist (based on semantic_scholar_id)
        if not current_df.empty:
            existing_ids = set(current_df['semantic_scholar_id'].values)
            external_df = external_df[~external_df['semantic_scholar_id'].isin(existing_ids)]

        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)

        print(f"Successfully imported {len(external_df)} authors from {csv_file}")
        return True
