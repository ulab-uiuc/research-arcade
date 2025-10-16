""" CSV version of the dataset """

import pandas as pd
import os
from typing import List, Optional, Tuple
from pathlib import Path
import json
# from ..data import *

# TODO: refactor the original ArxivCrawler into the ArxivCrawler file

class CSVArxivPapers:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        # Set up the target directory
        # Automatically create the csv path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_papers_table()
        self.arxiv_crawler = ArxivCrawler()
    
    def create_papers_table(self):
        """Create papers CSV with appropriate columns."""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'arxiv_id', 'base_arxiv_id', 'version', 
                'title', 'abstract', 'submit_date', 'metadata'
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
    
    def insert_paper(self, arxiv_id, base_arxiv_id, version, title, abstract=None, submit_date=None, metadata=None):
        """Insert a paper. Returns the generated paper id."""
        df = self._load_data()
        
        # Check for conflict
        if arxiv_id in df['arxiv_id'].values:
            return None
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        meta_str = json.dumps(metadata) if metadata is not None else None
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'arxiv_id': arxiv_id,
            'base_arxiv_id': base_arxiv_id,
            'version': version,
            'title': title,
            'abstract': abstract,
            'submit_date': submit_date,
            'metadata': meta_str
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id
    
    def delete_paper_by_arxiv_id(self, arxiv_id):
        """Delete a paper by its arxiv_id. Returns True if deleted, False if not found."""
        df = self._load_data()
        
        if arxiv_id not in df['arxiv_id'].values:
            return False
        
        df = df[df['arxiv_id'] != arxiv_id]
        self._save_data(df)
        return True
    
    def delete_paper_by_year(self, year):
        """Delete all papers from a specific year. Returns the number of papers deleted."""
        df = self._load_data()
        
        if df.empty or 'submit_date' not in df.columns:
            return 0
        
        # Convert submit_date to datetime and extract year
        df['submit_date'] = pd.to_datetime(df['submit_date'], errors='coerce')
        initial_count = len(df)
        
        # Filter out papers from the specified year
        df = df[df['submit_date'].dt.year != year]
        deleted_count = initial_count - len(df)
        
        self._save_data(df)
        return deleted_count
    
    def update_paper(self, arxiv_id, base_arxiv_id=None, version=None, title=None, abstract=None, submit_date=None, metadata=None):
        """Update a paper by arxiv_id. Returns True if updated, False if not found."""
        df = self._load_data()
        
        # Search for paper by arxiv id
        if arxiv_id not in df['arxiv_id'].values:
            return False
        
        # Update the corresponding paper
        mask = df['arxiv_id'] == arxiv_id
        
        if base_arxiv_id is not None:
            df.loc[mask, 'base_arxiv_id'] = base_arxiv_id
        if version is not None:
            df.loc[mask, 'version'] = version
        if title is not None:
            df.loc[mask, 'title'] = title
        if abstract is not None:
            df.loc[mask, 'abstract'] = abstract
        if submit_date is not None:
            df.loc[mask, 'submit_date'] = submit_date
        if metadata is not None:
            df.loc[mask, 'metadata'] = json.dumps(metadata)
        
        self._save_data(df)
        return True
    
    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[pd.DataFrame]:
        """Get a paper by its arxiv_id. Returns a DataFrame with the paper or None if not found."""
        df = self._load_data()
        
        if df.empty or arxiv_id not in df['arxiv_id'].values:
            return None
        
        paper = df[df['arxiv_id'] == arxiv_id]
        return paper
    
    def check_paper_exists(self, arxiv_id: str) -> bool:
        """Check if a paper exists by its arxiv_id."""
        df = self._load_data()
        
        if df.empty:
            return False
        
        return arxiv_id in df['arxiv_id'].values
    
    def construct_papers_table_from_csv(self, csv_file: str):
        """
        Construct the papers table from an external CSV file.
        Assumes the CSV has compatible columns or can be mapped to the papers schema.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        # Load the external CSV
        external_df = pd.read_csv(csv_file)
        
        # Load current data
        current_df = self._load_data()
        
        # Ensure the external CSV has required columns
        required_cols = ['arxiv_id', 'base_arxiv_id', 'version', 'title']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        # Add optional columns if they don't exist
        for col in ['abstract', 'submit_date', 'metadata']:
            if col not in external_df.columns:
                external_df[col] = None
        
        # Generate IDs for new papers
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        # Filter out papers that already exist (based on arxiv_id)
        if not current_df.empty:
            existing_ids = set(current_df['arxiv_id'].values)
            external_df = external_df[~external_df['arxiv_id'].isin(existing_ids)]
        
        # Concatenate and save
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} papers from {csv_file}")
        return True