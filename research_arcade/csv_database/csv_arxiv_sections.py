import pandas as pd
import os
from typing import Optional
from pathlib import Path


class CSVArxivSections:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_sections.csv"
        self.csv_path = csv_path
        # Set up the target directory
        # Automatically create the csv path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_sections_table()
        # self.arxiv_crawler = ArxivCrawler()


    def create_sections_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'content', 'title', 'appendix', 'paper_arxiv_id'
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


    def insert_section(self, content, title, is_appendix, paper_arxiv_id):
        """Insert a section. Returns the generated section id."""
        df = self._load_data()
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'content': content,
            'title': title,
            'appendix': is_appendix,
            'paper_arxiv_id': paper_arxiv_id
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def delete_section_by_id(self, id):
        """Delete a section by its id. Returns True if deleted, False if not found."""
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True
    
    def delete_section_by_paper_arxiv_id(self, paper_arxiv_id):
        """Delete all sections for a specific paper arxiv_id. Returns the number of sections deleted."""
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns:
            return 0
        
        initial_count = len(df)
        
        df = df[df['paper_arxiv_id'] != paper_arxiv_id]
        deleted_count = initial_count - len(df)
        
        self._save_data(df)
        return deleted_count
    
    def update_section(self, id, content=None, title=None, is_appendix=None, paper_arxiv_id=None):
        """Update a section by id. Returns True if updated, False if not found."""
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        mask = df['id'] == id
        
        if content is not None:
            df.loc[mask, 'content'] = content
        if title is not None:
            df.loc[mask, 'title'] = title
        if is_appendix is not None:
            df.loc[mask, 'appendix'] = is_appendix
        if paper_arxiv_id is not None:
            df.loc[mask, 'paper_arxiv_id'] = paper_arxiv_id
        
        self._save_data(df)
        return True
    
    def get_section_by_id(self, id: int) -> Optional[pd.DataFrame]:
        """Get a section by its id. Returns a DataFrame with the section or None if not found."""
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        section = df[df['id'] == id]
        return section

    def get_sections_by_arxiv_id(self, arxiv_id: str) -> Optional[pd.DataFrame]:
        """Get all sections for a paper by its arxiv_id. Returns a DataFrame with sections or None if not found."""
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns:
            return None
        
        sections = df[df['paper_arxiv_id'] == arxiv_id]
        
        if sections.empty:
            return None
        
        return sections
    
    def check_section_exists(self, id: int) -> bool:
        """Check if a section exists by its id."""
        df = self._load_data()
        
        if df.empty:
            return False
        
        return id in df['id'].values
    
    def construct_sections_table_from_csv(self, csv_file: str):
        """
        Construct the sections table from an external CSV file.
        Assumes the CSV has compatible columns or can be mapped to the sections schema.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()
        
        required_cols = ['content', 'title', 'appendix', 'paper_arxiv_id']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} sections from {csv_file}")
        return True