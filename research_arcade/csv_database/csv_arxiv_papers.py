""" CSV version of the dataset """

import pandas as pd
import os
from typing import Optional
from pathlib import Path
import json
import sys
# from ..data import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor

# TODO: refactor the original ArxivCrawler into the ArxivCrawler file

class CSVArxivPapers:
    def __init__(self, csv_dir: str):
        csv_path = os.path.join(csv_dir, 'arxiv_papers.csv')
        self.csv_path = csv_path
        # Set up the target directory
        # Automatically create the csv path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_papers_table()
        # self.arxiv_crawler = ArxivCrawler()
    
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
            dtype_map = {
                "arxiv_id": str,
                "base_arxiv_id": str,
                "metadata": str
            }
            df = pd.read_csv(self.csv_path, dtype=dtype_map)
            return df

    def _save_data(self, df: pd.DataFrame):
        df.to_csv(self.csv_path, index=False)

    def sample_papers(self, sample_size: int) -> Optional[pd.DataFrame]:
        """Sample a number of papers randomly. Returns a DataFrame with sampled papers or None if empty."""
        df = self._load_data()
        
        if df.empty:
            return None
        
        sampled_df = df.sample(n=sample_size)
        return sampled_df.reset_index(drop=True)
    
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
    
    def delete_paper_by_id(self, arxiv_id):
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
    def get_all_papers(self, is_all_features=True) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()
    
    def get_paper_by_id(self, arxiv_id) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        mask = df['arxiv_id'] == arxiv_id
        result = df[mask].copy()
        
        if result.empty:
            print(f"No author found with author_openreview_id {arxiv_id}.")
            return None
        
        return result


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

    def construct_papers_table_from_api(self, arxiv_ids, dest_dir):

        # Check if papers already exists in the directory
        downloaded_paper_ids = []
        for arxiv_id in arxiv_ids:
            paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            if not os.path.exists(paper_dir):
                downloaded_paper_ids.append(arxiv_id)

        for arxiv_id in downloaded_paper_ids:
            md = MultiDownload()
            try:
                md.download_arxiv(input=arxiv_id, input_type = "id", output_type="latex", dest_dir=dest_dir)
                print(f"paper with id {arxiv_id} downloaded")
            except RuntimeError as e:
                print(f"[ERROR] Failed to download {arxiv_id}: {e}")
                continue
        
        for arxiv_id in arxiv_ids:
            # add metadata
            # read paper information

            base_arxiv_id, version = arxiv_id_processor(arxiv_id=arxiv_id)
            # Read metadata from path specified

            try:
                metadata_path = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)  # Use json.load(), not json.loads()
                
                # Validate required fields
                required_fields = ['title', 'abstract', 'published']
                if not all(field in metadata for field in required_fields):
                    raise ValueError(f"Missing required fields in metadata for {arxiv_id}")
                
                self.insert_paper(
                    arxiv_id=arxiv_id,
                    base_arxiv_id=base_arxiv_id,
                    version=version,
                    title=metadata['title'],
                    abstract=metadata['abstract'],
                    submit_date=metadata['published'],
                    metadata=metadata
                )
                print("Downloaded")
            except Exception:
                print(f"Paper {arxiv_id} does not have metadata downloaded")


    def construct_table_from_csv(self, csv_file):
        """
        Construct the papers table from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file containing paper data
            
        Expected CSV format:
            - Required columns: arxiv_id, base_arxiv_id, version, title
            - Optional columns: abstract, submit_date, metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

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

            if external_df.empty:
                print("No new papers to import (all papers already exist)")
                return True

            # Ensure correct column order
            external_df = external_df[['id', 'arxiv_id', 'base_arxiv_id', 'version', 'title', 'abstract', 'submit_date', 'metadata']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} papers from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing papers from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the papers table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file containing paper data
            
        Expected JSON format:
            [
                {
                    "arxiv_id": "1706.03762v7",
                    "base_arxiv_id": "1706.03762",
                    "version": 7,
                    "title": "Attention Is All You Need",
                    "abstract": "...",
                    "submit_date": "2017-06-12",
                    "metadata": {"venue": "NeurIPS 2017"}
                },
                ...
            ]
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(json_file):
            print(f"Error: JSON file {json_file} does not exist.")
            return False

        try:
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(json_data, dict):
                if 'papers' in json_data:
                    papers_list = json_data['papers']
                else:
                    papers_list = [json_data]
            elif isinstance(json_data, list):
                papers_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not papers_list:
                print("Error: No paper data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(papers_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['arxiv_id', 'base_arxiv_id', 'version', 'title']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            for col in ['abstract', 'submit_date']:
                if col not in external_df.columns:
                    external_df[col] = None
            
            # Handle metadata field - convert dict to JSON string if needed
            if 'metadata' not in external_df.columns:
                external_df['metadata'] = None
            else:
                external_df['metadata'] = external_df['metadata'].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )

            # Generate IDs for new papers
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Filter out papers that already exist
            if not current_df.empty:
                existing_ids = set(current_df['arxiv_id'].values)
                external_df = external_df[~external_df['arxiv_id'].isin(existing_ids)]

            if external_df.empty:
                print("No new papers to import (all papers already exist)")
                return True

            # Ensure correct column order
            external_df = external_df[['id', 'arxiv_id', 'base_arxiv_id', 'version', 'title', 'abstract', 'submit_date', 'metadata']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} papers from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing papers from JSON: {e}")
            return False
