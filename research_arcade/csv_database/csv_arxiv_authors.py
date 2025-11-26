import pandas as pd
import os
from typing import Optional
from pathlib import Path
import sys
from semanticscholar import SemanticScholar

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor

class CSVArxivAuthors:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_authors.csv"
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

    def update_author(self, semantic_scholar_id=None, name=None, homepage=None):
        """Update an author by id. Returns True if updated, False if not found."""
        df = self._load_data()
        
        if semantic_scholar_id not in df['semantic_scholar_id'].values:
            return False
        
        mask = df['semantic_scholar_id'] == semantic_scholar_id
        
        if semantic_scholar_id is not None:
            df.loc[mask, 'semantic_scholar_id'] = semantic_scholar_id
        if name is not None:
            df.loc[mask, 'name'] = name
        if homepage is not None:
            df.loc[mask, 'homepage'] = homepage
        
        self._save_data(df)
        return True
        
    def get_author_by_id(self, semantic_scholar_id: int) -> Optional[pd.DataFrame]:
        """Get an author by its id. Returns a DataFrame with the author or None if not found."""
        df = self._load_data()
        
        if df.empty or semantic_scholar_id not in df['id'].values:
            return None
        
        author = df[df['semantic_scholar_id'] == semantic_scholar_id]
        return author

    def check_author_exists(self, semantic_scholar_id: int) -> bool:
        """Check if an author exists by its id."""
        df = self._load_data()

        if df.empty:
            return False

        return semantic_scholar_id in df['semantic_scholar_id'].values

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
    
    def get_all_authors(self, is_all_features=True):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()

    def construct_authors_table_from_api(self, arxiv_ids, dest_dir):
        """
        Given arxiv ids, find the semantic scholar ids and pages of the authors
        """
        # Search for authors online
        sch = SemanticScholar()
        for arxiv_id in arxiv_ids:
            base_arxiv_id, version = arxiv_id_processor(arxiv_id=arxiv_id)
            print(f"base_arxiv_id: {base_arxiv_id}")
            try:
                paper_sch = sch.get_paper(f"ARXIV:{base_arxiv_id}")
                authors = paper_sch.authors
                for author in authors:
                    semantic_scholar_id = author.authorId
                    author_r = sch.get_author(semantic_scholar_id)
                    name = author_r.name
                    url = author_r.url
                
                    self.insert_author(semantic_scholar_id=semantic_scholar_id, name=name, homepage=url)
            except Exception as e:
                print(f"Paper with arxiv id {base_arxiv_id} not found on semantic scholar: {e}")
                # return False
                continue

    def construct_table_from_csv(self, csv_file):
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
    
        try:
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

            # Generate IDs for new authors
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Filter out authors that already exist (based on semantic_scholar_id)
            if not current_df.empty:
                existing_ids = set(current_df['semantic_scholar_id'].values)
                external_df = external_df[~external_df['semantic_scholar_id'].isin(existing_ids)]

            if external_df.empty:
                print("No new authors to import (all authors already exist)")
                return True

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} authors from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing authors from CSV: {e}")
            return False

    def construct_table_from_json(self, json_file):
        if not os.path.exists(json_file):
            print(f"Error: JSON file {json_file} does not exist.")
            return False

        try:
            import json
            
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(json_data, dict):
                # If it's a dict, look for an 'authors' key
                if 'authors' in json_data:
                    authors_list = json_data['authors']
                else:
                    # Treat the dict as a single author record
                    authors_list = [json_data]
            elif isinstance(json_data, list):
                authors_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not authors_list:
                print("Error: No author data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(authors_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['semantic_scholar_id', 'name']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            if 'homepage' not in external_df.columns:
                external_df['homepage'] = None

            # Generate IDs for new authors
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Filter out authors that already exist (based on semantic_scholar_id)
            if not current_df.empty:
                existing_ids = set(current_df['semantic_scholar_id'].values)
                external_df = external_df[~external_df['semantic_scholar_id'].isin(existing_ids)]

            if external_df.empty:
                print("No new authors to import (all authors already exist)")
                return True

            # Ensure correct column order and data types
            external_df = external_df[['id', 'semantic_scholar_id', 'name', 'homepage']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} authors from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing authors from JSON: {e}")
            return False