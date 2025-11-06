import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json


class CSVArxivPaperAuthor:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paper_authors.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paper_authors_table()

    def create_paper_authors_table(self):
        df = pd.DataFrame(columns=['paper_arxiv_id', 'author_id', 'author_sequence'])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paper_authors CSV at {self.csv_path}")

    def _load_data(self):
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()

    def _save_data(self, df):
        df.to_csv(self.csv_path, index=False)

    def insert_paper_author(self, paper_arxiv_id, author_id, author_sequence):
        df = self._load_data()
        conflict = df[
            (df['paper_arxiv_id'] == paper_arxiv_id) &
            (df['author_id'] == author_id)
        ]
        if not conflict.empty:
            return False
        new_row = pd.DataFrame([{
            'paper_arxiv_id': paper_arxiv_id,
            'author_id': author_id,
            'author_sequence': author_sequence
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True

    
    def get_all_paper_authors(self, is_all_features=True):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()
        
    def get_paper_neighboring_authors(self, paper_arxiv_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paper_arxiv_id'] == paper_arxiv_id].copy()
        
        if result.empty:
            return None
        
        result = result.sort_values('author_sequence', ascending=True)
        
        return result.reset_index(drop=True)

    def get_author_neighboring_papers(self, author_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['author_id'] == author_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)

    def delete_paper_author_by_id(self, paper_arxiv_id: str, author_id: str) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = (df['paper_arxiv_id'] == paper_arxiv_id) & (df['author_id'] == author_id)
        
        if not mask.any():
            return False
        
        df = df[~mask]
        self._save_data(df)
        
        return True


    def delete_paper_author_by_paper_id(self, paper_arxiv_id: str) -> int:
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


    def delete_paper_author_by_author_id(self, author_id: str) -> int:
        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['author_id'] == author_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count


    def construct_table_from_csv(self, csv_file):
        """
        Construct the paper-author relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, author_id, author_sequence
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['paper_arxiv_id', 'author_id', 'author_sequence']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Filter out relationships that already exist
            if not current_df.empty:
                existing_pairs = set(zip(current_df['paper_arxiv_id'], current_df['author_id']))
                external_df['_pair'] = list(zip(external_df['paper_arxiv_id'], external_df['author_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new paper-author relationships to import")
                return True

            # Ensure correct column order
            external_df = external_df[['paper_arxiv_id', 'author_id', 'author_sequence']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paper-author relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paper-author relationships from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):

        if not os.path.exists(json_file):
            print(f"Error: JSON file {json_file} does not exist.")
            return False

        try:
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(json_data, dict):
                if 'paper_authors' in json_data:
                    relations_list = json_data['paper_authors']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paper-author data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(relations_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['paper_arxiv_id', 'author_id', 'author_sequence']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Filter out relationships that already exist
            if not current_df.empty:
                existing_pairs = set(zip(current_df['paper_arxiv_id'], current_df['author_id']))
                external_df['_pair'] = list(zip(external_df['paper_arxiv_id'], external_df['author_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new paper-author relationships to import")
                return True

            # Ensure correct column order
            external_df = external_df[['paper_arxiv_id', 'author_id', 'author_sequence']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paper-author relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paper-author relationships from JSON: {e}")
            return False


