import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..csv_database.csv_arxiv_categories import CSVArxivCategory

class CSVArxivPaperCategory:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paper_category.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paper_category_table()
        self.csvac = CSVArxivCategory(csv_dir=csv_dir)

    def create_paper_category_table(self):
        df = pd.DataFrame(columns=['paper_arxiv_id', 'category_id'])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paper_category CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_paper_category(self, paper_arxiv_id, category_id):
        df = self._load_data()
        conflict = df[
            (df['paper_arxiv_id'] == paper_arxiv_id) &
            (df['category_id'] == category_id)
        ]
        if not conflict.empty:
            return False
        new_row = pd.DataFrame([{
            'paper_arxiv_id': paper_arxiv_id,
            'category_id': category_id
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True
    
    def get_all_paper_categories(self):

        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()
    

    def get_paper_neighboring_categories(self, paper_arxiv_id: str) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paper_arxiv_id'] == paper_arxiv_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def get_category_neighboring_papers(self, category_id: str) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['category_id'] == category_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)

    # def ?(self):
    def construct_paper_categories_table_from_api(self, arxiv_ids, dest_dir):
        # Check if papers already exists in the directory
        md = MultiDownload()
        downloaded_paper_ids = []
        for arxiv_id in arxiv_ids:
            paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            if not os.path.exists(paper_dir):
                downloaded_paper_ids.append(arxiv_id)
        
        for arxiv_id in downloaded_paper_ids:
            try:
                md.download_arxiv(input=arxiv_id, input_type = "id", output_type="latex", dest_dir=self.dest_dir)
                print(f"paper with id {arxiv_id} downloaded")
                downloaded_paper_ids.append(arxiv_id)
            except RuntimeError as e:
                print(f"[ERROR] Failed to download {arxiv_id}: {e}")
                continue
        
        for arxiv_id in arxiv_ids:
            # Search if the corresponding paper graph exists
            json_path = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"
            
            try:
                with open(json_path, 'r') as file:
                    file_json = json.load(file)
                    categories = file_json['categories']
                    # Search categories in dataset?
                    for category in categories:
                        category_id = self.csvac.get_id_by_category(category)
                        self.insert_paper_category(paper_arxiv_id=arxiv_id, category_id=category_id)
            
            except FileNotFoundError:
                print(f"Error: The file at path '{json_path}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON at path '{json_path}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue


    def delete_paper_category_by_id(self, paper_arxiv_id: str, category_id: str) -> bool:

        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = (df['paper_arxiv_id'] == paper_arxiv_id) & (df['category_id'] == category_id)
        
        if not mask.any():
            return False
        
        df = df[~mask]
        self._save_data(df)
        
        return True


    def delete_paper_category_by_paper_id(self, paper_arxiv_id: str) -> int:

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


    def delete_paper_category_by_category_id(self, category_id: str) -> int:

        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['category_id'] == category_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count



    def construct_table_from_csv(self, csv_file):
        """
        Construct the paper-category relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, category_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['paper_arxiv_id', 'category_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Filter out relationships that already exist
            if not current_df.empty:
                existing_pairs = set(zip(current_df['paper_arxiv_id'], current_df['category_id']))
                external_df['_pair'] = list(zip(external_df['paper_arxiv_id'], external_df['category_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new paper-category relationships to import")
                return True

            # Ensure correct column order
            external_df = external_df[['paper_arxiv_id', 'category_id']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paper-category relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paper-category relationships from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the paper-category relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {"paper_arxiv_id": "1706.03762v7", "category_id": "cs.AI"},
                {"paper_arxiv_id": "1706.03762v7", "category_id": "cs.LG"},
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
                if 'paper_categories' in json_data:
                    relations_list = json_data['paper_categories']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paper-category data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(relations_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['paper_arxiv_id', 'category_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Filter out relationships that already exist
            if not current_df.empty:
                existing_pairs = set(zip(current_df['paper_arxiv_id'], current_df['category_id']))
                external_df['_pair'] = list(zip(external_df['paper_arxiv_id'], external_df['category_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new paper-category relationships to import")
                return True

            # Ensure correct column order
            external_df = external_df[['paper_arxiv_id', 'category_id']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paper-category relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paper-category relationships from JSON: {e}")
            return False


