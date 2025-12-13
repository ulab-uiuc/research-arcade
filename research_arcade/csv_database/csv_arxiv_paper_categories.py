import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json


class CSVArxivPaperCategory:
    def __init__(self, csv_dir: str):
        self.csv_dir = csv_dir
        csv_path = f"{csv_dir}/arxiv_paper_category.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paper_category_table()

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

    def insert_category_id_by_name(self, category_name):
        # load another dataframe?

        csv_path2 = f"{self.csv_dir}/arxiv_category.csv"
        df2 = pd.read_csv(csv_path2)

        # Unique return
        return [df2['name'] == category_name]['id'][0]



    def insert_paper_category_by_name(self, paper_arxiv_id, category_name):
        # search id by name
        category_id = self.insert_category_id_by_name(category_name)
        # insert by id
        self.insert_category_id_by_name(paper_arxiv_id, category_id)



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


    def construct_paper_category_table_from_api(self, arxiv_ids, dest):
        # The same logic, that we first open the file, then create the corresponding stuff.

        # In fact, we only need to add paper category
        # Open the metadata

        for arxiv_id in arxiv_ids:
            metadata_path = f"{dest}/{arxiv_id}/{arxiv_id}_metadata.json"

            with open(metadata_path, 'r') as file:
                metadata_json = json.load(file)
                categories = metadata_json['categories']
                for category in categories:
                    self.insert_paper_category_by_name(paper_arxiv_id=arxiv_id, category=category)
