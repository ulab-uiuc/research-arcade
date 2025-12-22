import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json

class CSVArxivPaperTable:
    def __init__(self, csv_dir: str):
        self.csv_dir = csv_dir
        csv_path = f"{csv_dir}/arxiv_paper_tables.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paper_tables_table()

    def create_paper_tables_table(self):
        df = pd.DataFrame(columns=['paper_arxiv_id', 'table_id'])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paper_tables CSV at {self.csv_path}")

    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            dtype_map = {
                "paper_arxiv_id": str
            }
            # df = pd.read_csv(self.csv_path)
            df = pd.read_csv(self.csv_path, dtype=dtype_map)
            return df
        return pd.DataFrame()


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


    def match_table_id(self, paper_arxiv_id, label):
        csv_path2 = f"{self.csv_dir}/arxiv_tables.csv"
        
        if not os.path.exists(csv_path2):
            return None
        
        if label is None:
            return None
        
        df2 = pd.read_csv(csv_path2, dtype={'paper_arxiv_id': str, 'label': str})
        
        mask = (
            (df2['paper_arxiv_id'] == str(paper_arxiv_id)) & 
            (df2['label'] == str(label))
        )

        matched_rows = df2[mask]
        
        return matched_rows.iloc[0]['id'] if len(matched_rows) > 0 else None


    def construct_table_from_csv(self, csv_file):
        """
        Construct the paper-table relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, table_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['paper_arxiv_id', 'table_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Filter out relationships that already exist
            if not current_df.empty:
                existing_pairs = set(zip(current_df['paper_arxiv_id'], current_df['table_id']))
                external_df['_pair'] = list(zip(external_df['paper_arxiv_id'], external_df['table_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new paper-table relationships to import")
                return True

            # Ensure correct column order
            external_df = external_df[['paper_arxiv_id', 'table_id']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paper-table relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paper-table relationships from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the paper-table relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {"paper_arxiv_id": "1706.03762v7", "table_id": 1},
                {"paper_arxiv_id": "1706.03762v7", "table_id": 2},
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
                if 'paper_tables' in json_data:
                    relations_list = json_data['paper_tables']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paper-table data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(relations_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['paper_arxiv_id', 'table_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Filter out relationships that already exist
            if not current_df.empty:
                existing_pairs = set(zip(current_df['paper_arxiv_id'], current_df['table_id']))
                external_df['_pair'] = list(zip(external_df['paper_arxiv_id'], external_df['table_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new paper-table relationships to import")
                return True

            # Ensure correct column order
            external_df = external_df[['paper_arxiv_id', 'table_id']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paper-table relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paper-table relationships from JSON: {e}")
            return False


    def construct_paper_tables_table_from_api(self, arxiv_ids, dest_dir):

        for arxiv_id in arxiv_ids:
            json_path = f"{dest_dir}/output/{arxiv_id}.json"

            try:
                with open(json_path, 'r') as file:
                    file_json = json.load(file)
            except FileNotFoundError:
                print(f"Error: The file '{file_json}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{file_json}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue

            table_jsons = file_json['table']
            for table_json in table_jsons:
                
                caption = table_json['caption']
                label = table_json['label']
                table = table_json['tabular']
                # We don't currently store the table anywhere as a file so the table path is empty
                path = None

                table_id = self.match_table_id(paper_arxiv_id=arxiv_id, label=label)
                
                
                self.insert_paper_table(paper_arxiv_id = arxiv_id, table_id=table_id)

