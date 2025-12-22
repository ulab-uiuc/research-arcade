import pandas as pd
import os
from pathlib import Path
from typing import Optional, List, Tuple
import json

def figure_iteration_recursive(figure_json):

        # Create a set of figures along with the
        # list represents (path, caption, label)
        path_to_info: List[Tuple[str, str, str]] = []

        # First iterate through parent, then go into the children

        def figure_iteration(figure_json):
            nonlocal path_to_info

            if not figure_json:
                return
            if figure_json['figure_paths']:
                path = figure_json['figure_paths'][0]
                caption = figure_json['caption']
                label = figure_json['label']
                path_to_info.append((path, caption, label))
            subfigures = figure_json['subfigures']
            
            for subfigure in subfigures:
                figure_iteration(subfigure)
        
        figure_iteration(figure_json=figure_json)
        return path_to_info



class CSVArxivPaperFigure:
    def __init__(self, csv_dir: str):
        self.csv_dir = csv_dir
        csv_path = f"{csv_dir}/arxiv_paper_figures.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paper_figures_table()
    
    def create_paper_figures_table(self):
        df = pd.DataFrame(columns=['paper_arxiv_id', 'figure_id'])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paper_figures CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_paper_figure(self, paper_arxiv_id, figure_id):
        df = self._load_data()
        conflict = df[
            (df['paper_arxiv_id'] == paper_arxiv_id) &
            (df['figure_id'] == figure_id)
        ]
        if not conflict.empty:
            return False
        new_row = pd.DataFrame([{
            'paper_arxiv_id': paper_arxiv_id,
            'figure_id': figure_id
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True

    def get_all_paper_figures(self):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()
    
    def get_paper_neighboring_figures(self, paper_arxiv_id: str) -> Optional[pd.DataFrame]:
        
        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paper_arxiv_id'] == paper_arxiv_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def get_figure_neighboring_papers(self, figure_id: int) -> Optional[pd.DataFrame]:
        
        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['figure_id'] == figure_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def delete_paper_figure_by_id(self, paper_arxiv_id: str, figure_id: int) -> bool:

        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = (df['paper_arxiv_id'] == paper_arxiv_id) & (df['figure_id'] == figure_id)
        
        if not mask.any():
            return False
        
        df = df[~mask]
        self._save_data(df)
        
        return True


    def delete_paper_figure_by_paper_id(self, paper_arxiv_id: str) -> int:

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


    def delete_paper_figure_by_figure_id(self, figure_id: int) -> int:

        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['figure_id'] == figure_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count
    



    def construct_table_from_csv(self, csv_file):
        """
        Construct the paper-figure relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, figure_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['paper_arxiv_id', 'figure_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Filter out relationships that already exist
            if not current_df.empty:
                existing_pairs = set(zip(current_df['paper_arxiv_id'], current_df['figure_id']))
                external_df['_pair'] = list(zip(external_df['paper_arxiv_id'], external_df['figure_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new paper-figure relationships to import")
                return True

            # Ensure correct column order
            external_df = external_df[['paper_arxiv_id', 'figure_id']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paper-figure relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paper-figure relationships from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the paper-figure relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {"paper_arxiv_id": "1706.03762v7", "figure_id": 1},
                {"paper_arxiv_id": "1706.03762v7", "figure_id": 2},
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
                if 'paper_figures' in json_data:
                    relations_list = json_data['paper_figures']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paper-figure data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(relations_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['paper_arxiv_id', 'figure_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Filter out relationships that already exist
            if not current_df.empty:
                existing_pairs = set(zip(current_df['paper_arxiv_id'], current_df['figure_id']))
                external_df['_pair'] = list(zip(external_df['paper_arxiv_id'], external_df['figure_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new paper-figure relationships to import")
                return True

            # Ensure correct column order
            external_df = external_df[['paper_arxiv_id', 'figure_id']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paper-figure relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paper-figure relationships from JSON: {e}")
            return False

    def match_figure_id(self, paper_arxiv_id, label):
        csv_path2 = f"{self.csv_dir}/arxiv_figures.csv"
        
        if not os.path.exists(csv_path2):
            return None
        
        df2 = pd.read_csv(csv_path2, dtype={'paper_arxiv_id': str, 'label': str})
        
        # Convert inputs to strings for consistent comparison
        paper_arxiv_id_str = str(paper_arxiv_id)
        label_str = str(label) if label is not None else None
        
        if label_str is None:
            return None
        
        mask = (
            (df2['paper_arxiv_id'].astype(str) == paper_arxiv_id_str) & 
            (df2['label'].astype(str) == label_str)
        )
        
        matched_rows = df2[mask]

        
        if len(matched_rows) > 0:
            return matched_rows.iloc[0]['id']
        else:
            return None        

    def construct_paper_figures_table_from_api(self, arxiv_ids, dest_dir):

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

            figure_jsons = file_json['figure']

            for figure_json in figure_jsons:
                
                figures = figure_iteration_recursive(figure_json=figure_json)

                for figure in figures:
                    path, caption, label = figure
                    print("arxiv_id")
                    print(arxiv_id)
                    print("label")
                    print(label)
                    figure_id = self.match_figure_id(paper_arxiv_id=arxiv_id, label=label)

                    self.insert_paper_figure(paper_arxiv_id=arxiv_id, figure_id=figure_id)
