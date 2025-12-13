import pandas as pd
import os
from typing import Optional
from pathlib import Path
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor, figure_iteration_recursive
class CSVArxivFigure:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_figures.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_figures_table()
        # self.arxiv_crawler = ArxivCrawler()

    def create_figures_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'paper_arxiv_id', 'path', 'caption', 'label', 'name'
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


    def insert_figure(self, paper_arxiv_id, path, caption=None, label=None, name=None):
        df = self._load_data()
        
        if name in df['name'].values:
            return None
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'paper_arxiv_id': paper_arxiv_id,
            'path': path,
            'caption': caption,
            'label': label,
            'name': name
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id


    def delete_figure_by_id(self, id):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True


    def update_figure(self, id, paper_arxiv_id, path, caption=None, label=None, name=None):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        mask = df['id'] == id
        
        if paper_arxiv_id is not None:
            df.loc[mask, 'paper_arxiv_id'] = paper_arxiv_id
        if path is not None:
            df.loc[mask, 'path'] = path
        if caption is not None:
            df.loc[mask, 'caption'] = caption
        if label is not None:
            df.loc[mask, 'label'] = label
        if name is not None:
            df.loc[mask, 'name'] = name
        
        self._save_data(df)
        return True


    def get_figure_by_id(self, id: int) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        figure = df[df['id'] == id]
        return figure
    
    def check_figure_exists(self, id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        return id in df['id'].values

    def construct_figure_table_from_csv(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()

        required_cols = ['paper_arxiv_id', 'path','caption','label', 'name']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} figures from {csv_file}")
        return True
    

    def get_all_figures(self, is_all_features=True):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()


    def construct_figures_table_from_api(self, arxiv_ids, dest_dir):
        md = MultiDownload()
        
        # Check if papers already exists in the directory
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

            json_path = f"{dest_dir}/output/{arxiv_id}.json"
            if not os.path.exists(json_path):
                # arxiv_id_graph.append(arxiv_id)
                try:
                    # Build corresponding graph
                    md.build_paper_graph(
                        input=arxiv_id,
                        input_type="id",
                        dest_dir=dest_dir
                    )
                except Exception as e:
                    print(f"[Warning] Failed to process papers: {e}")
                    continue

            try:
                with open(json_path, 'r') as file:
                    file_json = json.load(file)
                figure_jsons = file_json['figure']
                for figure_json in figure_jsons:

                    figures = figure_iteration_recursive(figure_json=figure_json)
                    for figure in figures:
                        path, caption, label = figure
                        self.insert_figure(paper_arxiv_id=arxiv_id, path=path, caption=caption,label=label)

            except FileNotFoundError:
                print(f"Error: The file with path '{json_path}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from path '{json_path}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue




                
    def construct_table_from_csv(self, csv_file):
        """
        Construct the figures table from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file containing figure data
            
        Expected CSV format:
            - Required columns: paper_arxiv_id, path
            - Optional columns: caption, label, name
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['paper_arxiv_id', 'path']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            for col in ['caption', 'label', 'name']:
                if col not in external_df.columns:
                    external_df[col] = None

            # Generate IDs for new figures
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Filter out figures that already exist (based on name if provided)
            if not current_df.empty and 'name' in external_df.columns:
                existing_names = set(current_df['name'].dropna().values)
                # Only filter if name is not null
                mask = external_df['name'].notna() & external_df['name'].isin(existing_names)
                external_df = external_df[~mask]

            if external_df.empty:
                print("No new figures to import")
                return True

            # Ensure correct column order
            external_df = external_df[['id', 'paper_arxiv_id', 'path', 'caption', 'label', 'name']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} figures from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing figures from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the figures table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file containing figure data
            
        Expected JSON format:
            [
                {
                    "paper_arxiv_id": "1706.03762v7",
                    "path": "/path/to/figure1.png",
                    "caption": "Architecture diagram",
                    "label": "fig:architecture",
                    "name": "figure1"
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
                if 'figures' in json_data:
                    figures_list = json_data['figures']
                else:
                    figures_list = [json_data]
            elif isinstance(json_data, list):
                figures_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not figures_list:
                print("Error: No figure data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(figures_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['paper_arxiv_id', 'path']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            for col in ['caption', 'label', 'name']:
                if col not in external_df.columns:
                    external_df[col] = None

            # Generate IDs for new figures
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Filter out figures that already exist (based on name if provided)
            if not current_df.empty and 'name' in external_df.columns:
                existing_names = set(current_df['name'].dropna().values)
                mask = external_df['name'].notna() & external_df['name'].isin(existing_names)
                external_df = external_df[~mask]

            if external_df.empty:
                print("No new figures to import")
                return True

            # Ensure correct column order
            external_df = external_df[['id', 'paper_arxiv_id', 'path', 'caption', 'label', 'name']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} figures from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing figures from JSON: {e}")
            return False

    def construct_paper_figures_table_from_api(self, arxiv_ids, )