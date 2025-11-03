import pandas as pd
import os
from typing import Optional
from pathlib import Path
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
import json

class CSVArxivTable:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_tables.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_tables_table()
        # self.arxiv_crawler = ArxivCrawler()
    
    
    def create_tables_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'paper_arxiv_id', 'path', 'caption', 'label', 'table_text'
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


    def insert_table(self, paper_arxiv_id, path=None, caption=None, label=None, table_text=None):

        df = self._load_data()

        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'paper_arxiv_id': paper_arxiv_id,
            'path': path,
            'caption': caption,
            'label': label,
            'table_text': table_text
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def delete_table_by_id(self, id):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True
    
    
    def update_table(self, id, paper_arxiv_id, path=None, caption=None, label=None, table_text=None):
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
        if table_text is not None:
            df.loc[mask, 'table_text'] = table_text
        
        self._save_data(df)
        return True
    
    def get_table_by_id(self, id: int) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        table = df[df['id'] == id]
        return table
    
    def check_table_exists(self, id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        return id in df['id'].values

    def construct_tables_table_from_csv(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()

        required_cols = ['paper_arxiv_id', 'path','caption','label', 'table_text']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} tables from {csv_file}")
        return True
    
    def get_all_tables(self, is_all_features=True):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()

    def construct_tables_table_from_api(self, arxiv_ids, dest_dir):
        # Check if papers already exists in the directory
        downloaded_paper_ids = []
        for arxiv_id in arxiv_ids:
            paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

            if not os.path.exists(paper_dir):
                downloaded_paper_ids.append(arxiv_id)

        for arxiv_id in downloaded_paper_ids:
            md = MultiDownload()
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
                    table_jsons = file_json['table']
                    for table_json in table_jsons:

                        caption = table_json['caption']
                        label = table_json['label']
                        table = table_json['tabular']
                        self.insert_table(paper_arxiv_id=arxiv_id, path=None, caption=caption, label=label, table_text=table)

            except FileNotFoundError:
                print(f"Error: The file '{file_json}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{file_json}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue
