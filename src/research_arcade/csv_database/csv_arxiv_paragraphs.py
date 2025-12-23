import pandas as pd
import os
from typing import Optional
from pathlib import Path
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.paper_collector.paper_graph_processor import PaperGraphProcessor
from ..arxiv_utils.utils import get_paragraph_num, arxiv_ids_hashing


class CSVArxivParagraphs:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paragraphs.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paragraphs_table()

    def create_paragraphs_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'paragraph_id', 'content', 'paper_arxiv_id', 'paper_section'
            ])
            df.to_csv(self.csv_path, index=False)
            print(f"Created empty CSV file at {self.csv_path}")

    def _load_data(self) -> pd.DataFrame:
        dtype_map = {
            "paper_arxiv_id": str,
            "paper_section": str
        }
        # df = pd.read_csv(self.csv_path)
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path, dtype=dtype_map)
            return df
        return pd.DataFrame()

    def _save_data(self, df: pd.DataFrame):
        df.to_csv(self.csv_path, index=False)

    def insert_paragraph(self, paragraph_id, content, paper_arxiv_id, paper_section, section_id=None, paragraph_in_paper_id=None):
        df = self._load_data()

        conflict = df[
            (df['paragraph_id'] == paragraph_id) & 
            (df['paper_arxiv_id'] == paper_arxiv_id) & 
            (df['paper_section'] == paper_section)
        ]
        if not conflict.empty:
            return None

        new_id = df['id'].max() + 1 if not df.empty else 1

        new_row = pd.DataFrame([{
            'id': new_id,
            'paragraph_id': paragraph_id,
            'content': content,
            'paper_arxiv_id': paper_arxiv_id,
            'paper_section': paper_section,
            'section_id': section_id,
            'paragraph_in_paper_id': paragraph_in_paper_id
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def delete_paragraph_by_id(self, id):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True
    
    def delete_paragraph_by_paper_arxiv_id(self, paper_arxiv_id):
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns:
            return 0
        
        initial_count = len(df)
        
        df = df[df['paper_arxiv_id'] != paper_arxiv_id]
        deleted_count = initial_count - len(df)
        
        self._save_data(df)
        return deleted_count
    
    def delete_paragraph_by_paper_section(self, paper_arxiv_id, paper_section):
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns or 'paper_section' not in df.columns:
            return 0
        
        initial_count = len(df)
        
        df = df[~((df['paper_arxiv_id'] == paper_arxiv_id) & (df['paper_section'] == paper_section))]
        deleted_count = initial_count - len(df)
        
        self._save_data(df)
        return deleted_count
    
    def update_paragraph(self, id, paragraph_id=None, content=None, paper_arxiv_id=None, paper_section=None):
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        mask = df['id'] == id
        
        if paragraph_id is not None:
            df.loc[mask, 'paragraph_id'] = paragraph_id
        if content is not None:
            df.loc[mask, 'content'] = content
        if paper_arxiv_id is not None:
            df.loc[mask, 'paper_arxiv_id'] = paper_arxiv_id
        if paper_section is not None:
            df.loc[mask, 'paper_section'] = paper_section
        
        self._save_data(df)
        return True
    
    def get_paragraph_by_id(self, id: int) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        paragraph = df[df['id'] == id]
        return paragraph

    def get_paragraphs_by_arxiv_id(self, arxiv_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns:
            return None
        
        paragraphs = df[df['paper_arxiv_id'] == arxiv_id]
        
        if paragraphs.empty:
            return None
        
        return paragraphs
    
    def get_paragraphs_by_paper_section(self, paper_arxiv_id, paper_section):
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns or 'paper_section' not in df.columns:
            return None
        
        paragraphs = df[(df['paper_arxiv_id'] == paper_arxiv_id) & (df['paper_section'] == paper_section)]
        
        if paragraphs.empty:
            return None
        
        return paragraphs

    def get_id_by_arxiv_id_section_paragraph_id(self, paper_arxiv_id, paper_section, paragraph_id):        
        df = self._load_data()

        if df.empty or 'paper_arxiv_id' not in df.columns or 'paper_section' not in df.columns:
            return None
        
        paragraphs = df[(df['paper_arxiv_id'] == paper_arxiv_id) & (df['paper_section'] == paper_section) & (df['paragraph_id'] == paragraph_id)]
        
        if paragraphs.empty:
            return None
        
        return paragraphs.iloc[0]['id']


    
    def check_paragraph_exists(self, id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        return id in df['id'].values
    
    def construct_paragraph_table_from_csv(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()
        
        required_cols = ['paragraph_id', 'content', 'paper_arxiv_id', 'paper_section']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} paragraphs from {csv_file}")
        return True

    def get_all_paragraphs(self, is_all_features=True):
        
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()

    def construct_paragraphs_table_from_api(self, arxiv_ids, dest_dir):
        # Check if papers already exists in the directory
        """
        section id and paragraph order required further
        Or maybe we can write a incremental method to process such information
        TODO
        """
        downloaded_paper_ids = []
        md = MultiDownload()
        
        data_dir_path = f"{dest_dir}/output"
        figures_dir_path = f"{dest_dir}/output/images"
        output_dir_path = f"{dest_dir}/output/paragraphs"
        pgp = PaperGraphProcessor(data_dir=data_dir_path, figures_dir=figures_dir_path, output_dir=output_dir_path, arxiv_ids=arxiv_ids)

        papers = []
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
            
            papers.append(arxiv_id)

        paper_paths = []
        # We first build paper node
        # We loop through the provided arxiv ids of paper.
        for arxiv_id in papers:
            paper_paths.append(f"{dest_dir}/output/{arxiv_id}.json")
        pgp.process_papers(paper_paths)

        # We apply the hashing
        prefix = arxiv_ids_hashing(arxiv_ids=arxiv_ids)
        # Build the paragraphs
        paragraph_path = f"{dest_dir}/output/paragraphs/{prefix}/text_nodes.jsonl"
        with open(paragraph_path) as f:
            data = [json.loads(line) for line in f]

        # Use arxiv_id + section name as key
        # Find the smallest paragraph_id generated by knowledge debugger
        # Subtract all paragraph id of the same section (of the same paper) with the smallest one to ensure that order starts with zero

        section_min_paragraph = {}

        for paragraph in data:
            paragraph_id = paragraph.get('id')
            # Extract paragraph_id
            id_number = get_paragraph_num(paragraph_id)
            paper_arxiv_id = paragraph.get('paper_id')
            paper_section = paragraph.get('section')
            if (paper_arxiv_id, paper_section) not in section_min_paragraph:
                section_min_paragraph[(paper_arxiv_id, paper_section)] = int(id_number)
            else:
                section_min_paragraph[(paper_arxiv_id, paper_section)] = min(section_min_paragraph[(paper_arxiv_id, paper_section)], int(id_number))

        for paragraph in data:
            paragraph_id = paragraph.get('id')
            content = paragraph.get('content')
            paper_arxiv_id = paragraph.get('paper_id')
            paper_section = paragraph.get('section')
            id_number = get_paragraph_num(paragraph_id)
            id_zero_based = id_number - section_min_paragraph[(paper_arxiv_id, paper_section)]
            self.insert_paragraph(paragraph_id=id_zero_based, content=content, paper_arxiv_id=paper_arxiv_id, paper_section=paper_section)


    def construct_table_from_csv(self, csv_file):
        """
        Construct the paragraphs table from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file containing paragraph data
            
        Expected CSV format:
            - Required columns: paragraph_id, content, paper_arxiv_id, paper_section
            - Optional columns: section_id, paragraph_in_paper_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['paragraph_id', 'content', 'paper_arxiv_id', 'paper_section']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            for col in ['section_id', 'paragraph_in_paper_id']:
                if col not in external_df.columns:
                    external_df[col] = None

            # Generate IDs for new paragraphs
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Filter out paragraphs that already exist (based on paragraph_id, paper_arxiv_id, and paper_section)
            if not current_df.empty:
                existing_tuples = set(zip(
                    current_df['paragraph_id'], 
                    current_df['paper_arxiv_id'], 
                    current_df['paper_section']
                ))
                external_df['_tuple'] = list(zip(
                    external_df['paragraph_id'],
                    external_df['paper_arxiv_id'],
                    external_df['paper_section']
                ))
                external_df = external_df[~external_df['_tuple'].isin(existing_tuples)]
                external_df = external_df.drop(columns=['_tuple'])

            if external_df.empty:
                print("No new paragraphs to import (all paragraphs already exist)")
                return True

            # Ensure correct column order
            external_df = external_df[['id', 'paragraph_id', 'content', 'paper_arxiv_id', 'paper_section', 'section_id', 'paragraph_in_paper_id']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paragraphs from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paragraphs from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the paragraphs table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file containing paragraph data
            
        Expected JSON format:
            [
                {
                    "paragraph_id": 0,
                    "content": "This paper introduces the Transformer...",
                    "paper_arxiv_id": "1706.03762v7",
                    "paper_section": "introduction",
                    "section_id": 1,
                    "paragraph_in_paper_id": 0
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
                if 'paragraphs' in json_data:
                    paragraphs_list = json_data['paragraphs']
                else:
                    paragraphs_list = [json_data]
            elif isinstance(json_data, list):
                paragraphs_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not paragraphs_list:
                print("Error: No paragraph data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(paragraphs_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['paragraph_id', 'content', 'paper_arxiv_id', 'paper_section']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            for col in ['section_id', 'paragraph_in_paper_id']:
                if col not in external_df.columns:
                    external_df[col] = None

            # Generate IDs for new paragraphs
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Filter out paragraphs that already exist
            if not current_df.empty:
                existing_tuples = set(zip(
                    current_df['paragraph_id'],
                    current_df['paper_arxiv_id'],
                    current_df['paper_section']
                ))
                external_df['_tuple'] = list(zip(
                    external_df['paragraph_id'],
                    external_df['paper_arxiv_id'],
                    external_df['paper_section']
                ))
                external_df = external_df[~external_df['_tuple'].isin(existing_tuples)]
                external_df = external_df.drop(columns=['_tuple'])

            if external_df.empty:
                print("No new paragraphs to import (all paragraphs already exist)")
                return True

            # Ensure correct column order
            external_df = external_df[['id', 'paragraph_id', 'content', 'paper_arxiv_id', 'paper_section', 'section_id', 'paragraph_in_paper_id']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paragraphs from {json_file}")
            return True

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paragraphs from JSON: {e}")
            return False
