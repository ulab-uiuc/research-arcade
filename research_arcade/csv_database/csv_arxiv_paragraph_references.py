import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json
from ..arxiv_utils.utils import get_paragraph_num, arxiv_ids_hashing


class CSVArxivParagraphReference:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paragraph_references.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paragraph_references_table()

    def create_paragraph_references_table(self):
        df = pd.DataFrame(columns=[
            'id', 'paragraph_id', 'paper_section',
            'paper_arxiv_id', 'reference_label', 'reference_type'
        ])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paragraph_references CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_paragraph_reference(self, paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type):
        df = self._load_data()
        new_id = df['id'].max() + 1 if not df.empty else 1
        new_row = pd.DataFrame([{
            'id': new_id, 'paragraph_id': paragraph_id,
            'paper_section': paper_section, 'paper_arxiv_id': paper_arxiv_id,
            'reference_label': reference_label, 'reference_type': reference_type
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def get_all_paragraph_references(self):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()


    def get_paragraph_neighboring_references(self, paragraph_id: int) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paragraph_id'] == paragraph_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def get_reference_neighboring_paragraphs(self, reference_id: int) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['id'] == reference_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def delete_paragraph_reference_by_id(self, paragraph_id: int, reference_id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = (df['paragraph_id'] == paragraph_id) & (df['id'] == reference_id)
        
        if not mask.any():
            return False
        
        df = df[~mask]
        self._save_data(df)
        
        return True


    def delete_paragraph_reference_by_paragraph_id(self, paragraph_id: int) -> int:

        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['paragraph_id'] == paragraph_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count

    def delete_paragraph_reference_by_reference_id(self, reference_id: int) -> int:

        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['id'] == reference_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count


    def construct_table_from_csv(self, csv_file):
        """
        Construct the paragraph-reference relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paragraph_id, paper_section, paper_arxiv_id, reference_label, reference_type
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['paragraph_id', 'paper_section', 'paper_arxiv_id', 'reference_label', 'reference_type']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Generate IDs for new references
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Note: Not filtering duplicates as this table allows multiple references per paragraph

            # Ensure correct column order
            external_df = external_df[['id', 'paragraph_id', 'paper_section', 'paper_arxiv_id', 'reference_label', 'reference_type']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paragraph-reference relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paragraph-reference relationships from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the paragraph-reference relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {
                    "paragraph_id": 1,
                    "paper_section": "introduction",
                    "paper_arxiv_id": "1706.03762v7",
                    "reference_label": "fig:1",
                    "reference_type": "figure"
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
                if 'paragraph_references' in json_data:
                    relations_list = json_data['paragraph_references']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paragraph-reference data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(relations_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['paragraph_id', 'paper_section', 'paper_arxiv_id', 'reference_label', 'reference_type']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Generate IDs for new references
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Ensure correct column order
            external_df = external_df[['id', 'paragraph_id', 'paper_section', 'paper_arxiv_id', 'reference_label', 'reference_type']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paragraph-reference relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paragraph-reference relationships from JSON: {e}")
            return False

    def construct_paragraph_references_table_from_api(self, arxiv_ids, dest_dir):
        # Assume that the paragraph has already been processed
        prefix = arxiv_ids_hashing(arxiv_ids=arxiv_ids)
        # Build the paragraphs
        paragraph_path = f"{dest_dir}/output/paragraphs/{prefix}/text_nodes.jsonl"


        with open(paragraph_path) as f:
            data = [json.loads(line) for line in f]
        
        section_min_paragraph = {}

        # First pass
        for paragraph in data:
            paragraph_id = paragraph.get("id")
            paper_arxiv_id = paragraph.get("paper_id")
            paper_section = paragraph.get("section")

            if not paragraph_id or not paper_arxiv_id or not paper_section:
                continue

            id_number = get_paragraph_num(paragraph_id)
            key = (paper_arxiv_id, paper_section)

            if key not in section_min_paragraph:
                section_min_paragraph[key] = int(id_number)
            else:
                section_min_paragraph[key] = min(section_min_paragraph[key], int(id_number))

        # Second pass
        for paragraph in data:
            paragraph_id = paragraph.get("id")
            paper_arxiv_id = paragraph.get("paper_id")
            paper_section = paragraph.get("section")

            if not paragraph_id or not paper_arxiv_id or not paper_section:
                continue

            key = (paper_arxiv_id, paper_section)
            if key not in section_min_paragraph:
                continue

            id_number = get_paragraph_num(paragraph_id)
            id_zero_based = id_number - section_min_paragraph[key]

            paragraph_ref_labels = paragraph.get('ref_labels') or []
            for ref_label in paragraph_ref_labels:

                ref_type = None
                # First search bib_key in databases.
                # If presented in one of them, we can determine the type of reference

                is_figure = ref_label.startswith("figure")
                is_table = ref_label.startswith("table")
                if is_figure:
                    ref_type = 'figure'
                elif is_table:
                    ref_type = 'table'
                self.insert_paragraph_reference(paragraph_id=id_zero_based, paper_section=paper_section, paper_arxiv_id=paper_arxiv_id,reference_label=ref_label,reference_type=ref_type)
