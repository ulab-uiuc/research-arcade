"""

"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json
from ..arxiv_utils.utils import get_paragraph_num



class CSVArxivParagraphTable:
    def __init__(self, csv_dir: str):
        self.csv_dir = csv_dir
        csv_path = f"{csv_dir}/arxiv_paragraph_tables.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paragraph_tables_table()

    def create_paragraph_tables_table(self):
        df = pd.DataFrame(columns=[
            'id', 'paragraph_id', 'table_id',
            'paper_arxiv_id', 'paper_section_id'
        ])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paragraph_tables CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_paragraph_table_table(self, paragraph_id, table_id, paper_arxiv_id, paper_section_id):
        df = self._load_data()
        new_id = df['id'].max() + 1 if not df.empty else 1
        new_row = pd.DataFrame([{
            'id': new_id, 'paragraph_id': paragraph_id,
            'table_id': table_id, 'paper_arxiv_id': paper_arxiv_id,
            'paper_section_id': paper_section_id
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def get_all_paragraph_tables(self):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()


    def get_paragraph_neighboring_tables(self, paragraph_id: int) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paragraph_id'] == paragraph_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def get_table_neighboring_paragraphs(self, table_id: int) -> Optional[pd.DataFrame]:

        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['table_id'] == table_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)


    def delete_paragraph_table_by_table_id(self, table_id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = (df['table_id'] == table_id)
        
        if not mask.any():
            return False
        
        df = df[~mask]
        self._save_data(df)
        
        return True


    def delete_paragraph_table_by_paragraph_id(self, paragraph_id: int) -> int:

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

    def delete_paragraph_table_by_paragraph_table_id(self, paragrapg_id: int, table_id: int) -> int:

        df = self._load_data()
        
        if df.empty:
            return 0

        mask = (df['paragraph_id'] == paragrapg_id) & (df['table_id'] == table_id)

        count = mask.sum()

        if count == 0:
            return 0

        df = df[~mask]
        self._save_data(df)
        
        return count



    def construct_table_from_csv(self, csv_file):
        """
        Construct the paragraph-table relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paragraph_id, table_id,
            paper_arxiv_id, paper_section_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['id', 'paragraph_id', 'table_id', 'paper_arxiv_id', 'paper_section_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Generate IDs for new tables
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Note: Not filtering duplicates as this table allows multiple tables per paragraph

            # Ensure correct column order
            external_df = external_df[['id', 'paragraph_id', 'table_id', 'paper_arxiv_id', 'paper_section_id']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paragraph-table relationships from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paragraph-table relationships from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the paragraph-table relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {
                    "paragraph_id": 1,
                    "paper_section": "introduction",
                    "paper_arxiv_id": "1706.03762v7",
                    "table_id": "xxxxx",
                    "paper_section_id": "xxxxx"
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
                if 'paragraph_tables' in json_data:
                    relations_list = json_data['paragraph_tables']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paragraph-table data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(relations_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['paragraph_id', 'table_id',
            'paper_arxiv_id', 'paper_section_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Generate IDs for new tables
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Ensure correct column order
            external_df = external_df[['id', 'paragraph_id', 'table_id',
            'paper_arxiv_id', 'paper_section_id']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paragraph-table relationships from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paragraph-table relationships from JSON: {e}")
            return False



    def construct_paragraph_tables_table_from_api(self, arxiv_ids, dest_dir=None):
        """
        Assume that we already have preprocessed paragraph-reference edges and table nodes in the dataset.

        First select all refs in the paragraph-reference that are tables
        Then for these refs, we match them with their corresponding paragraphs and tables.

        Finally we can save the edges.
        """

        # First open the reference table
        para_ref_path = f"{self.csv_dir}/arxiv_paragraph_references.csv"
        paragraph_path = f"{self.csv_dir}/arxiv_paragraphs.csv"
        table_path = f"{self.csv_dir}/arxiv_tables.csv"
        section_path = f"{self.csv_dir}/arxiv_sections.csv"

        df2 = pd.read_csv(para_ref_path, dtype={'paper_arxiv_id': str})
        df3 = pd.read_csv(paragraph_path, dtype={'paper_arxiv_id': str})
        df4 = pd.read_csv(table_path, dtype={'paper_arxiv_id': str})
        df5 = pd.read_csv(section_path, dtype={'paper_arxiv_id': str})
        
        for arxiv_id in arxiv_ids:
            result = df2[(df2['paper_arxiv_id'] == arxiv_id) & (df2['reference_type'] == 'table')].copy()

            # Find two things: One is the exact paragraph, the other is the exact table
            for idx, para_table in result.iterrows():
                paragraph_id = para_table['paragraph_id']
                paper_section = para_table['paper_section']
                reference_label = para_table['reference_label']
                
                # Construct label - adjust format based on your actual data
                label = f"\\label{{{reference_label}}}"
                
                # First search for the global paragraph id
                paragraph_result = df3[(df3['paper_arxiv_id'] == arxiv_id) & 
                                    (df3['paper_section'] == paper_section) & 
                                    (df3['paragraph_id'] == paragraph_id)]
                
                if paragraph_result.empty:
                    print(f"Warning: Paragraph not found for arxiv_id={arxiv_id}, section={paper_section}, para_id={paragraph_id}")
                    continue
                
                global_paragraph_id = paragraph_result.iloc[0]['id']
                
                # Then we fetch table id
                table_result = df4[(df4['label'] == label) & (df4['paper_arxiv_id'] == arxiv_id)]
                
                if table_result.empty:
                    print(f"Warning: Figure not found for label={label}, arxiv_id={arxiv_id}")
                    continue
                
                table_id = table_result.iloc[0]['id']
                
                # We also need to search for the paper_section id
                section_result = df5[(df5['paper_arxiv_id'] == arxiv_id) & 
                                    (df5['title'] == paper_section)]
                
                if section_result.empty:
                    print(f"Warning: Section not found for arxiv_id={arxiv_id}, section={paper_section}")
                    continue
                
                section_id = section_result.iloc[0]['id']
                
                self.insert_paragraph_table_table(
                    paragraph_id=global_paragraph_id, 
                    table_id=table_id, 
                    paper_arxiv_id=arxiv_id, 
                    paper_section_id=section_id
                )