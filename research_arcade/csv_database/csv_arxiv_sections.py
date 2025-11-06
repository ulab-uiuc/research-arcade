import pandas as pd
import os
from typing import Optional
from pathlib import Path
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor

class CSVArxivSections:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_sections.csv"
        self.csv_path = csv_path
        # Set up the target directory
        # Automatically create the csv path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_sections_table()

    def create_sections_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'content', 'title', 'appendix', 'paper_arxiv_id'
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


    def insert_section(self, content, title, appendix, paper_arxiv_id, section_in_paper_id):
        """Insert a section. Returns the generated section id."""
        df = self._load_data()
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'content': content,
            'title': title,
            'appendix': appendix,
            'paper_arxiv_id': paper_arxiv_id,
            'section_in_paper_id': section_in_paper_id
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def delete_section_by_id(self, id):
        """Delete a section by its id. Returns True if deleted, False if not found."""
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        df = df[df['id'] != id]
        self._save_data(df)
        return True
    
    def delete_section_by_paper_arxiv_id(self, paper_arxiv_id):
        """Delete all sections for a specific paper arxiv_id. Returns the number of sections deleted."""
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns:
            return 0
        
        initial_count = len(df)
        
        df = df[df['paper_arxiv_id'] != paper_arxiv_id]
        deleted_count = initial_count - len(df)
        
        self._save_data(df)
        return deleted_count
    
    def update_section(self, id, content=None, title=None, is_appendix=None, paper_arxiv_id=None):
        """Update a section by id. Returns True if updated, False if not found."""
        df = self._load_data()
        
        if id not in df['id'].values:
            return False
        
        mask = df['id'] == id
        
        if content is not None:
            df.loc[mask, 'content'] = content
        if title is not None:
            df.loc[mask, 'title'] = title
        if is_appendix is not None:
            df.loc[mask, 'appendix'] = is_appendix
        if paper_arxiv_id is not None:
            df.loc[mask, 'paper_arxiv_id'] = paper_arxiv_id
        
        self._save_data(df)
        return True
    
    def get_section_by_id(self, id: int) -> Optional[pd.DataFrame]:
        """Get a section by its id. Returns a DataFrame with the section or None if not found."""
        df = self._load_data()
        
        if df.empty or id not in df['id'].values:
            return None
        
        section = df[df['id'] == id]
        return section

    def get_sections_by_arxiv_id(self, arxiv_id: str) -> Optional[pd.DataFrame]:
        """Get all sections for a paper by its arxiv_id. Returns a DataFrame with sections or None if not found."""
        df = self._load_data()
        
        if df.empty or 'paper_arxiv_id' not in df.columns:
            return None
        
        sections = df[df['paper_arxiv_id'] == arxiv_id]
        
        if sections.empty:
            return None
        
        return sections
    
    def check_section_exists(self, id: int) -> bool:
        """Check if a section exists by its id."""
        df = self._load_data()
        
        if df.empty:
            return False
        
        return id in df['id'].values
    
    def construct_sections_table_from_csv(self, csv_file: str):
        """
        Construct the sections table from an external CSV file.
        Assumes the CSV has compatible columns or can be mapped to the sections schema.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False
        
        external_df = pd.read_csv(csv_file)
        current_df = self._load_data()
        
        required_cols = ['content', 'title', 'appendix', 'paper_arxiv_id']
        missing_cols = [col for col in required_cols if col not in external_df.columns]
        
        if missing_cols:
            print(f"Error: External CSV is missing required columns: {missing_cols}")
            return False
        
        start_id = current_df['id'].max() + 1 if not current_df.empty else 1
        external_df['id'] = range(start_id, start_id + len(external_df))
        
        combined_df = pd.concat([current_df, external_df], ignore_index=True)
        self._save_data(combined_df)
        
        print(f"Successfully imported {len(external_df)} sections from {csv_file}")
        return True
    
    def get_all_sections(self, is_all_features=True):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()


    def construct_sections_table_from_api(self, arxiv_ids, dest_dir):
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
                    section_jsons = file_json['sections']

                    i = 0
                    for title, section_json in section_jsons.items():
                        i += 1
                        is_appendix = section_json['appendix'] == 'true'
                        content = section_json['content']
                        self.insert_section(content=content, title=title, appendix=is_appendix, paper_arxiv_id=arxiv_id, section_in_paper_id=i)

            except FileNotFoundError:
                print(f"Error: The file '{file_json}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{file_json}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue


    def construct_table_from_csv(self, csv_file):
        """
        Construct the sections table from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file containing section data
            
        Expected CSV format:
            - Required columns: content, title, appendix, paper_arxiv_id
            - Optional columns: section_in_paper_id
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['content', 'title', 'appendix', 'paper_arxiv_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            if 'section_in_paper_id' not in external_df.columns:
                external_df['section_in_paper_id'] = None

            # Generate IDs for new sections
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Note: Not filtering for duplicates as sections can be re-imported

            # Ensure correct column order
            external_df = external_df[['id', 'content', 'title', 'appendix', 'paper_arxiv_id', 'section_in_paper_id']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} sections from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing sections from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the sections table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file containing section data
            
        Expected JSON format:
            [
                {
                    "content": "Section content...",
                    "title": "Introduction",
                    "appendix": false,
                    "paper_arxiv_id": "1706.03762v7",
                    "section_in_paper_id": 1
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
                if 'sections' in json_data:
                    sections_list = json_data['sections']
                else:
                    sections_list = [json_data]
            elif isinstance(json_data, list):
                sections_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not sections_list:
                print("Error: No section data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(sections_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['content', 'title', 'appendix', 'paper_arxiv_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            if 'section_in_paper_id' not in external_df.columns:
                external_df['section_in_paper_id'] = None

            # Generate IDs for new sections
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Ensure correct column order
            external_df = external_df[['id', 'content', 'title', 'appendix', 'paper_arxiv_id', 'section_in_paper_id']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} sections from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing sections from JSON: {e}")
            return False

