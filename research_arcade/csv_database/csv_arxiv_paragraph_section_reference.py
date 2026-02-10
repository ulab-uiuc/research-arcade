"""
The edge between paragraph and section where the section is referenced by the paragraph,
in contrast to the edge between paragraph and section where the section is the parent of the paragraph.

This tracks cross-sectional references like "as shown in Section 3" or "see Section 2.1".
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json
from ..arxiv_utils.utils import get_paragraph_num
from ..arxiv_utils.get_paragraph_global_id import get_paragraph_global_id_csv


class CSVArxivParagraphSectionReference:
    def __init__(self, csv_dir: str):
        self.csv_dir = csv_dir
        csv_path = os.path.join(csv_dir, 'arxiv_paragraph_section_references.csv')
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paragraph_section_references_table()

    def create_paragraph_section_references_table(self):
        df = pd.DataFrame(columns=[
            'id', 'paragraph_id', 'referenced_section_id',
            'paper_arxiv_id', 'paper_section_id', 'paragraph_global_id'
        ])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paragraph_section_references CSV at {self.csv_path}")

    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            dtype_map = {
                "paper_arxiv_id": str
            }
            df = pd.read_csv(self.csv_path, dtype=dtype_map)
            return df
        return pd.DataFrame()

    def _save_data(self, df):
        df.to_csv(self.csv_path, index=False)

    def insert_paragraph_section_reference(
        self, 
        paragraph_id, 
        referenced_section_id, 
        paper_arxiv_id, 
        paper_section_id, 
        paragraph_global_id=None
    ) -> Optional[int]:
        df = self._load_data()
        new_id = df['id'].max() + 1 if not df.empty else 1

        if paragraph_global_id is None:
            paragraph_global_id = get_paragraph_global_id_csv(
                csv_dir=self.csv_dir,
                paper_arxiv_id=paper_arxiv_id,
                paper_section_id=paper_section_id,
                paragraph_id=paragraph_id
            )

        new_row = pd.DataFrame([{
            'id': new_id,
            'paragraph_id': paragraph_id,
            'referenced_section_id': referenced_section_id,
            'paper_arxiv_id': paper_arxiv_id,
            'paper_section_id': paper_section_id,
            'paragraph_global_id': paragraph_global_id
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def get_all_paragraph_section_references(self) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()

    def get_paragraph_neighboring_referenced_sections(self, paragraph_id: int) -> Optional[pd.DataFrame]:
        """Get all sections referenced by a given paragraph."""
        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paragraph_id'] == paragraph_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)

    def get_paragraph_global_id_neighboring_referenced_sections(self, paragraph_global_id: int) -> Optional[pd.DataFrame]:
        """Get all sections referenced by a given paragraph (using global ID)."""
        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paragraph_global_id'] == paragraph_global_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)

    def get_section_neighboring_referencing_paragraphs(self, referenced_section_id: int) -> Optional[pd.DataFrame]:
        """Get all paragraphs that reference a given section."""
        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['referenced_section_id'] == referenced_section_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)

    def delete_paragraph_section_reference_by_section_id(self, referenced_section_id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = (df['referenced_section_id'] == referenced_section_id)
        
        if not mask.any():
            return False
        
        df = df[~mask]
        self._save_data(df)
        
        return True

    def delete_paragraph_section_reference_by_paragraph_id(self, paragraph_id: int) -> int:
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

    def delete_paragraph_section_reference_by_paragraph_global_id(self, paragraph_global_id: int) -> int:
        df = self._load_data()
        
        if df.empty:
            return 0
        
        mask = df['paragraph_global_id'] == paragraph_global_id
        count = mask.sum()
        
        if count == 0:
            return 0
        
        df = df[~mask]
        self._save_data(df)
        
        return count

    def delete_paragraph_section_reference_by_id(self, id: int) -> bool:
        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = df['id'] == id
        
        if not mask.any():
            return False
        
        df = df[~mask]
        self._save_data(df)
        
        return True

    def delete_paragraph_section_reference_by_paragraph_section_id(
        self, 
        paragraph_id: int, 
        referenced_section_id: int
    ) -> int:
        df = self._load_data()
        
        if df.empty:
            return 0

        mask = (df['paragraph_id'] == paragraph_id) & (df['referenced_section_id'] == referenced_section_id)

        count = mask.sum()

        if count == 0:
            return 0

        df = df[~mask]
        self._save_data(df)
        
        return count

    def construct_table_from_csv(self, csv_file, dest_dir=None):
        """
        Construct the paragraph-section reference relationships from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Expected CSV format:
            - Required columns: paragraph_id, referenced_section_id,
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

            required_cols = ['paragraph_id', 'referenced_section_id', 'paper_arxiv_id', 'paper_section_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Generate IDs for new entries
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Add paragraph_global_id if not present
            if 'paragraph_global_id' not in external_df.columns:
                external_df['paragraph_global_id'] = None

            # Ensure correct column order
            external_df = external_df[['id', 'paragraph_id', 'referenced_section_id', 
                                       'paper_arxiv_id', 'paper_section_id', 'paragraph_global_id']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paragraph-section references from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing paragraph-section references from CSV: {e}")
            return False

    def construct_table_from_json(self, json_file):
        """
        Construct the paragraph-section reference relationships from an external JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Expected JSON format:
            [
                {
                    "paragraph_id": 1,
                    "referenced_section_id": 5,
                    "paper_arxiv_id": "1706.03762v7",
                    "paper_section_id": 2,
                    "paragraph_global_id": 123  # optional
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
                if 'paragraph_section_references' in json_data:
                    relations_list = json_data['paragraph_section_references']
                else:
                    relations_list = [json_data]
            elif isinstance(json_data, list):
                relations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not relations_list:
                print("Error: No paragraph-section reference data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(relations_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['paragraph_id', 'referenced_section_id', 'paper_arxiv_id', 'paper_section_id']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Generate IDs for new entries
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Add paragraph_global_id if not present
            if 'paragraph_global_id' not in external_df.columns:
                external_df['paragraph_global_id'] = None

            # Ensure correct column order
            external_df = external_df[['id', 'paragraph_id', 'referenced_section_id',
                                       'paper_arxiv_id', 'paper_section_id', 'paragraph_global_id']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} paragraph-section references from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing paragraph-section references from JSON: {e}")
            return False

    def construct_paragraph_section_references_table_from_api(self, arxiv_ids, dest_dir=None):
        """
        Construct paragraph-section reference edges from preprocessed data.
        
        First select all refs in the paragraph-reference that are sections.
        Then for these refs, we match them with their corresponding paragraphs and referenced sections.
        Finally we save the edges.
        
        Args:
            arxiv_ids: List of arXiv paper IDs to process
            dest_dir: Optional destination directory (unused, kept for API consistency)
        """

        # Load required CSV files
        para_ref_path = f"{self.csv_dir}/arxiv_paragraph_references.csv"
        paragraph_path = f"{self.csv_dir}/arxiv_paragraphs.csv"
        section_path = f"{self.csv_dir}/arxiv_sections.csv"

        df_para_ref = pd.read_csv(para_ref_path, dtype={'paper_arxiv_id': str})
        df_paragraphs = pd.read_csv(paragraph_path, dtype={'paper_arxiv_id': str})
        df_sections = pd.read_csv(section_path, dtype={'paper_arxiv_id': str})
        
        for arxiv_id in arxiv_ids:
            # Filter for section references in this paper
            result = df_para_ref[
                (df_para_ref['paper_arxiv_id'] == arxiv_id) & 
                (df_para_ref['reference_type'] == 'section')
            ].copy()

            for idx, para_section_ref in result.iterrows():
                paragraph_id = para_section_ref['paragraph_id']
                paper_section = para_section_ref['paper_section']  # The section containing the paragraph
                reference_label = para_section_ref['reference_label']  # The referenced section label
                paragraph_global_id = para_section_ref.get('paragraph_global_id', None)
                    
                # Get paragraph global ID if not provided
                if paragraph_global_id is None:
                    paragraph_result = df_paragraphs[
                        (df_paragraphs['paper_arxiv_id'] == arxiv_id) & 
                        (df_paragraphs['paper_section'] == paper_section) & 
                        (df_paragraphs['paragraph_id'] == paragraph_id)
                    ]
                    
                    if paragraph_result.empty:
                        print(f"Warning: Paragraph not found for arxiv_id={arxiv_id}, "
                              f"section={paper_section}, para_id={paragraph_id}")
                        continue
                    
                    paragraph_global_id = paragraph_result.iloc[0]['id']
                
                # Get the section ID of the section containing this paragraph
                section_result = df_sections[
                    (df_sections['paper_arxiv_id'] == arxiv_id) & 
                    (df_sections['title'] == paper_section)
                ]
                
                if section_result.empty:
                    print(f"Warning: Section not found for arxiv_id={arxiv_id}, section={paper_section}")
                    continue
                
                paper_section_id = section_result.iloc[0]['id']
                
                # ============================================================
                # TODO: PLACEHOLDER - Find the referenced section ID
                # ============================================================
                # The reference_label could be in various formats:
                # - "sec:introduction", "sec:methods", "sec:3"
                # - "Section 3", "Section 3.1"
                # - Just a number "3" or "3.1"
                #
                # You need to implement logic to match reference_label to the 
                # actual section in the paper. Some approaches:
                #
                # 1. If sections have a 'label' column:
                #    referenced_section = df_sections[
                #        (df_sections['paper_arxiv_id'] == arxiv_id) & 
                #        (df_sections['label'] == reference_label)
                #    ]
                #
                # 2. If reference_label is a section number, match by order:
                #    import re
                #    section_num = re.search(r'(\d+(?:\.\d+)?)', reference_label)
                #    if section_num:
                #        # Match by section_order or parse title
                #        pass
                #
                # 3. If reference_label contains section title keywords:
                #    referenced_section = df_sections[
                #        (df_sections['paper_arxiv_id'] == arxiv_id) & 
                #        (df_sections['title'].str.contains(reference_label, case=False))
                #    ]
                #
                # For now, we'll try a simple label match:
                
                # Construct possible label formats to search for
                label_variants = [
                    f"\\label{{{reference_label}}}",
                    reference_label,
                    f"sec:{reference_label}",
                ]
                
                referenced_section_id = None
                
                # Try matching by label if sections have a label column
                if 'label' in df_sections.columns:
                    for label_variant in label_variants:
                        referenced_section = df_sections[
                            (df_sections['paper_arxiv_id'] == arxiv_id) & 
                            (df_sections['label'] == label_variant)
                        ]
                        if not referenced_section.empty:
                            referenced_section_id = referenced_section.iloc[0]['id']
                            break
                
                # Fallback: try matching by title
                if referenced_section_id is None:
                    referenced_section = df_sections[
                        (df_sections['paper_arxiv_id'] == arxiv_id) & 
                        (df_sections['title'].str.lower().str.contains(
                            reference_label.lower().replace('sec:', '').strip(), 
                            na=False
                        ))
                    ]
                    if not referenced_section.empty:
                        referenced_section_id = referenced_section.iloc[0]['id']
                
                if referenced_section_id is None:
                    print(f"Warning: Referenced section not found for label={reference_label}, "
                          f"arxiv_id={arxiv_id}")
                    continue
                # ============================================================
                # END PLACEHOLDER
                # ============================================================
                
                self.insert_paragraph_section_reference(
                    paragraph_id=paragraph_id,
                    referenced_section_id=referenced_section_id,
                    paper_arxiv_id=arxiv_id,
                    paper_section_id=paper_section_id,
                    paragraph_global_id=paragraph_global_id
                )

    def get_references_by_paper(self, paper_arxiv_id: str) -> Optional[pd.DataFrame]:
        """Get all paragraph-section references for a specific paper."""
        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paper_arxiv_id'] == paper_arxiv_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)

    def get_references_from_section(self, paper_section_id: int) -> Optional[pd.DataFrame]:
        """Get all section references made from paragraphs within a given section."""
        df = self._load_data()
        
        if df.empty:
            return None
        
        result = df[df['paper_section_id'] == paper_section_id].copy()
        
        if result.empty:
            return None
        
        return result.reset_index(drop=True)
