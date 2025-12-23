import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json
import sys
from rapidfuzz import fuzz, process
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.citation_processing import paper_citation_crawling, normalize_title

class CSVArxivCitation:
    def __init__(self, csv_dir: str):
        csv_path = os.path.join(csv_dir, 'arxiv_citations.csv')
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_citations_table()

    def create_citations_table(self):
        df = pd.DataFrame(columns=[
            'id', 'citing_arxiv_id', 'cited_arxiv_id',
            'bib_title', 'bib_key',
            'citing_sections', 'citing_paragraphs'
        ])
        df.to_csv(self.csv_path, index=False)
        print(f"Created citations CSV at {self.csv_path}")

    def _load_data(self): 
        dtype_map = {
            "citing_arxiv_id": str,
            "cited_arxiv_id": str,
            "bib_key": str
        }
        # return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
        return pd.read_csv(self.csv_path, dtype=dtype_map) if os.path.exists(self.csv_path) else pd.DataFrame()
    
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_citation(self, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, citing_sections=None):

        if citing_arxiv_id == cited_arxiv_id:
            return False
        df = self._load_data()
        conflict = df[
            (df['citing_arxiv_id'] == citing_arxiv_id) &
            (df['cited_arxiv_id'] == cited_arxiv_id)
        ]
        if not conflict.empty:
            return False
        new_id = df['id'].max() + 1 if not df.empty else 1
        citing_sections_str = json.dumps(citing_sections) if citing_sections else '[]'
        new_row = pd.DataFrame([{
            'id': new_id, 'citing_arxiv_id': citing_arxiv_id,
            'cited_arxiv_id': cited_arxiv_id, 'bib_title': bib_title,
            'bib_key': bib_key, 
            'citing_sections': citing_sections_str, 'citing_paragraphs': '[]'
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True

    # def construct_citations_table_from_api(self, arxiv_ids, dest_dir):
    #     # Check if papers already exists in the directory
    #     downloaded_paper_ids = []
    #     for arxiv_id in arxiv_ids:
    #         paper_dir = f"{dest_dir}/{arxiv_id}/{arxiv_id}_metadata.json"

    #         if not os.path.exists(paper_dir):
    #             downloaded_paper_ids.append(arxiv_id)

    #     for arxiv_id in downloaded_paper_ids:
    #         md = MultiDownload()
    #         try:
    #             md.download_arxiv(input=arxiv_id, input_type = "id", output_type="latex", dest_dir=self.dest_dir)
    #             print(f"paper with id {arxiv_id} downloaded")
    #             downloaded_paper_ids.append(arxiv_id)
    #         except RuntimeError as e:
    #             print(f"[ERROR] Failed to download {arxiv_id}: {e}")
    #             continue
        
    #     for arxiv_id in arxiv_ids:

    #         json_path = f"{dest_dir}/output/{arxiv_id}.json"
    #         if not os.path.exists(json_path):
    #             # arxiv_id_graph.append(arxiv_id)
    #             try:
    #                 # Build corresponding graph
    #                 md.build_paper_graph(
    #                     input=arxiv_id,
    #                     input_type="id",
    #                     dest_dir=dest_dir
    #                 )
    #             except Exception as e:
    #                 print(f"[Warning] Failed to process papers: {e}")
    #                 continue

    #         try:
    #             with open(json_path, 'r') as file:
    #                 file_json = json.load(file)
    #                 for citation in file_json['citations'].values():
    #                     # print(f"Citation: {citation}")
    #                     cited_arxiv_id = citation.get('arxiv_id')
    #                     bib_key = citation.get('bib_key')
    #                     bib_title = citation.get('bib_title')
    #                     bib_author = citation.get('bib_author ')
    #                     contexts = citation.get('context')
    #                     citing_sections = set()
    #                     for context in contexts:
    #                         citing_section = context['section']
    #                         citing_sections.add(citing_section)

    #                     self.insert_citation(citing_arxiv_id=arxiv_id, cited_arxiv_id=cited_arxiv_id, citing_sections=list(citing_sections), bib_title=bib_title, bib_key=bib_key)

    #         except FileNotFoundError:
    #             print(f"Error: The file '{file_json}' was not found.")
    #             continue
    #         except json.JSONDecodeError:
    #             print(f"Error: Could not decode JSON from '{file_json}'. Check if the file contains valid JSON.")
    #             continue
    #         except Exception as e:
    #             print(f"An unexpected error occurred: {e}")
    #             continue


    def delete_citation_by_id(self, citing_paper_id, cited_paper_id):

        df = self._load_data()
        
        if df.empty:
            return False
        
        mask = (df['citing_arxiv_id'] == citing_paper_id) & (df['cited_arxiv_id'] == cited_paper_id)
        
        if not df[mask].empty:

            df = df[~mask]
            self._save_data(df)
            print(f"Deleted citation: {citing_paper_id} -> {cited_paper_id}")
            return True
        else:
            print(f"Citation not found: {citing_paper_id} -> {cited_paper_id}")
            return False
    
    def get_all_citations(self, is_all_features=True):
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()

    
    def get_citing_neighboring_cited(self, citing_paper_id):

        df = self._load_data()
        
        if df.empty:
            return None
        
        citing_citations = df[df['citing_arxiv_id'] == citing_paper_id].copy()
        
        if citing_citations.empty:
            return None
        
        return citing_citations

    def get_cited_neighboring_citing(self, cited_paper_id):
        df = self._load_data()
        
        if df.empty:
            return None
        
        cited_by = df[df['cited_arxiv_id'] == cited_paper_id].copy()
        
        if cited_by.empty:
            return None
        
        return cited_by
        
    def get_bib_title_by_citing_arxiv_id_bib_key(self, citing_arxiv_id, bib_key):
        df = self._load_data()

        if df.empty:
            return None

        # Ensure required columns exist
        required_cols = {"citing_arxiv_id", "bib_key", "bib_title"}
        if not required_cols.issubset(df.columns):
            return None

        # Filter matching rows
        matches = df[
            (df["citing_arxiv_id"] == citing_arxiv_id) &
            (df["bib_key"] == bib_key)
        ]

        if matches.empty:
            return None

        # Return the first matching bib_title
        return matches.iloc[0]["bib_title"]



    def construct_table_from_csv(self, csv_file):
        """
        Construct the citations table from an external CSV file.
        
        Args:
            csv_file: Path to the CSV file containing citation data
            
        Expected CSV format:
            - Required columns: citing_arxiv_id, cited_arxiv_id, bib_title, bib_key
            - Optional columns: citing_sections, citing_paragraphs
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        try:
            external_df = pd.read_csv(csv_file)
            current_df = self._load_data()

            required_cols = ['citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 'bib_key']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: External CSV is missing required columns: {missing_cols}")
                return False

            # Add optional columns if they don't exist
            if 'citing_sections' not in external_df.columns:
                external_df['citing_sections'] = '[]'
            if 'citing_paragraphs' not in external_df.columns:
                external_df['citing_paragraphs'] = '[]'

            # Generate IDs for new citations
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Filter out citations that already exist (based on citing and cited papers)
            if not current_df.empty:
                existing_pairs = set(zip(current_df['citing_arxiv_id'], current_df['cited_arxiv_id']))
                external_df['_pair'] = list(zip(external_df['citing_arxiv_id'], external_df['cited_arxiv_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new citations to import (all citations already exist)")
                return True

            # Ensure correct column order
            external_df = external_df[['id', 'citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 'bib_key', 'citing_sections', 'citing_paragraphs']]

            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} citations from {csv_file}")
            return True
            
        except Exception as e:
            print(f"Error importing citations from CSV: {e}")
            return False


    def construct_table_from_json(self, json_file):
        """
        Construct the citations table from an external JSON file.
        
        Args:
            json_file: Path to the JSON file containing citation data
            
        Expected JSON format:
            [
                {
                    "citing_arxiv_id": "1706.03762v7",
                    "cited_arxiv_id": "1409.0473v7",
                    "bib_title": "Neural Machine Translation",
                    "bib_key": "bahdanau2014neural",
                    "citing_sections": ["introduction", "related_work"],
                    "citing_paragraphs": []
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
                if 'citations' in json_data:
                    citations_list = json_data['citations']
                else:
                    citations_list = [json_data]
            elif isinstance(json_data, list):
                citations_list = json_data
            else:
                print("Error: JSON file must contain either a list or a dictionary")
                return False
            
            if not citations_list:
                print("Error: No citation data found in JSON file")
                return False
            
            # Convert to DataFrame
            external_df = pd.DataFrame(citations_list)
            current_df = self._load_data()

            # Check for required columns
            required_cols = ['citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 'bib_key']
            missing_cols = [col for col in required_cols if col not in external_df.columns]

            if missing_cols:
                print(f"Error: JSON data is missing required fields: {missing_cols}")
                return False

            # Handle optional columns
            if 'citing_sections' not in external_df.columns:
                external_df['citing_sections'] = '[]'
            else:
                # Convert lists to JSON strings
                external_df['citing_sections'] = external_df['citing_sections'].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x
                )
            
            if 'citing_paragraphs' not in external_df.columns:
                external_df['citing_paragraphs'] = '[]'
            else:
                external_df['citing_paragraphs'] = external_df['citing_paragraphs'].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x
                )

            # Generate IDs for new citations
            start_id = current_df['id'].max() + 1 if not current_df.empty else 1
            external_df['id'] = range(start_id, start_id + len(external_df))

            # Filter out citations that already exist
            if not current_df.empty:
                existing_pairs = set(zip(current_df['citing_arxiv_id'], current_df['cited_arxiv_id']))
                external_df['_pair'] = list(zip(external_df['citing_arxiv_id'], external_df['cited_arxiv_id']))
                external_df = external_df[~external_df['_pair'].isin(existing_pairs)]
                external_df = external_df.drop(columns=['_pair'])

            if external_df.empty:
                print("No new citations to import (all citations already exist)")
                return True

            # Ensure correct column order
            external_df = external_df[['id', 'citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 'bib_key', 'citing_sections', 'citing_paragraphs']]
            
            # Combine and save
            combined_df = pd.concat([current_df, external_df], ignore_index=True)
            self._save_data(combined_df)

            print(f"Successfully imported {len(external_df)} citations from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing citations from JSON: {e}")
            return False



    def construct_citations_table_from_api(self, arxiv_ids, dest_dir):

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


            for citation in file_json['citations'].values():
                    # print(f"Citation: {citation}")
                    cited_arxiv_id = citation.get('arxiv_id')
                    bib_key = citation.get('bib_key')
                    bib_title = citation.get('bib_title')
                    bib_author = citation.get('bib_author ')
                    contexts = citation.get('context')
                    citing_sections = set()
                    for context in contexts:
                        citing_section = context['section']
                        citing_sections.add(citing_section)
                
                    #大切なこと
                    self.insert_citation(citing_arxiv_id=arxiv_id, cited_arxiv_id=cited_arxiv_id, citing_sections=list(citing_sections), bib_title=bib_title, bib_key=bib_key)
        

        # Finally, postprocess all the citations that do not have valid cited arxiv id
        result_df = paper_citation_crawling(arxiv_ids=arxiv_ids)
        print("result_df")
        print(result_df)
        self.citation_matching_csv(df_cit=result_df, arxiv_ids=arxiv_ids)
        


    def citation_matching_csv(
        self,
        df_cit,
        arxiv_ids,
        similarity_threshold=95
    ):
        """
        Match citations from Semantic Scholar with bibliography entries from CSV.
        Updates the original CSV file with matched arxiv IDs.
        """
        # Load bibkey from the CSV
        df_bib = self._load_data()
        
        # Convert arxiv_ids to strings for consistent comparison
        arxiv_ids_str = [str(aid) for aid in arxiv_ids]
        
        # Convert citing_arxiv_id column to string
        df_bib['citing_arxiv_id'] = df_bib['citing_arxiv_id'].astype(str)
        
        # Filter to only the arxiv_ids we care about AND where cited_arxiv_id is null/empty
        # Handle both NaN, None, and empty strings
        def is_empty_or_null(x):
            if pd.isna(x):
                return True
            if isinstance(x, str) and x.strip() == '':
                return True
            return False
        
        mask = (
            df_bib['citing_arxiv_id'].isin(arxiv_ids_str) & 
            df_bib['cited_arxiv_id'].apply(is_empty_or_null)
        )
        df_bib_filtered = df_bib[mask].copy()
        
        print(f"DEBUG: Total rows in CSV: {len(df_bib)}")
        print(f"DEBUG: Arxiv IDs to match: {arxiv_ids_str}")
        print(f"DEBUG: Rows with matching citing_arxiv_id: {df_bib['citing_arxiv_id'].isin(arxiv_ids_str).sum()}")
        print(f"DEBUG: Rows with empty cited_arxiv_id: {df_bib['cited_arxiv_id'].apply(is_empty_or_null).sum()}")
        print(f"DEBUG: Filtered rows for matching: {len(df_bib_filtered)}")
        
        if df_bib_filtered.empty:
            print("No bibliography entries found for matching")
            return
        
        df_bib_filtered["norm_bib_title"] = df_bib_filtered["bib_title"].apply(normalize_title)
                
        # Group bib titles by citing_arxiv_id, also track original index for updating
        bib_groups = defaultdict(list)
        for idx, row in df_bib_filtered.iterrows():
            bib_groups[str(row["citing_arxiv_id"])].append({
                'original_idx': idx,
                'bib_title': row["bib_title"],
                'norm_bib_title': row["norm_bib_title"]
            })
        
        # Ensure required columns exist in citation df
        required_cols = ["citing_arxiv_id", "cited_arxiv_id", "cited_paper_name"]
        for col in required_cols:
            if col not in df_cit.columns:
                raise ValueError(f"Missing required column: {col}")

        results = []
        updates = []  # Track updates to make to original CSV
        total = 0
        matched = 0
        no_match = 0

        print(f"Processing {df_cit['citing_arxiv_id'].nunique()} unique citing papers...")
        
        for citing_id, group in df_cit.groupby("citing_arxiv_id"):
            citing_id = str(citing_id)
            if citing_id not in bib_groups:
                continue

            bib_entries = bib_groups[citing_id]
            # Map normalized title -> (original_idx, original_title)
            norm_title_map = {
                entry['norm_bib_title']: (entry['original_idx'], entry['bib_title']) 
                for entry in bib_entries
            }
            norm_title_list = list(norm_title_map.keys())

            for _, row in group.iterrows():
                total += 1
                cited_name = normalize_title(row["cited_paper_name"])

                # Fuzzy match
                best = process.extractOne(
                    cited_name,
                    norm_title_list,
                    scorer=fuzz.ratio,
                    score_cutoff=similarity_threshold
                )

                if best:
                    norm_match, score, _ = best
                    original_idx, matched_title = norm_title_map[norm_match]
                    matched += 1

                    results.append({
                        "citing_arxiv_id": citing_id,
                        "cited_arxiv_id": row["cited_arxiv_id"],
                        "cited_paper_name": row["cited_paper_name"],
                        "matched_bib_title": matched_title,
                        "match_score": score
                    })
                    
                    # Track update for original CSV
                    updates.append({
                        'original_idx': original_idx,
                        'cited_arxiv_id': row["cited_arxiv_id"]
                    })
                else:
                    no_match += 1

        # Apply updates to original DataFrame and save back to CSV
        for update in updates:
            df_bib.at[update['original_idx'], 'cited_arxiv_id'] = update['cited_arxiv_id']
        print(results)
        self._save_data(df_bib)
