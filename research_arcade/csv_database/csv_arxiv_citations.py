import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor


class CSVArxivCitation:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_citations.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_citations_table()

    def create_citations_table(self):
        df = pd.DataFrame(columns=[
            'id', 'citing_arxiv_id', 'cited_arxiv_id',
            'bib_title', 'bib_key', 'author_cited_paper',
            'citing_sections', 'citing_paragraphs'
        ])
        df.to_csv(self.csv_path, index=False)
        print(f"Created citations CSV at {self.csv_path}")

    def _load_data(self): 
        return pd.read_csv(self.csv_path) if os.path.exists(self.csv_path) else pd.DataFrame()
    def _save_data(self, df): 
        df.to_csv(self.csv_path, index=False)

    def insert_citation(self, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, author_cited_paper, citing_sections=None):
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
            'bib_key': bib_key, 'author_cited_paper': author_cited_paper,
            'citing_sections': citing_sections_str, 'citing_paragraphs': '[]'
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return True

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
                        
                        self.insert_citation(citing_arxiv_id=arxiv_id, cited_arxiv_id=cited_arxiv_id, citing_sections=list(citing_section), bib_title=bib_title, bib_key=bib_key, author_cited_paper=bib_author)

            except FileNotFoundError:
                print(f"Error: The file '{file_json}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{file_json}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue
