import pandas as pd
import os
from typing import Optional
from pathlib import Path
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
import json
from ..paper_collector.paper_graph_processor import PaperGraphProcessor

class CSVArxivParagraphs:
    def __init__(self, csv_dir: str):
        csv_path = f"{csv_dir}/arxiv_paragraphs.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paragraphs_table()
        # self.arxiv_crawler = ArxivCrawler()
    

    def create_paragraphs_table(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                'id', 'paragraph_id', 'content', 'paper_arxiv_id', 'paper_section'
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


    def insert_paragraph(self, paragraph_id, content, paper_arxiv_id, paper_section, section_id, paragraph_in_paper_id):
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
        downloaded_paper_ids = []
                
        data_dir_path = f"{dest_dir}/output"
        figures_dir_path = f"{dest_dir}/output/images"
        output_dir_path = f"{dest_dir}/output/paragraphs"
        pgp = PaperGraphProcessor(data_dir=data_dir_path, figures_dir=figures_dir_path, output_dir=output_dir_path)

        papers = []
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
            
            papers.append(arxiv_id)

        paper_paths = []
        # We first build paper node
        # We loop through the provided arxiv ids of paper.
        for arxiv_id in papers:
            paper_paths.append(f"{self.dest_dir}/output/{arxiv_id}.json")
        pgp.process_papers(paper_paths)

        # Build the paragraphs
        
        paragraph_path = f"{dest_dir}/output/paragraphs/text_nodes.jsonl"
        with open(paragraph_path) as f:
            data = [json.loads(line) for line in f]
        
        
        # Use arxiv_id + section name as key
        # Find the smallest paragraph_id generated by knowledge debugger
        # Subtract all paragraph id of the same section (of the same paper) with the smallest one to ensure that order starts with zero
        section_min_paragraph = {}

        for paragraph in data:
            paragraph_id = paragraph.get('id')
            # Extract paragraph_id
            id_number = self.get_paragraph_num(paragraph_id)
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
            id_number = self.get_paragraph_num(paragraph_id)
            id_zero_based = id_number - section_min_paragraph[(paper_arxiv_id, paper_section)]
            self.insert_paragraph(paragraph_id=id_zero_based, content=content, paper_arxiv_id=paper_arxiv_id, paper_section=paper_section)

            paragraph_cite_bib_keys = paragraph.get('cites')
            for bib_key in paragraph_cite_bib_keys:
                self.db.insert_paragraph_citations(paragraph_id=id_zero_based, paper_section=paper_section, citing_arxiv_id=paper_arxiv_id, bib_key=bib_key)


            # paragraph_ref_labels = paragraph.get('ref_labels')


            # # def insert_paragraph_reference(self, paragraph_id, paper_arxiv_id, reference_label, reference_type=None):

            # for ref_label in paragraph_ref_labels:

            #     ref_type = None
            #     # First search bib_key in databases.
            #     # If presented in one of them, we can determine the type of reference

            #     is_figure = self.db.check_exist_figure(bib_key=ref_label)
            #     is_table = self.db.check_exist_table(bib_key=ref_label)
            #     if is_figure:
            #         ref_type = 'figure'
            #     elif is_table:
            #         ref_type = 'table'
                
            #     self.insert_paragraph(paragraph_id=id_zero_based, paper_section=paper_section, paper_arxiv_id=paper_arxiv_id, paper_section=paper_section, refe)

            #     self.db.insert_paragraph_reference(paragraph_id=id_zero_based, paper_section=paper_section, paper_arxiv_id=paper_arxiv_id, reference_label=ref_label, reference_type=ref_type)
