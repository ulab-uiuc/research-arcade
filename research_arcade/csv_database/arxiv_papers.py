"""
CSV version of the dataset
"""

import pandas as pd
import os
from typing import List, Optional, Tuple
from pathlib import Path
import json
from ..data import *


class ArxivCSVDataset:
    
    def __init__(self, csv_directory):
        self.csv_directory = csv_directory
        Path(csv_directory).mkdir(parents=True, exist_ok=True)

    def _get_path(self, table_name):
        return os.path.join(self.csv_directory, f"{table_name}.csv")
    
    def _load_table(self, table_name):
        path = self._get_path(table_name)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()
    
    def _save_table(self, table_name, df):
        path = self._get_path(table_name)
        df.to_csv(path, index=False)

    def create_papers_table(self):
        if not os.path.exists(self._get_path("papers")):
            df = pd.DataFrame(columns=[
                'id', 'arxiv_id', 'base_arxiv_id', 'version', 
                'title', 'abstract', 'submit_date', 'metadata'
            ])
            self._save_table("papers", df)
    
    def create_sections_table(self):
        if not os.path.exists(self._get_path("sections")):
            df = pd.DataFrame(columns=[
                'id', 'content', 'title', 'appendix', 'paper_arxiv_id'
            ])
            self._save_table("sections", df)
    
    def create_paragraphs_table(self):
        if not os.path.exists(self._get_path("paragraphs")):
            df = pd.DataFrame(columns=[
                'id', 'paragraph_id', 'content', 'paper_arxiv_id', 'paper_section'
            ])
            self._save_table("paragraphs", df)
    
    def create_authors_table(self):
        if not os.path.exists(self._get_path("authors")):
            df = pd.DataFrame(columns=[
                'id', 'semantic_scholar_id', 'name', 'homepage'
            ])
            self._save_table("authors", df)
    
    def create_categories_table(self):
        if not os.path.exists(self._get_path("categories")):
            df = pd.DataFrame(columns=['id', 'name', 'description'])
            self._save_table("categories", df)
    
    def create_institutions_table(self):
        if not os.path.exists(self._get_path("institutions")):
            df = pd.DataFrame(columns=['id', 'name', 'location'])
            self._save_table("institutions", df)
    
    def create_figures_table(self):
        if not os.path.exists(self._get_path("figures")):
            df = pd.DataFrame(columns=[
                'id', 'paper_arxiv_id', 'path', 'caption', 'label', 'name'
            ])
            self._save_table("figures", df)
    
    def create_tables_table(self):
        if not os.path.exists(self._get_path("tables")):
            df = pd.DataFrame(columns=[
                'id', 'paper_arxiv_id', 'path', 'caption', 'label', 'table_text'
            ])
            self._save_table("tables", df)
    
    def create_paper_authors_table(self):
        if not os.path.exists(self._get_path("paper_authors")):
            df = pd.DataFrame(columns=[
                'paper_arxiv_id', 'author_id', 'author_sequence'
            ])
            self._save_table("paper_authors", df)
    
    def create_paper_category_table(self):
        if not os.path.exists(self._get_path("paper_category")):
            df = pd.DataFrame(columns=['paper_arxiv_id', 'category_id'])
            self._save_table("paper_category", df)
    
    def create_citations_table(self):
        if not os.path.exists(self._get_path("citations")):
            df = pd.DataFrame(columns=[
                'id', 'citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 
                'bib_key', 'author_cited_paper', 'citing_sections', 'citing_paragraphs'
            ])
            self._save_table("citations", df)
    
    def create_paper_figures_table(self):
        if not os.path.exists(self._get_path("paper_figures")):
            df = pd.DataFrame(columns=['paper_arxiv_id', 'figure_id'])
            self._save_table("paper_figures", df)
    
    def create_paper_tables_table(self):
        if not os.path.exists(self._get_path("paper_tables")):
            df = pd.DataFrame(columns=['paper_arxiv_id', 'table_id'])
            self._save_table("paper_tables", df)
    
    def create_paragraph_citations_table(self):
        if not os.path.exists(self._get_path("paragraph_citations")):
            df = pd.DataFrame(columns=[
                'id', 'paragraph_id', 'paper_section', 'citing_arxiv_id', 'bib_key'
            ])
            self._save_table("paragraph_citations", df)
    
    def create_paragraph_references_table(self):
        if not os.path.exists(self._get_path("paragraph_references")):
            df = pd.DataFrame(columns=[
                'id', 'paragraph_id', 'paper_section', 'paper_arxiv_id', 
                'reference_label', 'reference_type'
            ])
            self._save_table("paragraph_references", df)
    
    def create_author_affiliation_table(self):
        if not os.path.exists(self._get_path("author_affiliation")):
            df = pd.DataFrame(columns=['author_id', 'institution_id'])
            self._save_table("author_affiliation", df)
    
    def create_citation_sch_table(self):
        if not os.path.exists(self._get_path("citation_sch")):
            df = pd.DataFrame(columns=[
                'id', 'arxiv_id', 'paper_id', 'title', 
                'year', 'abstract', 'external_ids'
            ])
            self._save_table("citation_sch", df)
    
    def create_all(self):
        """Create all CSV files."""
        self.create_papers_table()
        self.create_sections_table()
        self.create_paragraphs_table()
        self.create_authors_table()
        self.create_categories_table()
        self.create_institutions_table()
        self.create_figures_table()
        self.create_tables_table()
        self.create_paper_authors_table()
        self.create_paper_category_table()
        self.create_citations_table()
        self.create_paper_figures_table()
        self.create_paper_tables_table()
        self.create_author_affiliation_table()
        self.create_paragraph_references_table()
        self.create_paragraph_citations_table()
        self.create_citation_sch_table()

    def insert_paper(self, arxiv_id, base_arxiv_id, version, title, abstract=None, submit_date=None, metadata=None):
        df = self._load_table("papers")
        
        # Check for conflict
        if arxiv_id in df['arxiv_id'].values:
            return None

        new_id = df['id'].max() + 1 if not df.empty else 1
        meta_str = json.dumps(metadata) if metadata is not None else None
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'arxiv_id': arxiv_id,
            'base_arxiv_id': base_arxiv_id,
            'version': version,
            'title': title,
            'abstract': abstract,
            'submit_date': submit_date,
            'metadata': meta_str
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_table("papers", df)
        return new_id
    
    def insert_section(self, content, title, is_appendix, paper_arxiv_id):
        """Insert a section. Returns the generated section id."""
        df = self._load_table("sections")
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'content': content,
            'title': title,
            'appendix': is_appendix,
            'paper_arxiv_id': paper_arxiv_id
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_table("sections", df)
        return new_id
    
    def insert_paragraph(self, paragraph_id, content, paper_arxiv_id, paper_section):
        """Insert a paragraph. Returns the generated paragraph id."""
        df = self._load_table("paragraphs")
        
        # Check for conflict
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
            'paper_section': paper_section
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_table("paragraphs", df)
        return new_id
    
    def insert_author(self, semantic_scholar_id, name, homepage=None):
        """Insert an author. Returns the generated author id."""
        df = self._load_table("authors")
        
        # Check for conflict
        if semantic_scholar_id in df['semantic_scholar_id'].values:
            return None
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'semantic_scholar_id': semantic_scholar_id,
            'name': name,
            'homepage': homepage
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_table("authors", df)
        return new_id
    
    def insert_category(self, name, description=None):
        """Insert a category. Returns the category id."""
        df = self._load_table("categories")
        
        # Check for conflict
        existing = df[df['name'] == name]
        if not existing.empty:
            return existing.iloc[0]['id']
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'name': name,
            'description': description
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_table("categories", df)
        return new_id
    
    def insert_figure(self, paper_arxiv_id, path, caption=None, label=None, name=None):
        """Insert a figure. Returns the generated figure id."""
        df = self._load_table("figures")
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
        self._save_table("figures", df)
        return new_id
    
    def insert_citation(self, citing_arxiv_id, cited_arxiv_id, bib_title, 
                       bib_key, author_cited_paper, citing_sections):
        """Insert a citation. Returns True if inserted, False if exists or invalid."""
        if citing_arxiv_id == cited_arxiv_id:
            return False
        
        df = self._load_table("citations")
        
        # Check for conflict
        conflict = df[
            (df['citing_arxiv_id'] == citing_arxiv_id) & 
            (df['cited_arxiv_id'] == cited_arxiv_id)
        ]
        if not conflict.empty:
            return False
        
        new_id = df['id'].max() + 1 if not df.empty else 1
        sections_str = json.dumps(citing_sections) if citing_sections else '[]'
        
        new_row = pd.DataFrame([{
            'id': new_id,
            'citing_arxiv_id': citing_arxiv_id,
            'cited_arxiv_id': cited_arxiv_id,
            'bib_title': bib_title,
            'bib_key': bib_key,
            'author_cited_paper': author_cited_paper,
            'citing_sections': sections_str,
            'citing_paragraphs': '[]'
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_table("citations", df)
        return True
    
    def check_exist(self, paper_arxiv_id):
        """Check if a paper exists."""
        df = self._load_table("papers")
        return paper_arxiv_id in df['arxiv_id'].values
    
    def paper_authors_exist(self, paper_arxiv_id):
        """Check if paper has authors in the database."""
        df = self._load_table("paper_authors")
        return paper_arxiv_id in df['paper_arxiv_id'].values
    
    def insert_paper_obj(self, paper: Paper) -> Optional[int]:
        """Insert a paper from a Paper object."""
        return self.insert_paper(
            paper.arxiv_id, paper.base_arxiv_id, paper.version,
            paper.title, paper.abstract, paper.submit_date, paper.metadata
        )
    
    def insert_author_obj(self, author: Author) -> Optional[int]:
        """Insert an author from an Author object."""
        return self.insert_author(
            author.semantic_scholar_id, author.name, author.homepage
        )
    
    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """Retrieve a paper by arxiv_id and return a Paper object."""
        df = self._load_table("papers")
        result = df[df['arxiv_id'] == arxiv_id]
        
        if result.empty:
            return None
        
        row = result.iloc[0]
        metadata = json.loads(row['metadata']) if pd.notna(row['metadata']) else None
        
        return Paper(
            id=int(row['id']),
            arxiv_id=row['arxiv_id'],
            base_arxiv_id=row['base_arxiv_id'],
            version=row['version'],
            title=row['title'],
            abstract=row['abstract'] if pd.notna(row['abstract']) else None,
            submit_date=str(row['submit_date']) if pd.notna(row['submit_date']) else None,
            metadata=metadata
        )
    
    def get_sections_for_paper(self, paper_arxiv_id: str) -> List[Section]:
        """Retrieve all sections for a paper."""
        df = self._load_table("sections")
        results = df[df['paper_arxiv_id'] == paper_arxiv_id].sort_values('id')
        
        return [Section(
            id=int(row['id']),
            content=row['content'],
            title=row['title'],
            appendix=bool(row['appendix']),
            paper_arxiv_id=row['paper_arxiv_id']
        ) for _, row in results.iterrows()]
    
    def get_authors_for_paper(self, paper_arxiv_id: str) -> List[Tuple[Author, int]]:
        """Retrieve all authors for a paper with their sequence."""
        pa_df = self._load_table("paper_authors")
        a_df = self._load_table("authors")
        
        paper_authors = pa_df[pa_df['paper_arxiv_id'] == paper_arxiv_id]
        merged = paper_authors.merge(
            a_df, 
            left_on='author_id', 
            right_on='semantic_scholar_id'
        ).sort_values('author_sequence')
        
        return [(Author(
            id=int(row['id']),
            semantic_scholar_id=row['semantic_scholar_id'],
            name=row['name'],
            homepage=row['homepage'] if pd.notna(row['homepage']) else None
        ), int(row['author_sequence'])) for _, row in merged.iterrows()]
    
    def drop_all(self):
        """Delete all CSV files."""
        tables = [
            "papers", "sections", "paragraphs", "authors", "categories",
            "institutions", "figures", "tables", "paper_authors",
            "paper_category", "citations", "paper_figures", "paper_tables",
            "author_affiliation", "paragraph_citations",
            "paragraph_references", "citation_sch"
        ]
        for table in tables:
            path = self._get_path(table)
            if os.path.exists(path):
                os.remove(path)


