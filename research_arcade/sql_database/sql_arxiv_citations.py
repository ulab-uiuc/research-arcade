import os
from typing import Optional, List, Tuple
import psycopg2
import psycopg2.extras
import pandas as pd
import json



import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..arxiv_utils.multi_input.multi_download import MultiDownload
from ..arxiv_utils.graph_constructor.node_processor import NodeConstructor
from ..arxiv_utils.utils import arxiv_id_processor


class SQLArxivCitation:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
<<<<<<< HEAD
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_citations_table()
=======
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
>>>>>>> website_design

    def _get_connection(self):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )
        conn.autocommit = self.autocommit
        return conn

    # -------------------------
    # DDL
    # -------------------------
    def create_citations_table(self):
        """
        Creates the arxiv_citations table.
        Enforces no self-citations and unique (citing_arxiv_id, cited_arxiv_id) pairs.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_citations (
                    id SERIAL PRIMARY KEY,
                    citing_arxiv_id VARCHAR(100) NOT NULL,
                    cited_arxiv_id VARCHAR(100) NOT NULL,
                    bib_title TEXT,
                    bib_key VARCHAR(255),
                    author_cited_paper TEXT,
                    citing_sections TEXT[],
                    citing_paragraphs TEXT[],
                    CONSTRAINT no_self_citation CHECK (citing_arxiv_id != cited_arxiv_id),
                    CONSTRAINT unique_citation UNIQUE (citing_arxiv_id, cited_arxiv_id)
                )
            """)
            cur.close()
        finally:
            conn.close()

    # -------------------------
    # CRUD
    # -------------------------
    def insert_citation(
        self, 
        citing_arxiv_id: str, 
        cited_arxiv_id: str, 
        bib_title: str = None,
        bib_key: str = None, 
        author_cited_paper: str = None,
        citing_sections: List[str] = None,
        citing_paragraphs: List[str] = None
    ) -> Optional[int]:
        """
        Insert a citation row; returns generated id or None on conflict.
        Automatically prevents self-citations and duplicate citations.
        """
        if citing_arxiv_id == cited_arxiv_id:
            return None
            
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_citations 
                (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
                 author_cited_paper, citing_sections, citing_paragraphs)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT ON CONSTRAINT unique_citation DO NOTHING
                RETURNING id
                """,
                (
                    citing_arxiv_id, 
                    cited_arxiv_id, 
                    bib_title, 
                    bib_key, 
                    author_cited_paper,
                    citing_sections or [],
                    citing_paragraphs or []
                )
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def delete_citation_by_id(self, citing_paper_id: str, cited_paper_id: str) -> bool:
        """
        Delete a citation by citing and cited paper ids.
        Returns True if deleted, False if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_citations 
                WHERE citing_arxiv_id = %s AND cited_arxiv_id = %s 
                RETURNING id
                """,
                (citing_paper_id, cited_paper_id)
            )
            ok = cur.fetchone() is not None
            cur.close()
            if ok:
                print(f"Deleted citation: {citing_paper_id} -> {cited_paper_id}")
            else:
                print(f"Citation not found: {citing_paper_id} -> {cited_paper_id}")
            return ok
        finally:
            conn.close()

    def update_citation(
        self, 
        id: int, 
        citing_arxiv_id: str = None,
        cited_arxiv_id: str = None,
        bib_title: str = None,
        bib_key: str = None,
        author_cited_paper: str = None,
        citing_sections: List[str] = None,
        citing_paragraphs: List[str] = None
    ) -> bool:
        """
        Partial update by id. Only non-None fields are updated.
        Returns True if a row was updated.
        """
        sets: List[str] = []
        vals: List = []

        if citing_arxiv_id is not None:
            sets.append("citing_arxiv_id = %s")
            vals.append(citing_arxiv_id)
        if cited_arxiv_id is not None:
            sets.append("cited_arxiv_id = %s")
            vals.append(cited_arxiv_id)
        if bib_title is not None:
            sets.append("bib_title = %s")
            vals.append(bib_title)
        if bib_key is not None:
            sets.append("bib_key = %s")
            vals.append(bib_key)
        if author_cited_paper is not None:
            sets.append("author_cited_paper = %s")
            vals.append(author_cited_paper)
        if citing_sections is not None:
            sets.append("citing_sections = %s")
            vals.append(citing_sections)
        if citing_paragraphs is not None:
            sets.append("citing_paragraphs = %s")
            vals.append(citing_paragraphs)

        if not sets:
            return False

        sql = f"UPDATE arxiv_citations SET {', '.join(sets)} WHERE id = %s RETURNING id"
        vals.append(id)

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(vals))
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    def get_citation_by_id(self, id: int, return_all: bool = False):
        """
        If return_all=False: returns a single tuple
           (id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
            author_cited_paper, citing_sections, citing_paragraphs)
        If return_all=True: returns a list of such tuples.
        Returns None if not found.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                       author_cited_paper, citing_sections, citing_paragraphs
                FROM arxiv_citations WHERE id = %s
                """,
                (id,)
            )
            rows = cur.fetchall() if return_all else cur.fetchone()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_all_citations(self, is_all_features=True):
        """Get all citations from the database."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                       author_cited_paper, citing_sections, citing_paragraphs
                FROM arxiv_citations
                """
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_citing_neighboring_cited(self, citing_paper_id: str):
        """
        Get all papers cited by the given paper.
        Returns citations where this paper is the citing paper.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                       author_cited_paper, citing_sections, citing_paragraphs
                FROM arxiv_citations WHERE citing_arxiv_id = %s
                """,
                (citing_paper_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_cited_neighboring_citing(self, cited_paper_id: str):
        """
        Get all papers that cite the given paper.
        Returns citations where this paper is the cited paper.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                       author_cited_paper, citing_sections, citing_paragraphs
                FROM arxiv_citations WHERE cited_arxiv_id = %s
                """,
                (cited_paper_id,)
            )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else None
        finally:
            conn.close()

    def get_citations_by_paper(self, arxiv_id: str, as_citing: bool = True):
        """
        Get all citations for a paper.
        If as_citing=True: returns citations where this paper cites others
        If as_citing=False: returns citations where others cite this paper
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            if as_citing:
                cur.execute(
                    """
                    SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                           author_cited_paper, citing_sections, citing_paragraphs
                    FROM arxiv_citations WHERE citing_arxiv_id = %s
                    """,
                    (arxiv_id,)
                )
            else:
                cur.execute(
                    """
                    SELECT id, citing_arxiv_id, cited_arxiv_id, bib_title, bib_key,
                           author_cited_paper, citing_sections, citing_paragraphs
                    FROM arxiv_citations WHERE cited_arxiv_id = %s
                    """,
                    (arxiv_id,)
                )
            rows = cur.fetchall()
            cur.close()
            return rows if rows else []
        finally:
            conn.close()

    def check_citation_exists(self, citing_arxiv_id: str, cited_arxiv_id: str) -> bool:
        """
        Returns True if a citation already exists between these two papers.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1 FROM arxiv_citations 
                WHERE citing_arxiv_id = %s AND cited_arxiv_id = %s 
                LIMIT 1
                """,
                (citing_arxiv_id, cited_arxiv_id)
            )
            ok = cur.fetchone() is not None
            cur.close()
            return ok
        finally:
            conn.close()

    # -------------------------
    # Bulk import from CSV
    # -------------------------
    def construct_citation_table_from_csv(self, csv_file: str) -> bool:
        """
        Imports rows from a CSV with columns:
          ['citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 'bib_key', 
           'author_cited_paper', 'citing_sections', 'citing_paragraphs']
        Ignores any 'id' column; DB assigns SERIAL ids.
        Skips conflicts and self-citations.
        
        Note: citing_sections and citing_paragraphs should be JSON arrays in CSV,
        which will be parsed and stored as PostgreSQL arrays.
        """
        if not os.path.exists(csv_file):
            print(f"Error: CSV file {csv_file} does not exist.")
            return False

        df = pd.read_csv(csv_file)
        required_cols = ['citing_arxiv_id', 'cited_arxiv_id', 'bib_title', 
                        'bib_key']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: CSV is missing required columns: {missing}")
            return False

        # Add optional columns if missing
        if 'author_cited_paper' not in df.columns:
            df['author_cited_paper'] = None
        if 'citing_sections' not in df.columns:
            df['citing_sections'] = '[]'
        if 'citing_paragraphs' not in df.columns:
            df['citing_paragraphs'] = '[]'

        # Parse JSON arrays from CSV
        rows = []
        for _, row in df.iterrows():
            # Skip self-citations
            if row['citing_arxiv_id'] == row['cited_arxiv_id']:
                continue
                
            citing_sections = json.loads(row['citing_sections']) if pd.notna(row['citing_sections']) else []
            citing_paragraphs = json.loads(row['citing_paragraphs']) if pd.notna(row['citing_paragraphs']) else []
            
            rows.append((
                row['citing_arxiv_id'],
                row['cited_arxiv_id'],
                row['bib_title'] if pd.notna(row['bib_title']) else None,
                row['bib_key'] if pd.notna(row['bib_key']) else None,
                row['author_cited_paper'] if pd.notna(row['author_cited_paper']) else None,
                citing_sections,
                citing_paragraphs
            ))

        if not rows:
            print("No valid rows to import.")
            return True

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO arxiv_citations 
                (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
                 author_cited_paper, citing_sections, citing_paragraphs)
                VALUES %s
                ON CONFLICT ON CONSTRAINT unique_citation DO NOTHING
                """,
                rows,
                page_size=1000
            )
            cur.close()
        finally:
            conn.close()

        print(f"Successfully imported {len(rows)} citations from {csv_file}")
        return True

    def construct_table_from_csv(self, csv_file: str) -> bool:
        """Alias for construct_citation_table_from_csv for consistency."""
        return self.construct_citation_table_from_csv(csv_file)

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
            
            # Convert to list of tuples for bulk insert
            rows = []
            for citation in citations_list:
                if 'citing_arxiv_id' not in citation or 'cited_arxiv_id' not in citation:
                    print(f"Warning: Skipping citation missing required fields: {citation}")
                    continue
                
                # Skip self-citations
                if citation['citing_arxiv_id'] == citation['cited_arxiv_id']:
                    continue
                
                citing_sections = citation.get('citing_sections', [])
                if isinstance(citing_sections, str):
                    citing_sections = json.loads(citing_sections)
                    
                citing_paragraphs = citation.get('citing_paragraphs', [])
                if isinstance(citing_paragraphs, str):
                    citing_paragraphs = json.loads(citing_paragraphs)
                
                rows.append((
                    citation['citing_arxiv_id'],
                    citation['cited_arxiv_id'],
                    citation.get('bib_title', None),
                    citation.get('bib_key', None),
                    citation.get('author_cited_paper', None),
                    citing_sections,
                    citing_paragraphs
                ))

            if not rows:
                print("No valid citation records to import")
                return False

            conn = self._get_connection()
            try:
                cur = conn.cursor()
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO arxiv_citations 
                    (citing_arxiv_id, cited_arxiv_id, bib_title, bib_key, 
                     author_cited_paper, citing_sections, citing_paragraphs)
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT unique_citation DO NOTHING
                    """,
                    rows,
                    page_size=1000
                )
                cur.close()
            finally:
                conn.close()

            print(f"Successfully imported {len(rows)} citations from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON file - {e}")
            return False
        except Exception as e:
            print(f"Error importing citations from JSON: {e}")
            return False

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
                md.download_arxiv(input=arxiv_id, input_type = "id", output_type="latex", dest_dir=dest_dir)
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
                    md = MultiDownload()
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
                        
                        self.insert_citation(citing_arxiv_id=arxiv_id, cited_arxiv_id=cited_arxiv_id, citing_sections=list(citing_sections), bib_title=bib_title, bib_key=bib_key, author_cited_paper=bib_author)

            except FileNotFoundError:
                print(f"Error: The file '{json_path}' was not found.")
                continue
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from '{json_path}'. Check if the file contains valid JSON.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                continue