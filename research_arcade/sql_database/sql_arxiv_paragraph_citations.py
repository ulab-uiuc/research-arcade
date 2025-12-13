import os
from typing import Optional, List
import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values
import pandas as pd
import json
from rapidfuzz import fuzz, process
from collections import defaultdict
from semanticscholar import SemanticScholar
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class SQLArxivParagraphCitation:
    def __init__(self, host: str, dbname: str, user: str, password: str, port: str):
        self.host = host
        self.dbname = dbname
        self.user = user
        self.password = password
        self.port = port
        self.autocommit = True
        self.create_paragraph_references_table()

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
    def create_paragraph_references_table(self):
        """
        Creates the arxiv_paragraph_citations table.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS arxiv_paragraph_citations (
                    id SERIAL PRIMARY KEY,
                    paragraph_id INTEGER,
                    paper_section TEXT,
                    citing_arxiv_id VARCHAR(100) NOT NULL,
                    bib_key VARCHAR(255),
                    cited_arxiv_id VARCHAR(100),
                    paragraph_global_id INTEGER,
                    bib_title TEXT
                )
            """)
            cur.close()
            print("Created arxiv_paragraph_citations table")
        finally:
            conn.close()

    # -------------------------
    # CRUD Operations
    # -------------------------
    def insert_paragraph_reference(
        self,
        paragraph_id: int,
        paper_section: str,
        citing_arxiv_id: str,
        bib_key: str,
        cited_arxiv_id: Optional[str] = None,
        paragraph_global_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Insert a paragraph reference row; returns generated id.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO arxiv_paragraph_citations 
                (paragraph_id, paper_section, citing_arxiv_id, bib_key, 
                 cited_arxiv_id, paragraph_global_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    paragraph_id,
                    paper_section,
                    citing_arxiv_id,
                    bib_key,
                    cited_arxiv_id,
                    paragraph_global_id
                )
            )
            res = cur.fetchone()
            cur.close()
            return res[0] if res else None
        finally:
            conn.close()

    def get_all_paragraph_references(self) -> Optional[pd.DataFrame]:
        """
        Get all paragraph references from the database.
        Returns DataFrame or None if empty.
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT id, paragraph_id, paper_section, citing_arxiv_id, 
                       bib_key, cited_arxiv_id, paragraph_global_id, bib_title
                FROM arxiv_paragraph_citations
            """
            df = pd.read_sql(query, conn)
            return None if df.empty else df
        finally:
            conn.close()

    def get_reference_neighboring_paragraphs(self, reference_id: int) -> Optional[pd.DataFrame]:
        """
        Get a specific paragraph reference by ID.
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT id, paragraph_id, paper_section, citing_arxiv_id, 
                       bib_key, cited_arxiv_id, paragraph_global_id, bib_title
                FROM arxiv_paragraph_citations
                WHERE id = %s
            """
            df = pd.read_sql(query, conn, params=(reference_id,))
            return None if df.empty else df.reset_index(drop=True)
        finally:
            conn.close()

    def delete_paragraph_citation_by_paragraph_id(self, paragraph_id: int) -> int:
        """
        Delete paragraph citations by paragraph_id.
        Returns count of deleted rows.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM arxiv_paragraph_citations 
                WHERE paragraph_id = %s
                """,
                (paragraph_id,)
            )
            count = cur.rowcount
            cur.close()
            return count
        finally:
            conn.close()

    # -------------------------
    # Bulk Operations
    # -------------------------
    def construct_table_from_api(self, arxiv_ids: List[str], dest_dir: str):
        """
        Construct the paragraph citations table from API-generated paragraph data.
        Reads from JSONL file and populates the database.
        """
        from ..arxiv_utils.utils import get_paragraph_num
        
        paragraph_path = f"{dest_dir}/output/paragraphs/text_nodes.jsonl"

        if not os.path.exists(paragraph_path):
            print(f"Error: Paragraph file {paragraph_path} does not exist.")
            return

        with open(paragraph_path) as f:
            data = [json.loads(line) for line in f]

        section_min_paragraph = {}

        # First pass: find minimum paragraph ID per section
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

        # Second pass: insert citations
        rows = []
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

            cite_keys = paragraph.get("cites") or []
            for bib_key in cite_keys:
                rows.append((
                    id_zero_based,
                    paper_section,
                    paper_arxiv_id,
                    bib_key,
                    None,  # cited_arxiv_id
                    None   # paragraph_global_id
                ))

        if not rows:
            print("No citation data to insert.")
            return

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            execute_values(
                cur,
                """
                INSERT INTO arxiv_paragraph_citations 
                (paragraph_id, paper_section, citing_arxiv_id, bib_key, 
                 cited_arxiv_id, paragraph_global_id)
                VALUES %s
                """,
                rows,
                page_size=1000
            )
            cur.close()
            print(f"Inserted {len(rows)} paragraph citation records")
        finally:
            conn.close()

    # -------------------------
    # Matching Operations
    # -------------------------
    def arxiv_match_bib_key_to_bib_title(self, arxiv_citations):
        """
        Match bib_key to bib_title using arxiv_citations object.
        Updates bib_title column for rows where it's NULL or empty.
        
        Args:
            arxiv_citations: Object with method get_bib_title_by_citing_arxiv_id_bib_key
        
        Returns:
            Number of rows updated
        """
        conn = self._get_connection()
        try:
            # Ensure bib_title column exists
            cur = conn.cursor()
            cur.execute("""
                ALTER TABLE arxiv_paragraph_citations
                ADD COLUMN IF NOT EXISTS bib_title TEXT
            """)
            cur.close()

            # Get rows needing bib_title
            query = """
                SELECT id, citing_arxiv_id, bib_key
                FROM arxiv_paragraph_citations
                WHERE bib_title IS NULL OR bib_title = ''
            """
            df = pd.read_sql(query, conn)

            if df.empty:
                return 0
            
            updates = []
            for _, row in df.iterrows():
                bib_title = arxiv_citations.get_bib_title_by_citing_arxiv_id_bib_key(
                    row["citing_arxiv_id"],
                    row["bib_key"]
                )

                if bib_title:
                    updates.append((bib_title, row["id"]))

            if not updates:
                return 0

            # Batch update
            cur = conn.cursor()
            execute_values(
                cur,
                """
                UPDATE arxiv_paragraph_citations
                SET bib_title = data.bib_title
                FROM (VALUES %s) AS data(bib_title, id)
                WHERE arxiv_paragraph_citations.id = data.id
                """,
                updates,
                template="(%s, %s)"
            )
            cur.close()

            print(f"Updated {len(updates)} bib_title entries")
            return len(updates)
        finally:
            conn.close()

    def match_paragraph_citations_to_paragraph_global_id(self, arxiv_paragraphs):
        """
        Match paragraph citations to global paragraph IDs.
        
        Args:
            arxiv_paragraphs: Object with method get_id_by_arxiv_id_section_paragraph_id
        
        Returns:
            Number of rows updated
        """
        conn = self._get_connection()
        try:
            # Get rows needing paragraph_global_id
            query = """
                SELECT id, citing_arxiv_id, paper_section, paragraph_id
                FROM arxiv_paragraph_citations
                WHERE paragraph_global_id IS NULL
            """
            df = pd.read_sql(query, conn)

            if df.empty:
                return 0
            
            updates = []
            for _, row in df.iterrows():
                paragraph_global_id = arxiv_paragraphs.get_id_by_arxiv_id_section_paragraph_id(
                    row['citing_arxiv_id'],
                    row['paper_section'],
                    row['paragraph_id']
                )

                if paragraph_global_id is not None:
                    updates.append((paragraph_global_id, row['id']))

            if not updates:
                return 0

            # Batch update
            cur = conn.cursor()
            execute_values(
                cur,
                """
                UPDATE arxiv_paragraph_citations
                SET paragraph_global_id = data.paragraph_global_id
                FROM (VALUES %s) AS data(paragraph_global_id, id)
                WHERE arxiv_paragraph_citations.id = data.id
                """,
                updates,
                template="(%s, %s)"
            )
            cur.close()

            print(f"Updated {len(updates)} paragraph_global_id entries")
            return len(updates)
        finally:
            conn.close()

    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        if not title:
            return ""
        return " ".join(title.lower().strip().split())

    def bib_title_matching(self, similarity_threshold: int = 95) -> int:
        """
        Match bibliography titles to arxiv IDs using Semantic Scholar API
        and fuzzy matching, then update the database.
        
        Args:
            similarity_threshold: Minimum similarity score (0-100) for matching
        
        Returns:
            Number of matches found
        """
        conn = self._get_connection()
        
        try:
            # Ensure cited_arxiv_id column exists
            cur = conn.cursor()
            cur.execute("""
                ALTER TABLE arxiv_paragraph_citations
                ADD COLUMN IF NOT EXISTS cited_arxiv_id VARCHAR(100)
            """)
            cur.close()

            # Load data from database
            query = """
                SELECT id, citing_arxiv_id, bib_title, cited_arxiv_id
                FROM arxiv_paragraph_citations
                WHERE bib_title IS NOT NULL
            """
            df = pd.read_sql(query, conn)
            
            if df.empty:
                print("No data to process.")
                return 0
            
            # Get unique citing arxiv IDs
            citing_arxiv_ids = df["citing_arxiv_id"].dropna().unique().tolist()
            
            print(f"Number of citing arxiv ids to process: {len(citing_arxiv_ids)}")
            
            # Initialize Semantic Scholar client
            sch = SemanticScholar(api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
            
            # Build reference map from Semantic Scholar
            ref_map = defaultdict(list)
            
            for arxiv_id in citing_arxiv_ids:
                try:
                    paper = sch.get_paper(
                        f"arXiv:{arxiv_id}",
                        fields=[
                            "title",
                            "references",
                            "references.title",
                            "references.externalIds"
                        ]
                    )
                    
                    if not paper or not paper.references:
                        continue
                    
                    for ref in paper.references:
                        if ref.externalIds and "ArXiv" in ref.externalIds:
                            cited_id = ref.externalIds["ArXiv"]
                            cited_title = ref.title or ""
                            
                            if cited_id and cited_title:
                                ref_map[str(arxiv_id)].append(
                                    (str(cited_id), cited_title)
                                )
                
                except Exception as e:
                    print(f"[Warning] Failed Semantic Scholar query for {arxiv_id}: {e}")
                    continue
            
            # Perform fuzzy matching
            total = 0
            matched = 0
            no_match = 0
            updates = []
            
            for idx, row in df.iterrows():
                bib_title = row.get("bib_title")
                citing_id = str(row.get("citing_arxiv_id"))
                row_id = row.get("id")
                
                if not bib_title or citing_id not in ref_map:
                    no_match += 1
                    continue
                
                candidates = ref_map[citing_id]
                candidate_titles = [self._normalize_title(t[1]) for t in candidates]
                
                norm_query = self._normalize_title(bib_title)
                
                best = process.extractOne(
                    norm_query,
                    candidate_titles,
                    scorer=fuzz.ratio,
                    score_cutoff=similarity_threshold
                )
                
                total += 1
                
                if best:
                    _, score, best_idx = best
                    cited_arxiv_id = candidates[best_idx][0]
                    updates.append((cited_arxiv_id, row_id))
                    matched += 1
                else:
                    no_match += 1
            
            # Batch update database
            if updates:
                cur = conn.cursor()
                execute_values(
                    cur,
                    """
                    UPDATE arxiv_paragraph_citations
                    SET cited_arxiv_id = data.cited_arxiv_id
                    FROM (VALUES %s) AS data(cited_arxiv_id, id)
                    WHERE arxiv_paragraph_citations.id = data.id
                    """,
                    updates,
                    template="(%s, %s)"
                )
                cur.close()
            
            print("\n" + "=" * 50)
            print("Semantic Scholar Citation Matching Stats")
            print(f"Total processed: {total}")
            print(f"Matched: {matched}")
            print(f"No match: {no_match}")
            print("=" * 50)
            
            return matched
        
        finally:
            conn.close()


    def save_paragraph_with_reference(
        self, 
        arxiv_paragraphs,
        output_csv_path: str
    ) -> int:
        """
        Join citations with paragraph content and save to CSV.
        Uses the arxiv_paragraphs object to fetch paragraph content.
        
        Args:
            arxiv_paragraphs: SQLArxivParagraphs object or similar with get_paragraph_by_id method
            output_csv_path: Path to save output CSV
        
        Returns:
            Number of rows saved
        """
        conn = self._get_connection()
        
        try:
            query = """
                SELECT id, citing_arxiv_id, paragraph_global_id, cited_arxiv_id,
                    bib_key, bib_title, paper_section, paragraph_id
                FROM arxiv_paragraph_citations
                WHERE cited_arxiv_id IS NOT NULL 
                AND paragraph_global_id IS NOT NULL
            """
            
            
            df = pd.read_sql(query, conn)
            
            if df.empty:
                print("No data to process.")
                return 0

            output_rows = []

            for _, row in df.iterrows():
                cited_arxiv_id = row['cited_arxiv_id']
                paragraph_global_id = row['paragraph_global_id']
                
                if pd.isna(cited_arxiv_id) or pd.isna(paragraph_global_id):
                    continue
                
                para_obj = arxiv_paragraphs.get_paragraph_by_id(paragraph_global_id)

                if para_obj is None or para_obj.empty:
                    continue

                paragraph_text = para_obj["content"].iloc[0]

                if not paragraph_text or not isinstance(paragraph_text, str):
                    continue
                
                output_rows.append({
                    "citing_arxiv_id": row['citing_arxiv_id'],
                    "paragraph_global_id": paragraph_global_id,
                    "paragraph_text": paragraph_text,
                    "cited_arxiv_id": cited_arxiv_id,
                    "bib_key": row['bib_key'],
                    "bib_title": row['bib_title'],
                    "paper_section": row['paper_section'],
                    "paragraph_id": row['paragraph_id']
                })

            if not output_rows:
                print("No valid rows to save.")
                return 0

            out_df = pd.DataFrame(output_rows)
            out_df.to_csv(output_csv_path, index=False)

            print(f"Saved {len(out_df)} paragraph-reference rows to {output_csv_path}")
            return len(out_df)
        
        finally:
            conn.close()