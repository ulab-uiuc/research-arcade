"""
Since the original crawler_job.py fails to extract the citations and figures (sometimes), here we follow the code in paper_processing_test.py.

We need to first download a range of files, then store the downloaded files in the database by their arxiv id.

After that, we take papers from the database and process them, following paper_processing_test.py.

Below is the database for task management, which faciliates the tracking of paper extraction process
"""

import psycopg2
from psycopg2.extras import Json
import json

PASSWORD = "Lcs20031121!"

class TaskDatabase:

    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost", dbname="postgres",
            user="postgres", password=PASSWORD, port="5432"
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    def create_paper_task_table(self):
        """
        Records the tasks of papers.
        processing elements:
        - downloaded (whether the paper has been downloaded from arxiv before)
        - paper_graph (including section, figure and citation extraction)
        - Semantic Scholar author retrieval (as some latest papers are not yet uploaded to semantic scholar)
        - paragraphs
        """
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_task(
            id SERIAL PRIMARY KEY,
            paper_arxiv_id VARCHAR(100) UNIQUE,
            downloaded BOOLEAN,
            paper_graph BOOLEAN,
            citation BOOLEAN,
            semantic_scholar BOOLEAN,
            paragraph BOOLEAN
        )
        """)
    
    # def create_paper_search_intervals_table(self):
    #     # 1) Create enum safely (works on old PG too)
    #     self.cur.execute("""
    #     DO $$
    #     BEGIN
    #     IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'search_status') THEN
    #         CREATE TYPE search_status AS ENUM ('pending','success','failed');
    #     END IF;
    #     END$$;
    #     """)

    #     # 2) (Optional) install btree_gist; ignore if missing privilege/already installed
    #     self.cur.execute("""
    #     DO $$
    #     BEGIN
    #     BEGIN
    #         CREATE EXTENSION btree_gist;
    #     EXCEPTION
    #         WHEN duplicate_object THEN NULL;
    #         WHEN insufficient_privilege THEN
    #         RAISE NOTICE 'btree_gist not installed (no privilege); continuing';
    #     END;
    #     END$$;
    #     """)

    #     # 3) Table + index + exclusion constraint (no trailing comma!)
    #     self.cur.execute("""
    #     CREATE TABLE IF NOT EXISTS paper_search_intervals(
    #     id          BIGSERIAL PRIMARY KEY,
    #     period      DATERANGE NOT NULL,           -- [start, end)
    #     status      search_status NOT NULL DEFAULT 'pending',
    #     CHECK (lower(period) < upper(period))
    #     )
    #     """)

    #     self.cur.execute("""
    #     CREATE INDEX IF NOT EXISTS paper_search_intervals_period_gist
    #     ON paper_search_intervals USING GIST (period)
    #     """)

    # def drop_paper_search_intervals_table(self):
    #     self.cur.execute("DROP TABLE IF EXISTS paper_search_intervals")

    # def insert_paper_search_intervals(self, search_key, start_date, end_date, status='pending'):
    #     """
    #     Returns the uncovered sub-intervals within [start_date, end_date) that do not
    #     intersect existing rows for this search_key. Inserts those intervals (status=...).
    #     """
    #     # 1) compute uncovered
    #     self.cur.execute("""
    #         SELECT * FROM get_uncovered_paper_search_periods(%s, %s::date, %s::date)
    #     """, (search_key, start_date, end_date))
    #     gaps = [row[0] for row in self.cur.fetchall()]  # each row is a DATERANGE

    #     if not gaps:
    #         return []  # fully covered

    #     # 2) insert each uncovered interval (race-safe with ON CONFLICT on the exclusion constraint)
    #     inserted = []
    #     for g in gaps:
    #         self.cur.execute("""
    #             INSERT INTO paper_search_intervals (search_key, period, status)
    #             VALUES (%s, %s, %s)
    #             ON CONFLICT ON CONSTRAINT paper_search_intervals_no_overlap DO NOTHING
    #             RETURNING id
    #         """, (search_key, g, status))
    #         row = self.cur.fetchone()
    #         if row:
    #             inserted.append({'id': row[0], 'period': g})

    #     return inserted



    def drop_paper_task_table(self):
        self.cur.execute("""
        DROP TABLE IF EXISTS paper_task
        """)


    def set_states(self, paper_arxiv_id, downloaded=None, paper_graph=None, semantic_scholar=None, citation=None, paragraph=None):

        """
        Change the task states of a paper.
        - paper_arxiv_id: str
        - downloaded: bool
        - paper_graph: bool
        - semantic_scholar: bool
        - paragraph: bool
        """
        
        fields = []
        params = []
        if downloaded is not None:
            fields.append("downloaded = %s")
            params.append(downloaded)
        if paper_graph is not None:
            fields.append("paper_graph = %s")
            params.append(paper_graph)
        if semantic_scholar is not None:
            fields.append("semantic_scholar = %s")
            params.append(semantic_scholar)
        if citation is not None:
            fields.append("citation = %s")
            params.append(citation)
        if paragraph is not None:
            fields.append("paragraph = %s")
            params.append(paragraph)
        if not fields:
            return
        params.append(paper_arxiv_id)

        sql = f"""
        UPDATE paper_task
            SET {','.join(fields)}
        where paper_arxiv_id = %s
        """

        self.cur.execute(sql, params)

    def initialize_state(self, paper_arxiv_id):
        """
        Add a paper into task db with all task states being false.
        """
        sql = """
        INSERT INTO paper_task
          (paper_arxiv_id, downloaded, paper_graph, citation, semantic_scholar, paragraph)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_arxiv_id) DO NOTHING
        RETURNING id
        """

        params = (paper_arxiv_id, False, False, False, False, False)
        self.cur.execute(sql, params)

        