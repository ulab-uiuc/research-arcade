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
    
    def drop_paper_task_table(self):
        ("""
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