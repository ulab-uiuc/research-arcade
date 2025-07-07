from graph_constructor.node_processor import NodeConstructor
from multi_input.multi_download import MultiDownload
from paper_crawler.task_database import TaskDatabase
import psycopg2
from psycopg2 import errorcodes, errors
import pytz

import datetime
import arxiv

nc = NodeConstructor()
md = MultiDownload()

# We first obtain a series of arxiv id of papers published recently

class CrawlerJob:

    def __init__(self, dest_dir):

        """
        - dest_dir: str
        """

        self.tdb = TaskDatabase()
        self.nc = NodeConstructor()
        self.md = MultiDownload()
        self.dest_dir = dest_dir
    
    def create_database(self):
        self.tdb.create_paper_task_table()

    def crawl_recent_arxiv_paper(self, year, month, day, max_result=100):
        """
        Crawl arxiv paper ids of papers published after the given date
        - year: int
        - month: int
        - dat: int
        """

        cutoff = datetime.datetime(year, month, day, tzinfo=pytz.UTC)
        search = arxiv.Search(
            query    = "cat:cs.AI",
            sort_by  = arxiv.SortCriterion.SubmittedDate,
            sort_order = arxiv.SortOrder.Descending,
            max_results = max_result
        )
        recent_ids = []
        for result in search.results():
            if result.published < cutoff:
                break
            recent_ids.append(result.get_short_id())

        return recent_ids

    def initialize_paper_tasks(self, arxiv_ids):
        """
        Initialize a list of paper tasks given their arxiv ids
        - arxiv_ids: list[str]
        """

        for arxiv_id in arxiv_ids:
            # Maybe there is repetition?
            try:
                self.tdb.initialize_state(arxiv_id)
            except psycopg2.IntegrityError as e:
                self.conn.rollback()
                if e.pgcode == errorcodes.UNIQUE_VIOLATION:
                    print(f"[!]\tTask for {arxiv_id} already exists — skipping.")
                else:
                    raise
            else:
                print(f"[+] Task for {arxiv_id} initialized.")
    
    def download_papers(self, arxiv_ids):
        """
        Download the paper with specified arxiv id from arxiv and save metadata to JSON
        """
        for arxiv_id in arxiv_ids:
            try:
                self.md.download_arxiv(input=arxiv_id, input_type = "id", output_type="both", dest_dir=self.dest_dir)
                self.tdb.set_states(downloaded=True, paper_arxiv_id=arxiv_id)
            except RuntimeError as e:
                self.tdb.conn.rollback()

                self.tdb.set_states(
                    paper_arxiv_id=arxiv_id,
                    downloaded=False
                )
                print(f"[ERROR] Failed to download {arxiv_id}: {e}")
                continue


    def process_paper_graphs(self, arxiv_ids):
        """
        Build paper graph using the knowledge debugger
        """

        processed_paper_id = []

        for arxiv_id in arxiv_ids:
            try:
                self.md.build_paper_graph(
                    input=arxiv_ids,
                    input_type="id",
                    dest_dir=self.dest_dir
                )
                # Store the processed data into database afterward
                # nc.process_paper(arxiv_id=arxiv_id, dir_path=self.dest_dir)

                # Build the paragraphs json files
                # Use the node processor function in knowledge debugge
                
            except Exception as e:
                print(f"[Warning] Failed to process {arxiv_id}: {e}")
                continue
