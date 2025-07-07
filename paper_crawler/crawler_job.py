from graph_constructor.node_processor import NodeConstructor
from multi_input.multi_download import MultiDownload
from paper_crawler.task_database import TaskDatabase
from graph_constructor.database import Database
from paper_collector.paper_graph_processor import PaperGraphProcessor
import psycopg2
from psycopg2 import errorcodes, errors
import pytz
import json

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
        self.db = Database()
        self.nc = NodeConstructor()
        self.md = MultiDownload()
        self.pgp = PaperGraphProcessor()
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

        try:
            self.md.build_paper_graphs(
                input=arxiv_ids,
                input_type="id",
                dest_dir=self.dest_dir
            )
        except Exception as e:
                print(f"[Warning] Failed to process papers: {e}")


        for arxiv_id in arxiv_ids:
            try:
                # Store the processed data into database afterward
                is_processed = nc.process_paper(arxiv_id=arxiv_id, dir_path=self.dest_dir)

                if is_processed:
                    processed_paper_id.append(arxiv_id)

            except Exception as e:
                print(f"[Warning] Failed to add paper with arxiv id {arxiv_id} to database: {e}")
                continue
        
        return processed_paper_id

        # Build the paragraphs json files later?
        # Use the node processor function in knowledge debugger
    
    def process_paper_paragraphs(self, arxiv_ids=None):
        """
        Convert the existing paper's paper graph json file into a collection of multiple jsons, including paragraph files.
        Then store the paragraph information into database
        """
        # TODO
        
        paper_paths = []
        # We first build paper node
        if not arxiv_ids:
            self.pgp.process_all_papers()
        else:
            # We loop through the provided arxiv ids of paper.

            for arxiv_id in arxiv_ids:
                paper_paths.append(f"{self.dest_dir}/output/{arxiv_id}.json")
                
            self.pgp.process_papers(paper_paths)

        self.nc.process_paragraphs(dir_path=self.dest_dir)




    def process_paper_citations(self, arxiv_ids):
        """
        For papers with arxiv ids, search in database and see if any paper does not have arxiv id/semantic scholar/...
        If so, search paper title online and see if the cited papers have been uploaded
        """
        # TODO

        # Go into database, fetch papers that does not have citations and search paper names on arxiv and semantic scholars (?)
        
        for arxiv_id in arxiv_ids:
            # Go to citation table in the database
            sql = """
                SELECT * FROM citations where paper_arxiv_id = %s
            """

            result = self.db.cur.execute(sql, (arxiv_id,))



        

    def process_paper_authors(self, arxiv_ids):
        """
        For papers with arxiv ids, search papers on semantic scholar and fetch the author information. Store them into database
        """

        for arxiv_id in arxiv_ids:

            base_arxiv_id, version = self.arxiv_id_processor(arxiv_id)
            try:
                paper_sch = self.sch.get_paper(f"ARXIV:{base_arxiv_id}")
                authors = paper_sch.authors
            except Exception as e:
                print(f"Paper with arxiv id {base_arxiv_id} not found on semantic scholar: {e}")

        print(authors)

        #TODO then add authors into database


    def arxiv_id_processor(self, arxiv_id):
        """
        Given arxiv id, return base arxiv id and version
        - arxiv_id: str
        """
        return arxiv_id.split('v')



'''
After funishing the two TODO's. I run it!
'''



    
    
    