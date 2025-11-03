from ..graph_constructor.node_processor import NodeConstructor
from ..multi_input.multi_download import MultiDownload
from ..paper_crawler.task_database import TaskDatabase
from ..graph_constructor.database import Database
from ..paper_collector.paper_graph_processor import PaperGraphProcessor
import pytz
from arxiv import UnexpectedEmptyPageError
import datetime
import arxiv
from ..multi_input.arxiv_crawler_new import download_with_time, extract_arxiv_ids
from pathlib import Path


nc = NodeConstructor()
md = MultiDownload()

    # def __init__(
    #     self, data_dir: str, figures_dir: str, output_dir: str, threshold: float = 0.8
    # ):

# We first obtain a series of arxiv id of papers published recently

class CrawlerJob:

    def __init__(self, dest_dir):

        """
        - dest_dir: str
        """
        
        data_dir_path = f"{dest_dir}/output"
        figures_dir_path = f"{dest_dir}/output/images"
        output_dir_path = f"{dest_dir}/output/paragraphs"
        
        self.tdb = TaskDatabase()
        self.db = Database()
        self.nc = NodeConstructor()
        self.md = MultiDownload()
        self.pgp = PaperGraphProcessor(data_dir=data_dir_path, figures_dir=figures_dir_path, output_dir=output_dir_path)
        self.dest_dir = dest_dir
        
        print(self.dest_dir)
    
    def create_task_database(self):
        self.tdb.create_paper_task_table()
    
    def drop_task_database(self):
        self.tdb.drop_paper_task_table()

    def crawl_recent_arxiv_paper_new(self, start_date, end_date=None, path=None):
        save_path = download_with_time(start_date=start_date, end_date=end_date, save_path=path)
        arxiv_ids = extract_arxiv_ids(file_path=save_path)
        return arxiv_ids
    
    def crawl_recent_arxiv_paper(self, year, month, day, max_result):
        """
        Crawl arxiv paper ids of papers published after the given date.
        Stops when either:
          •  the paper's published date is older than cutoff, or
          •  arXiv returns an empty page (no more results).
        category: only crawl papers in certain categories listed below
        """

        cutoff = datetime.datetime(year, month, day, tzinfo=pytz.UTC)
        search = arxiv.Search(
            query       = "cat:cs.AI",
            sort_by     = arxiv.SortCriterion.SubmittedDate,
            sort_order  = arxiv.SortOrder.Descending,
            max_results = max_result
        )
        
        recent_ids = []
        iterator = search.results()

        try:
            for result in iterator:
                # stop early if we've gone past the cutoff
                if result.published < cutoff:
                    break
                recent_ids.append(result.get_short_id())
                # optional: break if we hit our max
                if len(recent_ids) >= max_result:
                    break

        except UnexpectedEmptyPageError:
            # arXiv had no entries on the next page—just stop
            pass

        return recent_ids
    
    def ids_with_major_category(self, arxiv_ids, category):
        client = arxiv.Client(page_size=1, num_retries=2)
        hits = []
        for arxiv_id in arxiv_ids:
            try:
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(client.results(search))
            except StopIteration:
                continue
            except Exception as e:
                print(f"Failed to fetch arXiv entry for {arxiv_id}: {e}")
                continue

            for cat in (paper.categories or []):
                major = cat.split('.', 1)[0]  # works even if there's no dot, e.g., "math-ph"
                if major == category:
                    hits.append(arxiv_id)
                    break
        return hits


    def initialize_paper_tasks(self, arxiv_ids, category = None):
        """
        Initialize a list of paper tasks given their arxiv ids
        - arxiv_ids: list[str]
        """

        # First select the paper of the category specified (if so)
        if category:
            id_category = self.ids_with_major_category(arxiv_ids=arxiv_ids, category=category)
        else:
            id_category = arxiv_ids
        
        cleaned_ids = []

        seen = set()

        for id in id_category or []:
            if not id:
                continue

            id = id.strip()

            if not id or id in seen:
                continue

            seen.add(id)
            cleaned_ids.append(id)
        added, skipped, failed = [], [], {}

        for id in cleaned_ids:
            try:
                # fetched = self.tdb.initialize_state(id)
                # if fetched:
                added.append(id)
                print(f"[+] Task for {id} initialized.")
                # else:
                #     skipped.append(id)
                #     print(f"[!]\tTask for {id} already exists — skipping.")
            
            except Exception as e:
                failed[id] = repr(e)
                print(f"[!]\tPaper {id} not added into database: {e}")

        return {"added": added, "skipped": skipped, "failed": failed}

        # added_arxiv_ids = []

        # for arxiv_id in arxiv_ids:
        #     try:
        #         self.tdb.initialize_state(arxiv_id)
        #         added_arxiv_ids.append(arxiv_id)
        #     except psycopg2.IntegrityError as e:
        #         # roll back so that the next iteration can run cleanly
        #         self.conn.rollback()

        #         if e.pgcode == errorcodes.UNIQUE_VIOLATION:
        #             print(f"[!]\tTask for {arxiv_id} already exists — skipping.")
        #         else:
        #             print(f"[!]\tPaper {arxiv_id} not added into database: {e}")

        #     except Exception as e:
        #         # catch any other exception
        #         self.conn.rollback()
        #         print(f"[!]\tUnexpected error initializing task for {arxiv_id}: {e}")

        #     else:
        #         print(f"[+] Task for {arxiv_id} initialized.")
        # return added_arxiv_ids
    
    def download_papers(self, arxiv_ids):
        """
        Download the paper with specified arxiv id from arxiv and save metadata to JSON
        If papers are not of the category specified (as is above)

        If the paper is not of the category specified, skip it.
        """
        downloaded_paper_ids = []
        for arxiv_id in arxiv_ids:
            try:              
                self.md.download_arxiv(input=arxiv_id, input_type = "id", output_type="latex", dest_dir=self.dest_dir)
                print(f"paper with id {arxiv_id} downloaded")
                self.tdb.set_states(downloaded=True, paper_arxiv_id=arxiv_id)
                downloaded_paper_ids.append(arxiv_id)
            except RuntimeError as e:
                self.tdb.conn.rollback()

                self.tdb.set_states(
                    paper_arxiv_id=arxiv_id,
                    downloaded=False
                )
                print(f"[ERROR] Failed to download {arxiv_id}: {e}")
                continue
    
    def add_existing_paper_graphs(self):
        """
        Given current paper graph json files, add them into the database.
        """

        # First find all the json files inside of the corresponding path
        # If the file name is not history.json, then we treat it as paper graph fils
        # Then extract the id before .json
        data_dir_path = f"{self.dest_dir}/output"

        arxiv_ids = []
        for p in Path(data_dir_path).iterdir():
            if p.is_file() and p.suffix == ".json":
                arxiv_id = p.stem            # e.g., "2508.00223v2"
                if arxiv_id != "history":
                    arxiv_ids.append(arxiv_id)

        processed_paper_ids = []

        for arxiv_id in arxiv_ids:
            try:
                # Store the processed data into database afterward
                is_processed = nc.process_paper(arxiv_id=arxiv_id, dir_path=f"{self.dest_dir}")

                if is_processed:
                    processed_paper_ids.append(arxiv_id)
                    self.tdb.set_states(paper_arxiv_id=arxiv_id, paper_graph=True)

            except Exception as e:
                print(f"[Warning] Failed to add paper with arxiv id {arxiv_id} to database: {e}")
                continue

        return processed_paper_ids

    def process_not_add(self, arxiv_ids):
        """
        Similarly, we can create but not add arxiv_ids
        """
        # processed_paper_ids = []

        try:
            self.md.build_paper_graphs(
                input=arxiv_ids,
                input_type="id",
                dest_dir=self.dest_dir
            )
        except Exception as e:
                print(f"[Warning] Failed to process papers: {e}")


    def process_paper_graphs(self, arxiv_ids):
        """
        Build paper graph using the knowledge debugger
        """

        processed_paper_ids = []

        # Instead of processing all papers at one, we process each paper separately
        # This avoids different papers using the same dir with some arxiv id (possible the id of the very first paper)
        
        for arxiv_id in arxiv_ids:
                    
            try:
                self.md.build_paper_graph(
                    input=arxiv_id,
                    input_type="id",
                    dest_dir=self.dest_dir
                )
            except Exception as e:
                    print(f"[Warning] Failed to process papers: {e}")

        # try:
        #     self.md.build_paper_graphs(
        #         input=arxiv_ids,
        #         input_type="id",
        #         dest_dir=self.dest_dir
        #     )
        # except Exception as e:
        #         print(f"[Warning] Failed to process papers: {e}")


        for arxiv_id in arxiv_ids:
            try:
                # Store the processed data into database afterward
                is_processed = nc.process_paper(arxiv_id=arxiv_id, dir_path=f"{self.dest_dir}")

                if is_processed:
                    processed_paper_ids.append(arxiv_id)
                    self.tdb.set_states(paper_arxiv_id=arxiv_id, paper_graph=True)
            
            except Exception as e:
                print(f"[Warning] Failed to add paper with arxiv id {arxiv_id} to database: {e}")
                continue

        return processed_paper_ids

        # Build the paragraphs json files later?
        # Use the node processor function in knowledge debugger


    def process_paper_paragraphs(self, arxiv_ids=None):
        """
        Convert the existing paper's paper graph json file into a collection of multiple jsons, including paragraph files.
        Then store the paragraph information into database
        """

        paper_paths = []
        # We first build paper node
        if not arxiv_ids:
            self.pgp.process_all_papers()
        else:
            # We loop through the provided arxiv ids of paper.
            for arxiv_id in arxiv_ids:
                paper_paths.append(f"{self.dest_dir}/output/{arxiv_id}.json")
            print(paper_paths)
            self.pgp.process_papers(paper_paths)

        self.nc.process_paragraphs(dir_path=self.dest_dir)

        for arxiv_id in arxiv_ids:

            self.tdb.set_states(paper_arxiv_id=arxiv_id, paragraph=True)


    def select_proceeded_task(self, task_type, max_results=100000):

        valid_types = {
            'downloaded',
            'paper_graph',
            'paragraph',
            'citation',
            'semantic_scholar',
        }
        if task_type not in valid_types:
            raise ValueError(f"Invalid task_type: {task_type!r}")

        
        sql = f"""
        SELECT paper_arxiv_id FROM paper_task WHERE {task_type} = TRUE ORDER BY id ASC LIMIT %s
        """

        self.tdb.cur.execute(sql, (max_results,))
        rows = self.tdb.cur.fetchall()
        return [row[0] for row in rows]
    
    def select_unproceeded_task(self, task_type, max_results=100000):

        valid_types = {
            'downloaded',
            'paper_graph',
            'paragraph',
            'citation',
            'semantic_scholar',
        }
        if task_type not in valid_types:
            raise ValueError(f"Invalid task_type: {task_type!r}")
        
        more_constraint = ""
        if task_type == "paragraph":
            more_constraint = "AND paper_graph = true "
        
        

        
        sql = f"""
        SELECT paper_arxiv_id FROM paper_task WHERE {task_type} = FALSE {more_constraint}ORDER BY id ASC LIMIT %s
        """
        
        self.tdb.cur.execute(sql, (max_results,))
        rows = self.tdb.cur.fetchall()
        return [row[0] for row in rows]



    def process_paper_citations(self, arxiv_ids):
        """
        For papers with arxiv ids, search in database and see if any paper does not have arxiv id/semantic scholar/...
        If so, search paper title online and see if the cited papers have been uploaded
        """

        # Go into database, fetch papers that does not have citations and search paper names on arxiv
        citation_added_paper_ids = []
        for arxiv_id in arxiv_ids:
            # Go to citation table in the database
            
            added = self.nc.citation_processor(arxiv_id=arxiv_id)

            if added:
                self.tdb.set_states(paper_arxiv_id=arxiv_id, citation=added)
                citation_added_paper_ids.append(arxiv_id)

        return citation_added_paper_ids

    def process_paper_authors(self, arxiv_ids):
        """
        For papers with arxiv ids, search papers on semantic scholar and fetch the author information. Store them into database
        """
        author_added_paper_ids = []

        for arxiv_id in arxiv_ids:
            
            added = self.nc.author_processor(arxiv_id)
            if added:
                self.tdb.set_states(paper_arxiv_id=arxiv_id, semantic_scholar=added)
                author_added_paper_ids.append(arxiv_id)
        
        return author_added_paper_ids

