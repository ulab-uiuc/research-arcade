from graph_constructor.node_processor import NodeConstructor
from multi_input.multi_download import MultiDownload
from paper_crawler.task_database import TaskDatabase

import datetime
import arxiv

nc = NodeConstructor()
md = MultiDownload()

# We first obtain a series of arxiv id of papers published recently

class CrawlerJob:

    def __init__(self):
        # Obtain the arxiv id of paper published after a certain date

        self.tdb = TaskDatabase()
        self.nc = NodeConstructor()
        self.md = MultiDownload()
        
        
        pass

    def crawl_arxiv_paper_id(self, year, month, day):
        """
        Crawl arxiv paper ids of papers published after the given date
        - year: int
        - month: int
        - dat: int
        """

        cutoff = datetime.datetime(2025, 6, 1)
        search = arxiv.Search(
            query    = "cat:cs.AI",
            sort_by  = arxiv.SortCriterion.SubmittedDate,
            sort_order = arxiv.SortOrder.Descending,
            max_results = 100
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
            self.tdb.initialize_state(arxiv_id)
    

    # Then we can proceed to other tasks

# First download the paper
# Why no citation?
arxiv_id = "2507.02822v1"
dir_path = "download"

# md.download_arxiv(input=arxiv_id, input_type="id", output_type="both", dest_dir=dir_path)
md.build_paper_graph(input=arxiv_id, input_type="id", dest_dir=dir_path)

# nc.drop_tables()

# At least I need to first download the paper
# nc.create_tables()
nc.process_paper(arxiv_id=arxiv_id, dir_path=dir_path)

