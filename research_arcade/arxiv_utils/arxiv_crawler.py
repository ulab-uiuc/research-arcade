sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob


class ArxivCrawler:
    def __init__(self, dest_dir):
        self.cj = CrawlerJob(dest_dir=dest_dir)
    
    def crawl_paper_by_time(self, start_date, end_date, dest_path=None):

        if dest_path: 
            dest_path = self.cj
        
        arxiv_ids = self.cj.crawl_recent_arxiv_paper_new(start_date=start_date, end_date=end_date, path=dest_path)

        return arxiv_ids
    
    # Wait a second... is this method sql based or csv based?
    def download_papers(self, arxiv_ids):
        self.cj.download_papers(arxiv_ids=arxiv_ids)

    def process_paper_graphs(self, arxiv_ids):
        self.cj.process_paper_graphs(arxiv_ids=arxiv_ids)
    
    def process_paper_paragraphs(self, arxiv_ids):
        self.cj.process_paper_paragraphs(arxiv_ids=arxiv_ids)

