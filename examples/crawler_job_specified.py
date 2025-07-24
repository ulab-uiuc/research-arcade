
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob

cj = CrawlerJob(dest_dir="download")

# cj.drop_task_database()
# cj.create_task_database()
# cj.db.drop_all()
cj.db.create_all()

# arxiv_ids = cj.crawl_recent_arxiv_paper(year=2025, month=1, day=1, max_result=10)

arxiv_ids = {'2412.17767v2'}

# cj.initialize_paper_tasks(arxiv_ids=arxiv_ids)

print(f"arxiv ids: {arxiv_ids}")

cj.download_papers(arxiv_ids=arxiv_ids)

cj.process_paper_graphs(arxiv_ids=arxiv_ids)
# cj.process_paper_paragraphs(arxiv_ids=arxiv_ids)

# cj.process_paper_authors(arxiv_ids=arxiv_ids)
# cj.process_paper_citations(arxiv_ids=arxiv_ids)

