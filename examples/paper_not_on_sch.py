"""
It seems that when crawling paper on arxiv, sometimes the paper cannot be searched.
Need to figure out why.
"""

from semanticscholar import SemanticScholar
from dotenv import load_dotenv
import os

sch = None

load_dotenv()
api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
api_key = None
if not api_key:
    # We may still proceed, but it takes longer
    print("SEMANTIC_SCHOLAR_API_KEY not set in .env")
    sch = SemanticScholar()
else:
    print(f"Semantic Scholar API Key: {api_key}")
    sch = SemanticScholar(api_key=api_key)

arxiv_id_list = ["2506.21532", "2506.21521", "2506.21506", "2506.21502", "2506.21552", "1706.03762"]

for id in arxiv_id_list:
    try:
        print(f"Searching with id: {id}")
        paper_sch = sch.get_paper(f"ARXIV:{id}")
        authors = paper_sch.authors
        print(authors)
    except Exception as e:
        print(f"Paper with arxiv id {id} not found on semantic scholar: {e}")

