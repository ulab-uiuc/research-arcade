import os
import re
import time
from difflib import SequenceMatcher

import arxiv
import requests
from beartype.typing import Any, Callable, Dict, List, Optional

from .error_handler import api_calling_error_exponential_backoff


def get_references(arxiv_id: str, max_retries: int = 5) -> List[Dict[str, Any]]:
    SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/"
    url = f"{SEMANTIC_SCHOLAR_API_URL}ARXIV:{arxiv_id}/references"
    params = {"limit": 100, "offset": 0, "fields": "title,abstract"}
    headers = {"User-Agent": "PaperProcessor/1.0"}

    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers)  # type: ignore
        if response.status_code == 200:
            data = response.json()
            references = []
            for ref in data.get("data", []):
                cited_paper = ref.get("citedPaper", {})
                if cited_paper:
                    ref_info = {
                        "title": cited_paper.get("title"),
                        "abstract": cited_paper.get("abstract"),
                    }
                    references.append(ref_info)
            return references
        else:
            wait_time = 2**attempt
            print(
                f"Error {response.status_code} fetching references for {arxiv_id}. Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)  # Exponential backoff
    print(f"Failed to fetch references for {arxiv_id} after {max_retries} attempts.")
    return []


def None_constraint(published: str) -> bool:
    return True


def year_constraint(start_year: int, end_year: int) -> Callable[[str], bool]:
    def constraint(published: str) -> bool:
        if published == "True":
            return True
        return start_year <= published.year <= end_year

    return constraint


def fetch_papers_cs(
    query: str, max_results_per_call: int = 100
) -> List[Dict[str, Any]]:
    # Get the results from ArXiv API
    search = arxiv.Search(
        query=query,
        max_results=max_results_per_call,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client()
    papers = []
    for paper in client.results(search):
        papers.append(
            {
                "paper": paper,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.published,
                "updated": paper.updated,
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "short_id": paper.get_short_id(),
                "entry_id": paper.entry_id,
            }
        )

    return papers


def download_latex_source(arxiv_id: str, save_dir: str = "latex_sources"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Downloading source")
    # Construct the URL for downloading the LaTeX source
    source_url = f"https://export.arxiv.org//e-print/{arxiv_id}"

    try:
        # Download the source as a tar file
        response = requests.get(source_url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f"{arxiv_id}.tar.gz")
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded source for {arxiv_id} to {file_path}")
        else:
            print(
                f"Failed to download source for {arxiv_id}, status code: {response.status_code}"
            )
    except Exception as e:
        print(f"An error occurred: {e}")


def similar(str1: str, str2: str) -> float:
    return SequenceMatcher(None, str1, str2).ratio()


@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def search_arxiv_id(arxiv_id: str) -> Optional[Any]:
    search = arxiv.Search(id_list=[arxiv_id])
    client = arxiv.Client()
    result = list(client.results(search))
    return result


@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def search_arxiv_query(query: str, max_results: int = 10) -> Optional[Any]:
    client = arxiv.Client()
    # Get the results from ArXiv API
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    result = list(client.results(search))
    return result


def query_and_match(title: str, author: str, max_results: int = 10):
    return None, 0
    title_ = re.sub(r"[:!{}()]", "", title)
    title_ = re.sub(r"[-_,\\]", " ", title_)
    author_ = re.sub(r"[:!{}(),]", "", author)
    author_ = re.sub(r"[-_,\\]", " ", author_)
    query = f"ti: {title_}"
    papers = search_arxiv_query(query, max_results)

    similarity = 0
    similar_paper = None
    for paper in papers:
        # print(dir(paper))
        # ['Author', 'Link', 'MissingFieldError', 'authors', 'categories', 'comment', 'doi', 'download_pdf', 'download_source', 'entry_id', 'get_short_id', 'journal_ref', 'links', 'pdf_url', 'primary_category', 'published', 'summary', 'title', 'updated']
        s = similar(paper.title.lower(), title.lower())
        if s > similarity:
            similarity = s
            similar_paper = paper
    if similarity < 0.8:
        query = f"au: {author_} AND ti: {title_}"
        papers = search_arxiv_query(query, max_results)
        for paper in papers:
            s = similar(paper.title.lower(), title.lower())
            if s > similarity:
                similarity = s
                similar_paper = paper

    return similar_paper, similarity
