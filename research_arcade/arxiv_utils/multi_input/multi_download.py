import requests
import datetime
import json
import os
import arxiv
from filelock import FileLock

from ..paper_collector.graph_construction import build_citation_graph_thread, build_citation_graph
from ..paper_collector.utils import None_constraint
from utils.error_handler import api_calling_error_exponential_backoff
from ..paper_collector.latex_parser import clean_latex_code
from semanticscholar import SemanticScholar

import os
from ..multi_input.multi_input import MultiInput
import arxiv
import json
import time
from typing import Optional, List, Dict, Any
import sys


TIMEOUT = 10
class MultiDownload:
    """
    This class supports downloading arxiv latex code, pdf and html webpage using arxiv id, link or bib
    """

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
    def download_arxiv(self, input: str, input_type: str, output_type: str, dest_dir: str = None):

        """
        Input type: 
            - id/arxiv_id -> arxiv id of the paper
            - bib/arxiv_bib -> bib of the paper which has the arxiv id field
            - url/link -> url of the link of paper on arxiv
        """

        mi = MultiInput()
        arxiv_id = mi.extract_arxiv_id(input, input_type)
        
        dest_dir = f"{dest_dir}/{arxiv_id}"
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(arxiv.Client().results(search))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch arXiv entry for {arxiv_id}: {e}")
        
        filename_pdf = arxiv_id + ".pdf"
        filename_latex = arxiv_id + ".tar.gz"

        # Save metadata
        if dest_dir:
            metadata = {
                'id': arxiv_id,
                'title': paper.title,
                'abstract': paper.summary,
                'authors': [a.name for a in paper.authors],
                'published': str(paper.published),
                'categories': paper.categories,
                'url': paper.entry_id,
            }
            os.makedirs(dest_dir, exist_ok=True)
            with open(f"{dest_dir}/{arxiv_id}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        if output_type == "pdf":
            pdf_path = paper.download_pdf(filename = filename_pdf, dirpath = dest_dir)

        if output_type == "latex":
            latex_path = paper.download_source(filename = filename_latex, dirpath = dest_dir)
            # Extract the real (with or without the version number) arxiv id
            print(latex_path)

        if output_type == "both":
            latex_path = paper.download_source(filename = filename_latex, dirpath = dest_dir)
            pdf_path = paper.download_pdf(filename = filename_pdf, dirpath = dest_dir)


    def download_papers_by_field_and_date(
        self,
        field: str,
        start_date: str,
        output_type: str = "both",
        max_results: Optional[int] = None,
        dest_dir: str = "./download_by_field",
        sort_order: str = "descending",
        page_size: int = 100,
        delay_seconds: float = 15.0,
    ):
        """
        Search arXiv for papers in a given subject category (e.g., "cs.AI") submitted from the start_date to today,
        then download PDFs and/or LaTeX source along with metadata.
        Note that due to the detection of arxiv API, the download might be incomplete, that the pdf or latex files might be missing.

        Parameters:
        - field: str, arXiv subject category, e.g., "cs.AI", "stat.ML", "math.AG", etc.
        - start_date: str, in "YYYY-MM-DD" format (inclusive).
        - output_type: str, one of "pdf", "latex", or "both".
        - max_results: Optional[int], maximum total number of papers to retrieve/download. If None, retrieves as many as possible up to API limits.
        - dest_dir: str, directory to save downloads and metadata.
        - sort_order: str, "ascending" or "descending" by submission date.
        - page_size: int, number of items per page/request to arXiv API (controls `page_size` in Client).
        - delay_seconds: float, seconds to wait between API calls (to be polite to arXiv servers).
        """
        # Parse dates
        try:
            dt_start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
            # dt_end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"start_date and end_date must be in YYYY-MM-DD format: {e}")

        # if dt_end < dt_start:
        #     raise ValueError("end_date must be the same or after start_date")

        # start_str = dt_start.strftime("%Y%m%d") + "0000"
        # end_str = dt_end.strftime("%Y%m%d") + "2359"

        query = f"cat:{field}"

        sort_by = arxiv.SortCriterion.SubmittedDate
        sort_order_enum = (
            arxiv.SortOrder.Ascending
            if sort_order.lower() == "ascending"
            else arxiv.SortOrder.Descending
        )
        print("ArXiv query:", query)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order_enum,
        )

        client = arxiv.Client(page_size=page_size, delay_seconds=delay_seconds)

        os.makedirs(dest_dir, exist_ok=True)

        downloaded_ids: List[str] = []
        count = 0
        for result in client.results(search):
            # Stop if we reached max_results
            if max_results is not None and count >= max_results:
                break
            
            # The time when the paper was updated
            paper_date = result.updated

            # Ensure that we are not downloading paper prior to the given date
            if paper_date.date() < dt_start:
                continue
            
            # Each result has attributes: entry_id (URL), pdf_url, title, summary, authors, published, updated, primary_category, categories, comment, journal_ref, doi, etc.
            arxiv_id = result.entry_id.split('/')[-1]
            paper_dir = os.path.join(dest_dir, arxiv_id)
            os.makedirs(paper_dir, exist_ok=True)
            # paper_date = result.

            # Save metadata to JSON
            metadata = {
                'id': arxiv_id,
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'published': result.published.isoformat() if hasattr(result.published, "isoformat") else str(result.published),
                'categories': result.categories,
                'url': result.entry_id,
            }
            metadata_path = os.path.join(paper_dir, f"{arxiv_id}_metadata.json")

            lock_path = metadata_path + ".lock"
            lock = FileLock(lock_path)
            with lock:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # ensure it's flushed to disk

            # Download PDF and/or LaTeX source
            if output_type in ("pdf", "both"):
                try:
                    # filename defaults to {id}.pdf
                    result.download_pdf(filename=f"{arxiv_id}.pdf", dirpath=paper_dir)
                except Exception as e:
                    print(f"[Warning] Failed to download PDF for {arxiv_id}: {e}")

            if output_type in ("latex", "both"):
                try:
                    # The arxiv library may download a tar.gz of source; filename {id}.tar.gz
                    result.download_source(filename=f"{arxiv_id}.tar.gz", dirpath=paper_dir)
                except Exception as e:
                    print(f"[Warning] Failed to download source for {arxiv_id}: {e}")
            downloaded_ids.append(arxiv_id)
            count += 1
            print(f"Downloaded {count}: {arxiv_id}")

        print(f"Finished: downloaded metadata for {count} papers in field '{field}' from {start_date}'")
        return downloaded_ids

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
    def download_semantic_scholar(self, input: str, input_type: str, dest_dir: str = None) -> None:

        mi = MultiInput()

        semantic_id = mi.extract_arxiv_id(input, input_type)

        sc = SemanticScholar(timeout=TIMEOUT)
        try:
            paper = sc.get_paper(paper_id=semantic_id)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch arXiv entry for {semantic_id}: {e}")

        # print(paper)

        print(type(paper))
        # paper.

        pass

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
    def build_paper_graph(self, input: str, input_type: str, dest_dir: str = None) -> None:
        """
        Extract the paper graph of the paper using knowledge_debugger, provided the paper id/url/bib, the type of input and the directory the output is stored
        - input: str
        - input_type: str
        - dest_dir: str
        """
        mi = MultiInput()

        arxiv_id = mi.extract_arxiv_id(input, input_type)

        arxiv_list = [arxiv_id]
        build_citation_graph_thread(
            seed=arxiv_list,
            source_path=f"{dest_dir}/{arxiv_id}",
            working_path=f"{dest_dir}/working_folder",
            output_path=f"{dest_dir}/output",
            debug_path=None,
            constraint=None_constraint,
            num_threads=len(arxiv_list),
            clear_source=False
        )
        
    @api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
    def build_paper_graphs(self, input: List[str], input_type: str, dest_dir: str = None) -> None:
        """
        Extract the paper graph of the paper using knowledge_debugger, provided the paper id/url/bib, the type of input and the directory the output is stored
        - input: str
        - input_type: str
        - dest_dir: str
        """
        mi = MultiInput()
        arxiv_list = []
        for id in input:
            arxiv_id = mi.extract_arxiv_id(id, input_type)
            arxiv_list.append(arxiv_id)
            source_path = f"{dest_dir}/{arxiv_id}"
            print(f"Source path: {source_path}")
            build_citation_graph_thread(
                seed=arxiv_list,
                source_path=source_path,
                working_path=f"{dest_dir}/working_folder",
                output_path=f"{dest_dir}/output",
                debug_path=None,
                constraint=None_constraint,
                num_threads=len(arxiv_list),
                clear_source=True,
                max_figure = sys.maxsize
            )

    def build_paragraphs(self, dest_dir: str):
        """
        - dest_dir: str
        """


    def get_abstract(self, input: str, input_type: str, dest_dir: str = None) -> str:

        mi = MultiInput()

        arxiv_id = mi.extract_arxiv_id(input, input_type)

        self.build_paper_graph(input, input_type, dest_dir)

        json_path = f"{dest_dir}/output/{arxiv_id}.json"

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'abstract' not in data:
            raise KeyError(f"'abstract' field not found in JSON at {json_path}")

        return clean_latex_code(data['abstract'])
        

    def get_title(self, input: str, input_type: str, dest_dir: str = None) -> str:

        mi = MultiInput()

        arxiv_id = mi.extract_arxiv_id(input, input_type)


        json_path = f"{dest_dir}/{arxiv_id}_metadata.json"
        # json_path = f"{dest_dir}/{arxiv_id}_metadeta.json"
        # print(json_path)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"The paper with arxiv id {arxiv_id} is not fould in the directory {dest_dir}: {e}")

        if 'title' not in data:
            raise KeyError(f"'abstract' field not found in JSON at {json_path}")

        return data['title']

    def get_references(self, input: str, input_type: str, max_retries: int = 8) -> List[Dict[str, Any]]:

        mi = MultiInput()

        arxiv_id = mi.extract_arxiv_id(input, input_type)

        SEMANTIC_SCHOLAR_API_URL = 'https://api.semanticscholar.org/graph/v1/paper/'
        url = f'{SEMANTIC_SCHOLAR_API_URL}ARXIV:{arxiv_id}/references'
        params = {'limit': 100, 'offset': 0, 'fields': 'title,abstract'}
        headers = {'User-Agent': 'PaperProcessor/1.0'}
        # print(f"max_retries: {max_retries}")
        for attempt in range(max_retries):
            # print(f"Attempt {attempt}")
            response = requests.get(url, params=params, headers=headers)  # type: ignore
            if response.status_code == 200:
                data = response.json()
                print(f"Retrieved data: {data}")
                references = []
                for ref in data.get('data', []):
                    cited_paper = ref.get('citedPaper', {})
                    if cited_paper:
                        ref_info = {
                            'title': cited_paper.get('title'),
                            'abstract': cited_paper.get('abstract'),
                            'paper_id': cited_paper.get('paperId')
                        }
                        references.append(ref_info)
                return references
            else:
                wait_time = 2**attempt
                print(
                    f'Error {response.status_code} fetching references for {arxiv_id}. Retrying in {wait_time}s...'
                )
                time.sleep(wait_time)  # Exponential backoff
        print(f'Failed to fetch references for {arxiv_id} after {max_retries} attempts.')
        return []

    def get_reference_arxiv(self, input: str, input_type: str) -> str:


        mi = MultiInput()
        arxiv_id = mi.extract_arxiv_id(input, input_type)
        
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(arxiv.Client().results(search))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch arXiv entry for {arxiv_id}: {e}")
        
        return paper.journal_ref

    
    # @api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
    # def download_metadata_arxiv(self, input: str, input_type: str, dest_dir: str = None) -> dict:
    #     """
    #     Retrieve metadata for an arXiv paper (given by ID, bib entry string, or URL/link), returning a dict
    #     and optionally saving it to dest_dir as JSON (and also a .bib file with a simple BibTeX entry).

    #     Returns:
    #         metadata (dict): keys include:
    #             - 'arxiv_id'
    #             - 'title'
    #             - 'authors' (list of names)
    #             - 'summary'
    #             - 'published' (ISO 8601 string)
    #             - 'updated' (ISO 8601 string) if available
    #             - 'comment'
    #             - 'journal_ref'
    #             - 'doi'
    #             - 'primary_category'
    #             - 'categories' (list)
    #             - 'pdf_url'
    #             - 'arxiv_url'
    #             - 'bibtex' (a simple generated BibTeX entry)
    #     """
    #     mi = MultiInput()
    #     arxiv_id = mi.extract_arxiv_id(input, input_type)


    #     try:
    #         search = arxiv.Search(id_list=[arxiv_id])
    #         paper = next(arxiv.Client().results(search))
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to fetch arXiv entry for {arxiv_id}: {e}")


    #     metadata = {}
    #     metadata['arxiv_id'] = arxiv_id
    #     metadata['title'] = getattr(paper, "title", None)
        
    #     authors = []
    #     try:
    #         for a in paper.authors:
    #             name = getattr(a, "name", a) if a is not None else None
    #             if name:
    #                 authors.append(name)
    #     except Exception:
    #         if isinstance(paper.authors, list):
    #             authors = paper.authors.copy()
    #     metadata['authors'] = authors

    #     metadata['summary'] = getattr(paper, "summary", None)
    #     pub = getattr(paper, "published", None)
    #     if isinstance(pub, (datetime.datetime,)):
    #         metadata['published'] = pub.isoformat()
    #     else:
    #         metadata['published'] = str(pub) if pub is not None else None
    #     upd = getattr(paper, "updated", None)
    #     if isinstance(upd, (datetime.datetime,)):
    #         metadata['updated'] = upd.isoformat()
    #     else:
    #         metadata['updated'] = str(upd) if upd is not None else None

    #     metadata['comment'] = getattr(paper, "comment", None)
    #     metadata['journal_ref'] = getattr(paper, "journal_ref", None)
    #     metadata['doi'] = getattr(paper, "doi", None)


    #     metadata['primary_category'] = getattr(paper, "primary_category", None)

    #     cats = getattr(paper, "categories", None)
    #     if isinstance(cats, (list, tuple)):
    #         metadata['categories'] = list(cats)
    #     else:
    #         if isinstance(cats, str):
    #             metadata['categories'] = cats.split()
    #         else:
    #             metadata['categories'] = None

    #     metadata['pdf_url'] = getattr(paper, "pdf_url", None)
    #     metadata['arxiv_url'] = getattr(paper, "entry_id", None)


    #     bib_id = arxiv_id.replace('/', '_')

    #     title_bib = metadata['title'] or ""

    #     title_bib = title_bib.replace("{", "\\{").replace("}", "\\}")
    #     authors_bib = " and ".join(authors) if authors else ""
    #     year = None
    #     try:
    #         if isinstance(pub, datetime.datetime):
    #             year = pub.year
    #         else:
    #             year = int(str(pub)[:4])
    #     except Exception:
    #         year = None
    #     bibtex_lines = [f"@article{{{bib_id},"]
    #     if title_bib:
    #         bibtex_lines.append(f"  title = {{{title_bib}}},")
    #     if authors_bib:
    #         bibtex_lines.append(f"  author = {{{authors_bib}}},")

    #     bibtex_lines.append(f"  journal = {{arXiv preprint arXiv:{arxiv_id}}},")
    #     if year:
    #         bibtex_lines.append(f"  year = {{{year}}},")

    #     if metadata.get('doi'):
    #         bibtex_lines.append(f"  doi = {{{metadata['doi']}}},")
    #     if metadata.get('journal_ref'):

    #         jr = metadata['journal_ref'].replace("{", "\\{").replace("}", "\\}")
    #         bibtex_lines.append(f"  note = {{{jr}}},")

    #     if len(bibtex_lines) > 1:
    #         last = bibtex_lines[-1]
    #         if last.endswith(','):
    #             bibtex_lines[-1] = last[:-1]
    #     bibtex_lines.append("}")
    #     metadata['bibtex'] = "\n".join(bibtex_lines)

    #     if dest_dir:
    #         os.makedirs(dest_dir, exist_ok=True)
    #         # JSON
    #         json_path = os.path.join(dest_dir, f"{arxiv_id.replace('/', '_')}_metadata.json")
    #         try:
    #             with open(json_path, "w", encoding="utf-8") as f:
    #                 json.dump(metadata, f, indent=2, ensure_ascii=False)
    #         except Exception as e:
    #             raise IOError(f"Could not write metadata JSON to {json_path}: {e}")

    #     return metadata


# id_string = "1806.08804"
# # dest_path = "./download"
# start_time = "2024-11-21"
# end_time = "2024-12-22"
# area = "cs.AI"
# mo = MultiDownload()


