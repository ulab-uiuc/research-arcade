import requests
import datetime
import json
import os
import arxiv

from paper_collector.graph_construction import build_citation_graph_thread
from paper_collector.utils import None_constraint
from utils.error_handler import api_calling_error_exponential_backoff
from paper_collector.latex_parser import clean_latex_code
from semanticscholar import SemanticScholar

import os
from multi_input.multi_input import MultiInput
import arxiv
import json
import time
from typing import Optional, List, Dict, Any
# # Search for the paper by its arXiv ID
# search = arxiv.Search(id_list=["2503.12600"])
# paper = next(arxiv.Client().results(search))
# # Download the PDF and source latex code to the current directory
# paper.download_pdf()
# paper.download_source()

TIMEOUT = 10
class MultiDownload:
    """
    This class supports downloading arxiv latex code, pdf and html webpage using arxiv id, link or bib
    """

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
    def download_arxiv(self, input: str, input_type: str, output_type: str, dest_dir: str = None):

        mi = MultiInput()
        arxiv_id = mi.extract_arxiv_id(input, input_type)

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
            paper.download_pdf(filename = filename_pdf, dirpath = dest_dir)

        if output_type == "latex":
            paper.download_source(filename = filename_latex, dirpath = dest_dir)
        
        if output_type == "both":
            paper.download_source(filename = filename_latex, dirpath = dest_dir)
            paper.download_pdf(filename = filename_pdf, dirpath = dest_dir)

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
        # Extract the paper graph of the provided paper using knowledge_debugger
        mi = MultiInput()

        arxiv_id = mi.extract_arxiv_id(input, input_type)

        arxiv_list = [arxiv_id]
        build_citation_graph_thread(
            arxiv_list,
            dest_dir,
            f"{dest_dir}/working_folder",
            f"{dest_dir}/output",
            None,
            None_constraint,
            len(arxiv_list),
            1000,
            True,
            len(arxiv_list),
        )
    
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


id_string = "1806.08804"
dest_path = "./download"

mo = MultiDownload()

# mo.download_arxiv(id_string, "id", "both", dest_path)
# mo.download_metadata_arxiv(id_string, "id",  dest_path)
# abstract = mo.get_title(id_string, "id",  dest_path)
cp = mo.get_references(id_string, "id", 8)
print("Cited Paper:")
print(cp)
print(len(cp))
# print(abstract)

# sc_id = "39ad6c911f3351a3b390130a6e4265355b4d593b"
# mo = multi_download()

# mo.download_semantic_scholar(sc_id, "id")

