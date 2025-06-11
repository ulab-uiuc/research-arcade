import datetime
import json
import os
import arxiv

from paper_collector.graph_construction import build_citation_graph_thread
from paper_collector.utils import None_constraint
from utils.error_handler import api_calling_error_exponential_backoff
from paper_collector.latex_parser import clean_latex_code
from semanticscholar import SemanticScholar


from multi_input.input_conversion import multi_input

# # Search for the paper by its arXiv ID
# search = arxiv.Search(id_list=["2503.12600"])
# paper = next(arxiv.Client().results(search))

# # Download the PDF and source latex code to the current directory
# paper.download_pdf()
# paper.download_source()

TIMEOUT = 10
class multi_download:
    """
    This class supports downloading arxiv latex code, pdf and html webpage using arxiv id, link or bib
    """

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
    def download_arxiv(self, input: str, input_type: str, output_type: str, dest_dir: str = None):

        mi = multi_input()
        input_type = input_type.lower()
        arxiv_id = ""
        if input_type == "id" or input_type == "arxiv_id":
            arxiv_id = input

        elif input_type == "bib" or input_type == "arxiv_bib":
            bib_dict = mi.extract_bib_from_string(input)
            arxiv_id = mi.extract_arxiv_id(bib_dict)

        elif input_type == "url" or input_type == "link":
            arxiv_id = mi.arxiv_url_to_id(input)

        else:
            # Raise error for unknown input_type
            raise ValueError(f"Unknown input_type '{input_type}'. "
                            f"Expected one of: 'id', 'arxiv_id', 'bib', 'arxiv_bib', 'url', 'link'.")

        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(arxiv.Client().results(search))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch arXiv entry for {arxiv_id}: {e}")
        
        filename_pdf = arxiv_id + ".pdf"
        filename_latex = arxiv_id + ".tar.gz"


        if output_type == "pdf":
            paper.download_pdf(filename = filename_pdf, dirpath = dest_dir)

        if output_type == "latex":
            paper.download_source(filename = filename_latex, dirpath = dest_dir)
        
        if output_type == "both":
            paper.download_source(filename = filename_latex, dirpath = dest_dir)
            paper.download_pdf(filename = filename_pdf, dirpath = dest_dir)

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
    def download_metadata_arxiv(self, input: str, input_type: str, dest_dir: str = None) -> dict:
        """
        Retrieve metadata for an arXiv paper (given by ID, bib entry string, or URL/link), returning a dict
        and optionally saving it to dest_dir as JSON (and also a .bib file with a simple BibTeX entry).

        Returns:
            metadata (dict): keys include:
                - 'arxiv_id'
                - 'title'
                - 'authors' (list of names)
                - 'summary'
                - 'published' (ISO 8601 string)
                - 'updated' (ISO 8601 string) if available
                - 'comment'
                - 'journal_ref'
                - 'doi'
                - 'primary_category'
                - 'categories' (list)
                - 'pdf_url'
                - 'arxiv_url'
                - 'bibtex' (a simple generated BibTeX entry)
        """
        mi = multi_input()
        input_type = input_type.lower()
        arxiv_id = ""
        if input_type in ("id", "arxiv_id"):
            arxiv_id = input
        elif input_type in ("bib", "arxiv_bib"):
            bib_dict = mi.extract_bib_from_string(input)
            arxiv_id = mi.extract_arxiv_id(bib_dict)
        elif input_type in ("url", "link"):
            arxiv_id = mi.arxiv_url_to_id(input)
        else:
            raise ValueError(f"Unknown input_type '{input_type}'. "
                            f"Expected one of: 'id', 'arxiv_id', 'bib', 'arxiv_bib', 'url', 'link'.")


        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(arxiv.Client().results(search))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch arXiv entry for {arxiv_id}: {e}")


        metadata = {}
        metadata['arxiv_id'] = arxiv_id
        metadata['title'] = getattr(paper, "title", None)
        
        authors = []
        try:
            for a in paper.authors:
                name = getattr(a, "name", a) if a is not None else None
                if name:
                    authors.append(name)
        except Exception:
            if isinstance(paper.authors, list):
                authors = paper.authors.copy()
        metadata['authors'] = authors

        metadata['summary'] = getattr(paper, "summary", None)
        pub = getattr(paper, "published", None)
        if isinstance(pub, (datetime.datetime,)):
            metadata['published'] = pub.isoformat()
        else:
            metadata['published'] = str(pub) if pub is not None else None
        upd = getattr(paper, "updated", None)
        if isinstance(upd, (datetime.datetime,)):
            metadata['updated'] = upd.isoformat()
        else:
            metadata['updated'] = str(upd) if upd is not None else None

        metadata['comment'] = getattr(paper, "comment", None)
        metadata['journal_ref'] = getattr(paper, "journal_ref", None)
        metadata['doi'] = getattr(paper, "doi", None)


        metadata['primary_category'] = getattr(paper, "primary_category", None)

        cats = getattr(paper, "categories", None)
        if isinstance(cats, (list, tuple)):
            metadata['categories'] = list(cats)
        else:
            if isinstance(cats, str):
                metadata['categories'] = cats.split()
            else:
                metadata['categories'] = None

        metadata['pdf_url'] = getattr(paper, "pdf_url", None)
        metadata['arxiv_url'] = getattr(paper, "entry_id", None)


        bib_id = arxiv_id.replace('/', '_')

        title_bib = metadata['title'] or ""

        title_bib = title_bib.replace("{", "\\{").replace("}", "\\}")
        authors_bib = " and ".join(authors) if authors else ""
        year = None
        try:
            if isinstance(pub, datetime.datetime):
                year = pub.year
            else:
                year = int(str(pub)[:4])
        except Exception:
            year = None
        bibtex_lines = [f"@article{{{bib_id},"]
        if title_bib:
            bibtex_lines.append(f"  title = {{{title_bib}}},")
        if authors_bib:
            bibtex_lines.append(f"  author = {{{authors_bib}}},")

        bibtex_lines.append(f"  journal = {{arXiv preprint arXiv:{arxiv_id}}},")
        if year:
            bibtex_lines.append(f"  year = {{{year}}},")

        if metadata.get('doi'):
            bibtex_lines.append(f"  doi = {{{metadata['doi']}}},")
        if metadata.get('journal_ref'):

            jr = metadata['journal_ref'].replace("{", "\\{").replace("}", "\\}")
            bibtex_lines.append(f"  note = {{{jr}}},")

        if len(bibtex_lines) > 1:
            last = bibtex_lines[-1]
            if last.endswith(','):
                bibtex_lines[-1] = last[:-1]
        bibtex_lines.append("}")
        metadata['bibtex'] = "\n".join(bibtex_lines)

        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
            # JSON
            json_path = os.path.join(dest_dir, f"{arxiv_id.replace('/', '_')}_metadata.json")
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                raise IOError(f"Could not write metadata JSON to {json_path}: {e}")

        return metadata

    @api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
    def download_semantic_scholar(self, input: str, input_type: str, dest_dir: str = None) -> None:

        mi = multi_input()
        input_type = input_type.lower()
        semantic_id = ""
        if input_type == "id" or input_type == "arxiv_id":
            semantic_id = input

        elif input_type == "bib" or input_type == "arxiv_bib":
            bib_dict = mi.extract_bib_from_string(input)
            semantic_id = mi.extract_arxiv_id(bib_dict)

        elif input_type == "url" or input_type == "link":
            semantic_id = mi.arxiv_url_to_id(input)

        else:
            # Raise error for unknown input_type
            raise ValueError(f"Unknown input_type '{input_type}'. "
                            f"Expected one of: 'id', 'arxiv_id', 'bib', 'arxiv_bib', 'url', 'link'.")

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
        mi = multi_input()

        input_type = input_type.lower()
        arxiv_id = ""
        if input_type == "id" or input_type == "arxiv_id":
            arxiv_id = input

        elif input_type == "bib" or input_type == "arxiv_bib":
            bib_dict = mi.extract_bib_from_string(input)
            arxiv_id = mi.extract_arxiv_id(bib_dict)

        elif input_type == "url" or input_type == "link":
            arxiv_id = mi.arxiv_url_to_id(input)

        else:
            # Raise error for unknown input_type
            raise ValueError(f"Unknown input_type '{input_type}'. "
                            f"Expected one of: 'id', 'arxiv_id', 'bib', 'arxiv_bib', 'url', 'link'.")

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

        mi = multi_input()

        input_type = input_type.lower()
        arxiv_id = ""
        if input_type == "id" or input_type == "arxiv_id":
            arxiv_id = input

        elif input_type == "bib" or input_type == "arxiv_bib":
            bib_dict = mi.extract_bib_from_string(input)
            arxiv_id = mi.extract_arxiv_id(bib_dict)

        elif input_type == "url" or input_type == "link":
            arxiv_id = mi.arxiv_url_to_id(input)

        else:
            # Raise error for unknown input_type
            raise ValueError(f"Unknown input_type '{input_type}'. "
                            f"Expected one of: 'id', 'arxiv_id', 'bib', 'arxiv_bib', 'url', 'link'.")

        self.build_paper_graph(input, input_type, dest_dir)

        json_path = f"{dest_dir}/output/{arxiv_id}.json"

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'abstract' not in data:
            raise KeyError(f"'abstract' field not found in JSON at {json_path}")

        return clean_latex_code(data['abstract'])

    def get_title(self, input: str, input_type: str, dest_dir: str = None) -> str:

        mi = multi_input()

        input_type = input_type.lower()
        arxiv_id = ""
        if input_type == "id" or input_type == "arxiv_id":
            arxiv_id = input

        elif input_type == "bib" or input_type == "arxiv_bib":
            bib_dict = mi.extract_bib_from_string(input)
            arxiv_id = mi.extract_arxiv_id(bib_dict)

        elif input_type == "url" or input_type == "link":
            arxiv_id = mi.arxiv_url_to_id(input)

        else:
            # Raise error for unknown input_type
            raise ValueError(f"Unknown input_type '{input_type}'. "
                            f"Expected one of: 'id', 'arxiv_id', 'bib', 'arxiv_bib', 'url', 'link'.")

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


id_string = "2501.02725"
dest_path = "./download"

mo = multi_download()

# mo.download_arxiv(id_string, "id", "both", dest_path)
# mo.download_metadata_arxiv(id_string, "id",  dest_path)
abstract = mo.get_title(id_string, "id",  dest_path)

print(abstract)

# sc_id = "39ad6c911f3351a3b390130a6e4265355b4d593b"
# mo = multi_download()

# mo.download_semantic_scholar(sc_id, "id")

