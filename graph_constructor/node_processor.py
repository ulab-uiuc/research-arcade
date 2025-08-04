from graph_constructor.csv_graph import CSVGraph
from semanticscholar import SemanticScholar
import arxiv
from arxiv import UnexpectedEmptyPageError
from multi_input.multi_input import MultiInput
from paper_collector.latex_parser import clean_latex_format
import re
import json
import time
import os
from dotenv import load_dotenv
from typing import List, Tuple, Optional

DIR = "./csv"

class NodeConstructor:
    """
    Converts entities (authors, papers, etc.) into nodes and inserts them into the CSV-backed paper graph.
    """

    def __init__(self):
        self.csvGraph = CSVGraph(csv_dir=DIR)
        self.sch = None
        load_dotenv()
        api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        if not api_key:
            print("SEMANTIC_SCHOLAR_API_KEY not set in .env")
            self.sch = SemanticScholar()
        else:
            self.sch = SemanticScholar()

    # -------- constructors --------
    def author_constructor(self, semantic_scholar_id):
        author = self.sch.get_author(semantic_scholar_id)
        name = author.name
        url = author.url
        self.csvGraph.insert_author(semantic_scholar_id=semantic_scholar_id, name=name, homepage=url)

    def paper_constructor(self, arxiv_id, title, abstract=None, submit_date=None, metadata=None):
        base_arxiv_id, version = self.arxiv_id_processor(arxiv_id=arxiv_id)
        self.csvGraph.insert_paper(arxiv_id=arxiv_id, base_arxiv_id=base_arxiv_id,
                                   version=version, title=title, abstract=abstract,
                                   submit_date=submit_date, metadata=metadata)

    def paper_constructor_arxiv_id(self, arxiv_id):
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(arxiv.Client().results(search))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch arXiv entry for {arxiv_id}: {e}")

        metadata = {
            'id': arxiv_id,
            'title': paper.title,
            'summary': paper.summary,
            'authors': [a.name for a in paper.authors],
            'published': str(paper.published),
            'categories': paper.categories,
            'url': paper.entry_id,
        }

        abstract = clean_latex_format(paper.summary)
        title = paper.title
        submit_date = str(paper.published)
        base_arxiv_id, version = self.arxiv_id_processor(arxiv_id=arxiv_id)
        self.csvGraph.insert_paper(arxiv_id=arxiv_id, base_arxiv_id=base_arxiv_id,
                                   version=version, title=title, abstract=abstract,
                                   submit_date=submit_date, metadata=metadata)

    def paper_constructor_json(self, arxiv_id, json_file):
        title = json_file['title']
        abstract = clean_latex_format(json_file['abstract'])
        submit_date = json_file['published']
        base_arxiv_id, version = self.arxiv_id_processor(arxiv_id=arxiv_id)
        self.csvGraph.insert_paper(arxiv_id=arxiv_id, base_arxiv_id=base_arxiv_id,
                                   version=version, title=title, abstract=abstract,
                                   submit_date=submit_date, metadata=str(json_file))

    def category_constructor(self, name, description=None):
        self.csvGraph.insert_category(name=name, description=description)

    def institution_constructor(self, name, location=None):
        self.csvGraph.insert_institution(name=name, location=location)

    # -------- paper processing --------
    def process_paper(self, arxiv_id, dir_path):
        """
        Given a paper, store it and its associated metadata (sections, figures, tables, categories, citations).
        """
        # Existence check
        paper_exists = self.csvGraph.check_exist(arxiv_id)
        # TODO: remove forced override if not desired
        paper_exists = False

        times = {}
        json_path = f"{dir_path}/output/{arxiv_id}.json"
        metadata_path = f"{dir_path}/{arxiv_id}/{arxiv_id}_metadata.json"

        if paper_exists:
            print(f"The paper with arxiv_id {arxiv_id} already exists in the database")
            return False

        try:
            with open(json_path, 'r') as file:
                file_json = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file '{json_path}' was not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{json_path}'.")
            return False
        except Exception as e:
            print(f"Unexpected error loading {json_path}: {e}")
            return False

        try:
            with open(metadata_path, 'r') as file:
                metadata_json = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file '{metadata_path}' was not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{metadata_path}'.")
            return False
        except Exception as e:
            print(f"Unexpected error loading {metadata_path}: {e}")
            return False

        # Insert paper
        t0 = time.perf_counter()
        self.paper_constructor_json(arxiv_id=arxiv_id, json_file=metadata_json)
        times['paper_constructor'] = time.perf_counter() - t0
        print(f"Time of constructing paper json file: {times['paper_constructor']}")

        # Authors (optional; skipped if not available)
        author_order = 0
        # placeholder for future Semantic Scholar author fetching logic

        # Sections
        t0 = time.perf_counter()
        section_jsons = file_json.get('sections', {})
        for title, section_json in section_jsons.items():
            is_appendix = section_json.get('appendix', '') == 'true'
            content = section_json.get('content', '')
            self.csvGraph.insert_section(content=content, title=title, is_appendix=is_appendix, paper_arxiv_id=arxiv_id)

        # Figures
        figure_jsons = file_json.get('figure', [])
        for figure_json in figure_jsons:
            figures = self.figure_iteration_recursive(figure_json=figure_json)
            for figure in figures:
                path, caption, label = figure
                figure_id = self.csvGraph.insert_figure(paper_arxiv_id=arxiv_id, path=path, caption=caption, label=label, name=None)
                self.csvGraph.insert_paper_figure(paper_arxiv_id=arxiv_id, figure_id=figure_id)

        # Tables
        table_jsons = file_json.get('table', [])
        for table_json in table_jsons:
            caption = table_json.get('caption')
            label = table_json.get('label')
            table = table_json.get('tabular')
            path = None
            table_id = self.csvGraph.insert_table(paper_arxiv_id=arxiv_id, path=path, caption=caption, label=label, table_text=table)
            self.csvGraph.insert_paper_table(paper_arxiv_id=arxiv_id, table_id=table_id)

        # Categories
        categories = file_json.get('categories', [])
        for category in categories:
            category_id = self.csvGraph.insert_category(category)
            self.csvGraph.insert_paper_category(paper_arxiv_id=arxiv_id, category_id=category_id)

        times['info_extraction'] = time.perf_counter() - t0
        print(f"Time of adding figures, tables, and sections to database: {times['info_extraction']}")

        # Citations
        t0 = time.perf_counter()
        print("Now processing citations")
        for citation in file_json.get('citations', {}).values():
            cited_arxiv_id = citation.get('arxiv_id')
            bib_key = citation.get('bib_key')
            bib_title = citation.get('bib_title')
            bib_author = citation.get('bib_author ')
            contexts = citation.get('context', [])
            citing_sections = set()
            for context in contexts:
                citing_section = context.get('section')
                if citing_section:
                    citing_sections.add(citing_section)

            self.csvGraph.insert_citation(
                citing_arxiv_id=arxiv_id,
                cited_arxiv_id=cited_arxiv_id,
                bib_title=bib_title,
                bib_key=bib_key,
                author_cited_paper=bib_author,
                citing_sections=list(citing_sections)
            )
        times['citaion_extraction'] = time.perf_counter() - t0
        print(f"Time of processing citations: {times['citaion_extraction']}")

        return True

    def figure_iteration_recursive(self, figure_json):
        path_to_info: List[Tuple[str, str, str]] = []

        def figure_iteration(fj):
            if not fj:
                return
            if fj.get('figure_paths'):
                path = fj['figure_paths'][0]
                caption = fj.get('caption')
                label = fj.get('label')
                path_to_info.append((path, caption, label))
            for sub in fj.get('subfigures', []):
                figure_iteration(sub)

        figure_iteration(fj=figure_json)
        return path_to_info

    def process_paragraphs(self, dir_path):
        paragraph_path = f"{dir_path}/output/paragraphs/text_nodes.jsonl"
        with open(paragraph_path) as f:
            data = [json.loads(line) for line in f]

        section_min_paragraph = {}
        for paragraph in data:
            paragraph_id = paragraph.get('id')
            id_number = self.get_paragraph_num(paragraph_id)
            paper_arxiv_id = paragraph.get('paper_id')
            paper_section = paragraph.get('section')
            key = (paper_arxiv_id, paper_section)
            if key not in section_min_paragraph:
                section_min_paragraph[key] = int(id_number)
            else:
                section_min_paragraph[key] = min(section_min_paragraph[key], int(id_number))

        for paragraph in data:
            paragraph_id = paragraph.get('id')
            content = paragraph.get('content')
            paper_arxiv_id = paragraph.get('paper_id')
            paper_section = paragraph.get('section')
            id_number = self.get_paragraph_num(paragraph_id)
            id_zero_based = id_number - section_min_paragraph[(paper_arxiv_id, paper_section)]
            self.csvGraph.insert_paragraph(paragraph_id=id_zero_based, content=content,
                                           paper_arxiv_id=paper_arxiv_id, paper_section=paper_section)

            paragraph_cite_bib_keys = paragraph.get('cites', [])
            for bib_key in paragraph_cite_bib_keys:
                self.csvGraph.insert_paragraph_citations(paragraph_id=id_zero_based,
                                                         paper_section=paper_section,
                                                         citing_arxiv_id=paper_arxiv_id,
                                                         bib_key=bib_key)

            paragraph_ref_labels = paragraph.get('ref_labels', [])
            for ref_label in paragraph_ref_labels:
                ref_type = None
                is_figure = self.csvGraph.check_exist_figure(bib_key=ref_label)
                is_table = self.csvGraph.check_exist_table(bib_key=ref_label)
                if is_figure:
                    ref_type = 'figure'
                elif is_table:
                    ref_type = 'table'

                self.csvGraph.insert_paragraph_reference(paragraph_id=id_zero_based,
                                                        paper_section=paper_section,
                                                        paper_arxiv_id=paper_arxiv_id,
                                                        reference_label=ref_label,
                                                        reference_type=ref_type)

    def create_tables(self):
        # no-op for CSVGraph; kept for interface compatibility
        pass

    def drop_tables(self):
        # could implement wiping CSVs if needed
        pass

    # -------- citation fixing logic for csv version --------
    def citation_processor(self, arxiv_id):
        """
        Try to fill missing cited_arxiv_id by title+author surname lookup.
        """
        if self.csvGraph.citations.empty:
            return True
        mask = (self.csvGraph.citations["citing_arxiv_id"] == arxiv_id) & (
            (self.csvGraph.citations["cited_arxiv_id"].isna()) | (self.csvGraph.citations["cited_arxiv_id"] == "")
        )
        subset = self.csvGraph.citations[mask]
        for idx, row in subset.iterrows():
            bib_title = row.get("bib_title", "")
            bib_author_full = row.get("author_cited_paper", "")
            title_cleaned = self.title_cleaner(bib_title)
            author_surname = bib_author_full.split(',')[0].strip() if bib_author_full else ""
            found_id = self.search_title_with_name(title=title_cleaned, name=author_surname)
            if found_id:
                self.csvGraph.citations.at[idx, "cited_arxiv_id"] = found_id
                print(f"Updated citation row {row.get('id')} → {found_id}")
            else:
                print(f"Could not find arXiv ID for citation {row.get('id')}: '{bib_title}' by {author_surname}")
        # persist changes
        self.csvGraph._save("citations", self.csvGraph.citations)
        return True

    def search_title_with_name(self, title, name, max_result=20):
        query = f"ti:{title} AND au:{name}"
        search = arxiv.Search(
            query=query,
            max_results=max_result,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        print("Title of Cited Paper:")
        print(title)
        try:
            for result in search.results():
                if (self.title_cleaner(result.title) == title):
                    return result.entry_id
        except UnexpectedEmptyPageError:
            pass
        return None

    def arxiv_id_processor(self, arxiv_id):
        return arxiv_id.split('v')

    def author_processor(self, arxiv_id):
        base_arxiv_id, version = self.arxiv_id_processor(arxiv_id)
        print(f"base_arxiv_id: {base_arxiv_id}")
        try:
            paper_sch = self.sch.get_paper(f"ARXIV:{base_arxiv_id}")
            authors = paper_sch.authors
        except Exception as e:
            print(f"Paper with arxiv id {base_arxiv_id} not found on semantic scholar: {e}")
            return False

        author_order = 0
        if authors:
            for author in authors:
                self.author_constructor(author.authorId)
                author_order += 1
                self.csvGraph.insert_paper_author(paper_arxiv_id=arxiv_id, author_id=author.authorId, author_sequence=author_order)
        return True

    def title_cleaner(self, title: str) -> str:
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', title)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.strip().lower()

    def get_paragraph_num(self, pid):
        pattern = re.compile(r'^text_(\d+)$')
        m = pattern.match(pid)
        if not m:
            raise ValueError(f"Bad paragraph id format: {pid!r}")
        return int(m.group(1))