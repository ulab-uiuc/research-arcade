from graph_constructor.database import Database
from semanticscholar import SemanticScholar
import arxiv
from arxiv import UnexpectedEmptyPageError
from multi_input.multi_input import MultiInput
from paper_collector.latex_parser import clean_latex_format
import json
import time
import os

from dotenv import load_dotenv

class NodeConstructor:

    """
    This class serves for converting entities such as authors, papers into nodes and insert them into the paper graph database
    """

    def __init__(self):
        self.db = Database()
        self.sch = None
        load_dotenv()
        api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        if not api_key:
            # We may still proceed, but it takes longer
            print("SEMANTIC_SCHOLAR_API_KEY not set in .env")
            self.sch = SemanticScholar()
        else:
            self.sch = SemanticScholar(api_key=api_key)
    
    # Construct the author node based on his or her semantic scholar id
    def author_constructor(self, semantic_scholar_id):
        author = self.sch.get_author(semantic_scholar_id)
        name = author.name
        url = author.url
        self.db.insert_author(semantic_scholar_id=semantic_scholar_id, name=name, homepage=url)

    # Construct the paper node based on the full information provided
    def paper_constructor(self, arxiv_id, title, abstract=None, submit_date=None, metadata=None):
        base_arxiv_id, version = self.arxiv_id_processor(arxiv_id=arxiv_id)
        self.db.insert_paper(arxiv_id=arxiv_id, base_arxiv_id=base_arxiv_id, version=version, title=title, abstract=abstract, submit_date=submit_date, metadata=metadata)

    # Construct the paper node based on the arxiv_id used for fetching the rest of information from SemanticScholar
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
        self.db.insert_paper(arxiv_id=arxiv_id, base_arxiv_id=base_arxiv_id, version=version, title=title, abstract=abstract, submit_date=submit_date, metadata=metadata)

    # Construct the paper node based on given json file with information provided
    def paper_constructor_json(self, arxiv_id, json_file):
        
        # print(json_file)
        title = json_file['title']
        abstract = clean_latex_format(json_file['abstract'])
        submit_date = json_file['published']
        base_arxiv_id, version = self.arxiv_id_processor(arxiv_id=arxiv_id)
        self.db.insert_paper(arxiv_id=arxiv_id, base_arxiv_id=base_arxiv_id, version=version, title=title, abstract=abstract, submit_date=submit_date, metadata=str(json_file))


    # Construct category node with given category name and category description
    def category_constructor(self, name, description=None):
        self.db.insert_category(name=name, description=description)

    # Construct institution node with institution name and location
    def institution_constructor(self, name, location=None):
        self.db.insert_institution(name=name, location=location)

    # Construct figure node given the paper id, index of figure (?), path to figure and caption/label
    def figure_constructor(self, paper_id, figure_index, path, caption=None):
        self.db.insert_figure(paper_arxiv_id=paper_id, figure_index=figure_index, path=path, caption=caption)

    # Construct tabel node given the paper id, index of figure (?), path to figure and caption/label
    def figure_constructor(self, paper_id, table_index, path, caption=None):
        self.db.insert_figure(paper_arxiv_id=paper_id, table_index=table_index, path=path, caption=caption)


    '''
    Here, not that in the future when we process papers, we can authomatically add figures and tables into database and construct the papers into it
    '''


    '''
    Also, we need to build methods that can automatically store figures into
    '''

    # Given the arxiv_id and the directory path that stores the extracted paper information, store the paper, author, figure and tables into the database
    def process_paper(self, arxiv_id, dir_path):
        """
        Given a paper:
        1. Store it as a node
        2. Build edge to paper authors. If the author does not exist, create one
        """

        # First thing first: check if the paper exists in the database
        # If so, stop and return None

        paper_exists = self.db.check_exist(arxiv_id)


        times = {}

        # Find the corresponding files
        # json_path = f"{dir_path}/output/endpoints/{arxiv_id}.json"
        json_path = f"{dir_path}/output/endpoints/{arxiv_id}.json"
        # metadata_path = f"{dir_path}/{arxiv_id}/{arxiv_id}_metadata.json"
        metadata_path = f"{dir_path}/{arxiv_id}/{arxiv_id}_metadata.json"

        if paper_exists:
            print(f"The paper with arxiv_id {arxiv_id} already exists in the database")
            return
        # Then check if we have the json file of paper and meta_data
        try:
            with open(json_path, 'r') as file:
                file_json = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file '{file_json}' was not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{file_json}'. Check if the file contains valid JSON.")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return


        metadata_json = None

        t0 = time.perf_counter()

        try:
            with open(metadata_path, 'r') as file:
                metadata_json = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file '{metadata_path}' was not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{metadata_path}'. Check if the file contains valid JSON.")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return

        times['load_metadata'] = time.perf_counter() - t0

        print(f"Time of loading metadata: {times['load_metadata']}")

        t0 = time.perf_counter()
        self.paper_constructor_json(arxiv_id=arxiv_id, json_file=metadata_json)
        times['paper_constructor'] = time.perf_counter() - t0
        print(f"Time of constructing paper json file: {times['paper_constructor']}")

        t0 = time.perf_counter()
        authors = None
        # Add the author into paper directory if the paper is on semantic scholar
        base_arxiv_id, version = self.arxiv_id_processor(arxiv_id)
        try:
            paper_sch = self.sch.get_paper(f"ARXIV:{base_arxiv_id}")
            authors = paper_sch.authors
        except Exception as e:
            print(f"Paper with arxiv id {base_arxiv_id} not found on semantic scholar: {e}")


        # Add authors into database if not exist
        author_order = 0
        if authors:
            for author in authors:
                self.author_constructor(author.authorId)
                author_order += 1
                # Add paper-author edge as follows
                self.db.insert_paper_author(paper_arxiv_id=arxiv_id, author_id=author.authorId, author_sequence=author_order)

        times['author_adding'] = time.perf_counter() - t0
        print(f"Time of finding authors and adding authors to database: {times['author_adding']}")

        # Add figures to papers
        # Here we store the path to figures/images instead of directly storing them inside of the database
        # We don't really need the figure id LOL
        # Here, we use the existing json file of paper to extract the figure information


        t0 = time.perf_counter()


        # Go through all the figure files following the extracted information in json

        section_jsons = file_json['sections']

        for title, section_json in section_jsons.items():
            is_appendix = section_json['appendix'] == 'true'
            content = section_json['content']
            self.db.insert_section(content=content, title=title, is_appendix=is_appendix, paper_arxiv_id=arxiv_id)

        figure_jsons = file_json['figure']

        for figure_json in figure_jsons:
            label = figure_json['label']
            caption = figure_json['caption']
            file_name = figure_json['figure_paths'][0]
            file_name = file_name.split('/')[-1]
            path = f"{dir_path}/output/figures/figures_{file_name}"

            figure_id = self.db.insert_figure(paper_arxiv_id=arxiv_id, path=path, caption=caption, label=label, name=file_name)

            self.db.insert_paper_figure(paper_arxiv_id=arxiv_id, figure_id=figure_id)


            # We do the same thing to tables
            # For tables, we currently do not have a good way to reconstruct the table. Instead, we store the raw textual data of tables in the database directly.

        table_jsons = file_json['table']
        for table_json in table_jsons:
            
            caption = table_json['caption']
            label = table_json['label']
            table = table_json['tabular']
            # We don't currently store the table anywhere as a file so the table path is empty
            path = None
            
            table_id = self.db.insert_table(paper_arxiv_id=arxiv_id, path=path, caption=caption, label=label, table_text=table)
            
            self.db.insert_paper_table(paper_arxiv_id = arxiv_id, table_id=table_id)


        # We now add the table to the category.
        # We first insert the category into database.
        categories = file_json['categories']



        for category in categories:
            category_id = self.db.insert_category(category)
            self.db.insert_paper_category(category_id=category_id, paper_arxiv_id=arxiv_id)

        times['info_extraction'] = time.perf_counter() - t0
        print(f"Time of adding figures, tables, and sections to database: {times['info_extraction']}")

        # The last thing is to deal with citation maps.
        # For now we simply loop throught the citations part of the paper and obtain the arxiv ids of cited papers.
        # Comparing the semantic scholar, we choose to use extracted citations since they all provide arxiv ids.
        # Recall that we have to make sure every paper can be traced to its arxiv id since only downloading from arxiv provides us full context.


        t0 = time.perf_counter()

        for citation in file_json['citations'].values():
            # print(f"Citation: {citation}")
            cited_arxiv_id = citation.get('arxiv_id')
            bib_key = citation.get('bib_key')
            bib_title = citation.get('bib_title')
            bib_author = citation.get('bib_author ')
            contexts = citation.get('context')
            citing_sections = set()
            for context in contexts:
                citing_section = context['section']
                citing_sections.add(citing_section)
            # It seems that the cited paper sometimes does not provide arxiv id, or that column is null. How can I tackle this issue?
            if not cited_arxiv_id:
                # In that case, we need to use the tile for searching.
                # We first remove colon and plus sign in the title as they are prefix and relaitonal sign in the searching query
                title_cleaned = bib_title.replace('+', ' ').replace(':', '')

                # and we only use the family name of author
                bib_author_surname = bib_author.split(',')[0].strip()
                
                cited_arxiv_id = self.search_title_with_name(title=title_cleaned, name=bib_author_surname)
                
            self.db.insert_citation(citing_arxiv_id=arxiv_id, cited_arxiv_id=cited_arxiv_id, citing_sections=list(citing_sections),bib_title=bib_title, bib_key=bib_key, author_cited_paper=bib_author)
        
        times['citaion_extraction'] = time.perf_counter() - t0
        print(f"Time of searching arxiv id of cited paper (if not provided) and adding citation information to database: {times['citaion_extraction']}")


    def create_tables(self):
        self.db.create_all()

    def drop_tables(self):
        self.db.drop_all()
    
    def search_title_with_name(self, title, name, max_result=20):
        query = f"ti:{title} AND au:{name}"
        search = arxiv.Search(
            query=query,
            max_results=max_result,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        try:
            for result in search.results():
                if (result.title.strip().lower() == title.strip().lower()
                        and any(name.lower() in a.name.lower() for a in result.authors)):
                    return result.entry_id
        except UnexpectedEmptyPageError:
            # no more pages—stop iterating
            pass

        return None

    def arxiv_id_processor(self, arxiv_id):
        """
        Given arxiv id, return base arxiv id and version
        - arxiv_id: str
        """
        return arxiv_id.split('v')
    