from graph_constructor.database import Database
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
            # self.sch = SemanticScholar(api_key=api_key)
            self.sch = SemanticScholar()
    
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

    # Construct the paragraph given the paper id, index of paragraph, index of section, and the arxiv id that this paper belongs to
    def paragraph_constructor(self, paragraph, paragraph_index, section_index, arxiv_id):



        pass

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

        Assume that all the papers have been fully extracted.
        """

        # First thing first: check if the paper exists in the database
        # If so, stop and return None

        paper_exists = self.db.check_exist(arxiv_id)

        # TODO: remove it
        paper_exists = False

        times = {}

        # Find the corresponding files
        # json_path = f"{dir_path}/output/{arxiv_id}.json"
        json_path = f"{dir_path}/output/{arxiv_id}.json"
        print(f"File path: {json_path}")
        # metadata_path = f"{dir_path}/{arxiv_id}/{arxiv_id}_metadata.json"
        metadata_path = f"{dir_path}/{arxiv_id}/{arxiv_id}_metadata.json"

        if paper_exists:
            print(f"The paper with arxiv_id {arxiv_id} already exists in the database")
            return False
        # Then check if we have the json file of paper and meta_data
        try:
            with open(json_path, 'r') as file:
                file_json = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file '{file_json}' was not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{file_json}'. Check if the file contains valid JSON.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False


        metadata_json = None

        t0 = time.perf_counter()

        try:
            with open(metadata_path, 'r') as file:
                metadata_json = json.load(file)
        except FileNotFoundError:
            print(f"Error: The file '{metadata_path}' was not found.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{metadata_path}'. Check if the file contains valid JSON.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

        times['load_metadata'] = time.perf_counter() - t0

        print(f"Time of loading metadata: {times['load_metadata']}")

        t0 = time.perf_counter()
        self.paper_constructor_json(arxiv_id=arxiv_id, json_file=metadata_json)
        times['paper_constructor'] = time.perf_counter() - t0
        print(f"Time of constructing paper json file: {times['paper_constructor']}")

        t0 = time.perf_counter()
        authors = None

        # TODO: since a lot of latest papers are not yet on semantic scholar, we choose to move this process in later stages
        # Add the author into paper directory if the paper is on semantic scholar
        # base_arxiv_id, version = self.arxiv_id_processor(arxiv_id)
        # try:
        #     paper_sch = self.sch.get_paper(f"ARXIV:{base_arxiv_id}")
        #     authors = paper_sch.authors
        # except Exception as e:
        #     print(f"Paper with arxiv id {base_arxiv_id} not found on semantic scholar: {e}")


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

        #TODO here we change the way we store file path
        # We can test it alongside with new figure detection method
        for figure_json in figure_jsons:
            label = figure_json['label']
            caption = figure_json['caption']
            file_name = figure_json['figure_paths'][0]
            # path = f"{dir_path}/output/figures/figures_{file_name}"
            path = file_name

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
        print("Now processing citations")
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
            # if not cited_arxiv_id:
            #     # In that case, we need to use the tile for searching.
            #     # We first remove colon and plus sign in the title as they are prefix and relaitonal sign in the searching query
            #     title_cleaned = bib_title.replace('+', ' ').replace(':', '')

            #     # and we only use the family name of author
            #     bib_author_surname = bib_author.split(',')[0].strip()

            #     cited_arxiv_id = self.search_title_with_name(title=title_cleaned, name=bib_author_surname)

            self.db.insert_citation(citing_arxiv_id=arxiv_id, cited_arxiv_id=cited_arxiv_id, citing_sections=list(citing_sections),bib_title=bib_title, bib_key=bib_key, author_cited_paper=bib_author)

        times['citaion_extraction'] = time.perf_counter() - t0
        print(f"Time of searching arxiv id of cited paper (if not provided) and adding citation information to database: {times['citaion_extraction']}")

        return True

    def process_paragraphs(self, dir_path):
        """
        Process all the paragraphs after calling process_paper
        """
        
        paragraph_path = f"{dir_path}/output/paragraphs/text_nodes.jsonl"
        with open(paragraph_path) as f:
            data = [json.loads(line) for line in f]
        

        # Use arxiv_id + section name as key
        # Find the smallest paragraph_id generated by knowledge debugger
        # Subtract all paragraph id of the same section (of the same paper) with the smallest one to ensure that order starts with zero
        section_min_paragraph = {}

        for paragraph in data:
            paragraph_id = paragraph.get('id')
            # Extract paragraph_id
            id_number = self.get_paragraph_num(paragraph_id)
            paper_arxiv_id = paragraph.get('paper_id')
            paper_section = paragraph.get('section')
            if (paper_arxiv_id, paper_section) not in section_min_paragraph:
                section_min_paragraph[(paper_arxiv_id, paper_section)] = int(id_number)
            else:
                section_min_paragraph[(paper_arxiv_id, paper_section)] = min(section_min_paragraph[(paper_arxiv_id, paper_section)], int(id_number))


        
        for paragraph in data:
            paragraph_id = paragraph.get('id')
            content = paragraph.get('content')
            paper_arxiv_id = paragraph.get('paper_id')
            paper_section = paragraph.get('section')
            id_number = self.get_paragraph_num(paragraph_id)
            id_zero_based = id_number - section_min_paragraph[(paper_arxiv_id, paper_section)]
            self.db.insert_paragraph(paragraph_id=id_zero_based, content=content, paper_arxiv_id=paper_arxiv_id, paper_section=paper_section)

            paragraph_cite_bib_keys = paragraph.get('cites')
            for bib_key in paragraph_cite_bib_keys:
                self.db.insert_paragraph_citations(paragraph_id=id_zero_based, paper_section=paper_section, citing_arxiv_id=paper_arxiv_id, bib_key=bib_key)


            paragraph_ref_labels = paragraph.get('ref_labels')


            # def insert_paragraph_reference(self, paragraph_id, paper_arxiv_id, reference_label, reference_type=None):

            for ref_label in paragraph_ref_labels:

                ref_type = None
                # First search bib_key in databases.
                # If presented in one of them, we can determine the type of reference

                is_figure = self.db.check_exist_figure(bib_key=ref_label)
                is_table = self.db.check_exist_table(bib_key=ref_label)
                if is_figure:
                    ref_type = 'figure'
                elif is_table:
                    ref_type = 'table'


                self.db.insert_paragraph_reference(paragraph_id=id_zero_based, paper_section=paper_section, paper_arxiv_id=paper_arxiv_id, reference_label=ref_label, reference_type=ref_type)



            # For here, we also need to call insert_paragraph_reference and insert paragraphs-refs(including tables, figures, and more in the future) into the database



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

        print("Title of Cited Paper:")
        print(title)
        try:
            print("Result:")
            for result in search.results():
                print(result.title)
                if (self.title_cleaner(result.title )== title):
                        # and any(name.lower() in a.name.lower() for a in result.authors)):
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
    

    def author_processor(self, arxiv_id):

        """
        Motivation: sometimes the paper is not yet added into the semantic scholar. We can check if certain paper has author info in the db. If not, we search it on semantic scholar again.
        """

        # exists = self.db.paper_authors_exist(paper_arxiv_id=arxiv_id)

        base_arxiv_id, version = self.arxiv_id_processor(arxiv_id)
        print(f"base_arxiv_id: {base_arxiv_id}")
        try:
            paper_sch = self.sch.get_paper(f"ARXIV:{base_arxiv_id}")
            authors = paper_sch.authors
        except Exception as e:
            print(f"Paper with arxiv id {base_arxiv_id} not found on semantic scholar: {e}")
            return False


        # Add authors into database if not exist
        author_order = 0
        if authors:
            for author in authors:
                self.author_constructor(author.authorId)
                author_order += 1
                # Add paper-author edge as follows
                self.db.insert_paper_author(paper_arxiv_id=arxiv_id, author_id=author.authorId, author_sequence=author_order)

        return True


    def citation_processor(self, arxiv_id):
        """
        For all citations where we know the citing paper but the cited_arxiv_id is NULL or empty,
        try to look it up via title + author surname, and update the record in the DB.
        """
        # 1. Pull the relevant citation rows
        select_sql = """
            SELECT id, bib_title, author_cited_paper
            FROM citations
            WHERE citing_arxiv_id = %s
            AND (cited_arxiv_id IS NULL OR cited_arxiv_id = '')
        """
        self.db.cur.execute(select_sql, (arxiv_id,))
        rows = self.db.cur.fetchall()

        # 2. For each, try to look up the arXiv ID and write it back
        for row in rows:
            citation_id, bib_title, bib_author_full = row

            # clean up the title to remove problematic characters
            title_cleaned = self.title_cleaner(bib_title)
            # assume bib_author_full is "Last, First" or similar
            author_surname = bib_author_full.split(',')[0].strip()

            found_id = self.search_title_with_name(title=title_cleaned, name=author_surname)
            if found_id:
                update_sql = """
                    UPDATE citations
                    SET cited_arxiv_id = %s
                    WHERE id = %s
                """
                self.db.cur.execute(update_sql, (found_id, citation_id))
                print(f"Updated citation id {citation_id} → {found_id}")
            else:
                print(f"Could not find arXiv ID for citation {citation_id}: '{bib_title}' by {author_surname}")

        # 3. Commit once at the end (if your Database requires it)
        # If you're using autocommit=True, this is optional.
        try:
            self.db.conn.commit()
        except Exception:
            pass

        return True

    def title_cleaner(self, title: str) -> str:
        """
        Remove all symbols (non-alphanumeric, non-space characters) from the title.
        Collapses multiple spaces down to one and trims ends.
        """
        # Remove anything that isn't a letter, number, or whitespace
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', title)
        # Collapse multiple spaces and strip leading/trailing spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.strip().lower()

    def get_paragraph_num(self, pid):
        pattern = re.compile(r'^text_(\d+)$')
        m = pattern.match(pid)
        if not m:
            raise ValueError(f"Bad paragraph id format: {pid!r}")
        return int(m.group(1))
    
