from graph_constructor.database import Database
from semanticscholar import SemanticScholar
import arxiv


class NodeConstructor:

    """
    This class serves for converting entities such as authors, papers into nodes and insert them into the paper graph database
    """

    def __init__(self):
        self.db = Database()
        self.sch = SemanticScholar()
        # pass
    
    # Construct the author node based on his or her semantic scholar id
    def author_constructor(self, semantic_scholar_id):
        author = self.sch.get_author(semantic_scholar_id)
        name = author.name
        url = author.url
        self.db.insert_author(semantic_scholar_id=semantic_scholar_id, name=name, url=url)

    
    # Construct the paper node based on the full information provided
    def paper_constructor_arxiv_id(self, arxiv_id, title, abstract=None, submit_date=None, metadata=None):
        self.db.insert_paper(arxiv_id=arxiv_id, title=title, abstract=abstract, submit_date=submit_date, metadata=metadata)
        pass

    # Construct the paper node based on the arxiv_id used for fetching the rest of information from SemanticScholar
    def paper_constructor_arxiv_id(self, arxiv_id):
        paper = self.sch.get_paper(arxiv_id)
        title = paper.title
        abstract = paper.
        pass

    # Construct the paper node based on given json file with information provided
    def paper_constructor_arxiv_id(self, json_file):
        pass

    # Construct category node with given category name and category description
    def category_constructor(self, name, description=None):
        pass

    # Construct institution node with institution name and location
    def institution_constructor(self, name, location=None):
        pass

    # Construct figure node given the paper id, index of figure (?), path to figure and caption/label
    def figure_constructor(self, paper_id, figure_index, path, caption=None):
        pass

    # Construct tabel node given the paper id, index of figure (?), path to figure and caption/label
    def figure_constructor(self, paper_id, table_index, path, caption=None):
        pass
    

    '''
    Here, not that in the future when we process papers, we can authomatically add figures and tables into database and construct the papers into it
    '''


    '''
    Also, we need to build methods that can automatically store figures into
    '''
    
    # Given the arxiv_id and the directory path that stores the extracted paper information, store the paper, author, figure and tables into the database
    def paper_processor(self, arxiv_id, dir_path):
        pass

