from graph_constructor.database import Database
from semanticscholar import SemanticScholar
import arxiv
from multi_input.multi_input import MultiInput


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
            'abstract': paper.summary,
            'authors': [a.name for a in paper.authors],
            'published': str(paper.published),
            'categories': paper.categories,
            'url': paper.entry_id,
        }

        abstract = paper.summary
        title = paper.title
        submit_date = str(paper.published)

        self.db.insert_paper(arxiv_id=arxiv_id, title=title, abstract=abstract, submit_date=submit_date, metadata=metadata)


    # Construct the paper node based on given json file with information provided
    def paper_constructor_arxiv_id(self, arxiv_id, json_file):
        
        title = json_file['title']
        abstract = json_file['abstract']
        submit_date = json_file['published']
        self.db.insert_paper(arxiv_id=arxiv_id, title=title, abstract=abstract, submit_date=submit_date, metadata=str(json_file))

    # Construct category node with given category name and category description
    def category_constructor(self, name, description=None):
        self.db.insert_category(name=name, description=description)

    # Construct institution node with institution name and location
    def institution_constructor(self, name, location=None):
        self.db.insert_institution(name=name, location=location)

    # Construct figure node given the paper id, index of figure (?), path to figure and caption/label
    def figure_constructor(self, paper_id, figure_index, path, caption=None):
        self.db.insert_figure(paper_id=paper_id, figure_index=figure_index, path=path, caption=caption)

    # Construct tabel node given the paper id, index of figure (?), path to figure and caption/label
    def figure_constructor(self, paper_id, table_index, path, caption=None):
        self.db.insert_figure(paper_id=paper_id, table_index=table_index, path=path, caption=caption)
    

    '''
    Here, not that in the future when we process papers, we can authomatically add figures and tables into database and construct the papers into it
    '''


    '''
    Also, we need to build methods that can automatically store figures into
    '''
    
    # Given the arxiv_id and the directory path that stores the extracted paper information, store the paper, author, figure and tables into the database
    def paper_processor(self, arxiv_id, dir_path):
        pass
        
