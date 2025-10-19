# from typing import List

# """
# Below are node classes
# """

# class Paper:
#     id: int
#     arxiv_id: str
#     base_arxiv_id: str
#     version: str
#     title: str
#     abstract: str
#     submit_date: str
#     metadata: str

# class Section:
#     id: int
#     content: str 
#     title: str
#     appendix: bool
#     paper_arxiv_id: str

# class Paragraph:
#     id: int
#     paragraph_id: int
#     content: str
#     paper_arxiv_id: str
#     paper_section: str
#     # This part is to be added
#     in_paper_order: int

# class Authors:
#     # id (SERIAL PK)
#     id: int
#     semantic_scholar_id: str
#     name: str
#     homepage: str

# class Category:
#     id: int
#     name: str
#     description: str

# class Figure:
#     id: int
#     paper_arxiv_id: str
#     path: str
#     caption: str
#     label: str
#     name: str

# class Table:
#     id: int
#     paper_arxiv_id: str
#     path: str
#     caption: str
#     label: str
#     table_text: str

# """
# Below are edge classes
# """
# class PaperAuthor:
#     paper_arxiv_id: str
#     author_id: str
#     author_sequence: int

# class PaperCategory:
#     paper_arxiv_id: str
#     category_id: int

# class Citation:
#     citing_arxiv_id: str
#     cited_arxiv_id: str
#     citing_sections: List[str]

# class PaperFigure:
#     paper_arxiv_id: str
#     figure_id: int

# class PaperTable:
#     paper_arxiv_id: str
#     table_id: int


# class ParagraphCitation:
#     id: int
#     paragraph_id: int
#     paper_section: str
#     citing_arxiv_id: str
#     bib_key: str

# # Paragraph reference includes paragraph figure, paragraph table and other paragraph stuff. Here, more stuff needs to be incorporated

# class ParagraphReference:
#     id: int
#     paragraph_id : int
#     paragraph_section: str
#     paper_arxiv_id: str
#     reference_label: str
#     reference_type: str

#     #TODO: add these columns into database so that the edge connection is more direct
#     paragraph_global_id: int

# # TODO: add these two tables into the database along with a post data processing stage
# class ParagraphFigure:
#     id: int
#     paragraph_id : int
#     paragraph_section: str
#     paper_arxiv_id: str
#     paragraph_global_id: int
#     figure_id: int

# class ParagraphTable:
#     id: int
#     paragraph_id : int
#     paragraph_section: str
#     paper_arxiv_id: str
#     paragraph_global_id: int
#     table_id: int

from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Paper:
    arxiv_id: str
    base_arxiv_id: str
    version: str
    title: str
    abstract: Optional[str] = None
    submit_date: Optional[str] = None
    metadata: Optional[str] = None
    id: Optional[int] = None

@dataclass
class Section:
    content: str 
    title: str
    appendix: bool
    paper_arxiv_id: str
    id: Optional[int] = None

@dataclass
class Paragraph:
    paragraph_id: int
    content: str
    paper_arxiv_id: str
    paper_section: str
    in_paper_order: Optional[int] = None
    id: Optional[int] = None

@dataclass
class Author:
    semantic_scholar_id: str
    name: str
    homepage: Optional[str] = None
    id: Optional[int] = None

@dataclass
class Category:
    name: str
    description: Optional[str] = None
    id: Optional[int] = None

@dataclass
class Figure:
    paper_arxiv_id: str
    path: Optional[str] = None
    caption: Optional[str] = None
    label: Optional[str] = None
    name: Optional[str] = None
    id: Optional[int] = None

@dataclass
class Table:
    paper_arxiv_id: str
    path: Optional[str] = None
    caption: Optional[str] = None
    label: Optional[str] = None
    table_text: Optional[str] = None
    id: Optional[int] = None

"""
Edge classes
"""
@dataclass
class PaperAuthor:
    paper_arxiv_id: str
    author_id: str
    author_sequence: int

@dataclass
class PaperCategory:
    paper_arxiv_id: str
    category_id: int

@dataclass
class Citation:
    citing_arxiv_id: str
    cited_arxiv_id: str
    bib_title: Optional[str] = None
    bib_key: Optional[str] = None
    author_cited_paper: Optional[str] = None
    citing_sections: Optional[List[str]] = None
    citing_paragraphs: Optional[List[int]] = None
    id: Optional[int] = None

@dataclass
class PaperFigure:
    paper_arxiv_id: str
    figure_id: int

@dataclass
class PaperTable:
    paper_arxiv_id: str
    table_id: int

@dataclass
class ParagraphCitation:
    paragraph_id: int
    paper_section: str
    citing_arxiv_id: str
    bib_key: str
    id: Optional[int] = None

@dataclass
class ParagraphReference:
    paragraph_id: int
    paper_section: str
    paper_arxiv_id: str
    reference_label: str
    reference_type: Optional[str] = None
    id: Optional[int] = None
