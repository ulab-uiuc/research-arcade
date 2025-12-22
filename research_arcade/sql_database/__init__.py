# research_arcade/sql_database/__init__.py

# --- OpenReview SQLs ---
from .sql_openreview_arxiv import SQLOpenReviewArxiv
from .sql_openreview_authors import SQLOpenReviewAuthors
from .sql_openreview_papers_authors import SQLOpenReviewPapersAuthors
from .sql_openreview_papers_reviews import SQLOpenReviewPapersReviews
from .sql_openreview_papers_revisions import SQLOpenReviewPapersRevisions
from .sql_openreview_papers import SQLOpenReviewPapers
from .sql_openreview_reviews import SQLOpenReviewReviews
from .sql_openreview_revisions_reviews import SQLOpenReviewRevisionsReviews
from .sql_openreview_revisions import SQLOpenReviewRevisions
from .sql_openreview_paragraphs import SQLOpenReviewParagraphs

# --- Arxiv SQLs ---
from .sql_arxiv_authors import SQLArxivAuthors
from .sql_arxiv_categories import SQLArxivCategory
from .sql_arxiv_citations import SQLArxivCitation
from .sql_arxiv_figures import SQLArxivFigure
from .sql_arxiv_paper_authors import SQLArxivPaperAuthor
from .sql_arxiv_paper_categories import SQLArxivPaperCategory
from .sql_arxiv_paper_figures import SQLArxivPaperFigure
from .sql_arxiv_paper_tables import SQLArxivPaperTable
from .sql_arxiv_papers import SQLArxivPapers
from .sql_arxiv_paragraphs import SQLArxivParagraphs
from .sql_arxiv_sections import SQLArxivSections
from .sql_arxiv_tables import SQLArxivTable
from .sql_arxiv_paragraph_citations import SQLArxivParagraphCitation
from .sql_arxiv_paragraph_references import SQLArxivParagraphReference
from .sql_arxiv_paragraph_figures import SQLArxivParagraphFigure
from .sql_arxiv_paragraph_tables import SQLArxivParagraphTable

__all__ = [
    # --- OpenReview ---
    'SQLOpenReviewArxiv',
    'SQLOpenReviewAuthors',
    'SQLOpenReviewPapersAuthors',
    'SQLOpenReviewPapersReviews',
    'SQLOpenReviewPapersRevisions',
    'SQLOpenReviewPapers',
    'SQLOpenReviewReviews',
    'SQLOpenReviewRevisionsReviews',
    'SQLOpenReviewRevisions',
    'SQLOpenReviewParagraphs',

    # --- Arxiv ---
    'SQLArxivAuthors',
    'SQLArxivCategory',
    'SQLArxivCitation',
    'SQLArxivFigure',
    'SQLArxivPaperAuthor',
    'SQLArxivPaperCategory',
    'SQLArxivPaperFigure',
    'SQLArxivPaperTable',
    'SQLArxivPapers',
    'SQLArxivParagraphReference',
    'SQLArxivParagraphs',
    'SQLArxivSections',
    'SQLArxivTable',
    'SQLArxivParagraphCitation',
    'SQLArxivParagraphFigure',
    'SQLArxivParagraphTable'
]
