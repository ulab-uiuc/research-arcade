# research_arcade/csv_database/__init__.py

# --- OpenReview CSVs ---
from .csv_openreview_arxiv import CSVOpenReviewArxiv
from .csv_openreview_authors import CSVOpenReviewAuthors
from .csv_openreview_papers_authors import CSVOpenReviewPapersAuthors
from .csv_openreview_papers_reviews import CSVOpenReviewPapersReviews
from .csv_openreview_papers_revisions import CSVOpenReviewPapersRevisions
from .csv_openreview_papers import CSVOpenReviewPapers
from .csv_openreview_reviews import CSVOpenReviewReviews
from .csv_openreview_revisions_reviews import CSVOpenReviewRevisionsReviews
from .csv_openreview_revisions import CSVOpenReviewRevisions
from .csv_openreview_paragraphs import CSVOpenReviewParagraphs

# --- Arxiv CSVs ---
from .csv_arxiv_authors import CSVArxivAuthors
from .csv_arxiv_categories import CSVArxivCategory
from .csv_arxiv_citations import CSVArxivCitation
from .csv_arxiv_figures import CSVArxivFigure
from .csv_arxiv_paper_authors import CSVArxivPaperAuthor
from .csv_arxiv_paper_categories import CSVArxivPaperCategory
from .csv_arxiv_paper_figures import CSVArxivPaperFigure
from .csv_arxiv_paper_tables import CSVArxivPaperTable
from .csv_arxiv_papers import CSVArxivPapers
from .csv_arxiv_paragraph_references import CSVArxivParagraphReference
from .csv_arxiv_paragraphs import CSVArxivParagraphs
from .csv_arxiv_sections import CSVArxivSections
from .csv_arxiv_tables import CSVArxivTable
from .csv_arxiv_paragraph_citations import CSVArxivParagraphCitation

__all__ = [
    # --- OpenReview ---
    'CSVOpenReviewArxiv',
    'CSVOpenReviewAuthors',
    'CSVOpenReviewPapersAuthors',
    'CSVOpenReviewPapersReviews',
    'CSVOpenReviewPapersRevisions',
    'CSVOpenReviewPapers',
    'CSVOpenReviewReviews',
    'CSVOpenReviewRevisionsReviews',
    'CSVOpenReviewRevisions',
    'CSVOpenReviewParagraphs',

    # --- Arxiv ---
    'CSVArxivAuthors',
    'CSVArxivCategory',
    'CSVArxivCitation',
    'CSVArxivFigure',
    'CSVArxivPaperAuthor',
    'CSVArxivPaperCategory',
    'CSVArxivPaperFigure',
    'CSVArxivPaperTable',
    'CSVArxivPapers',
    'CSVArxivParagraphReference',
    'CSVArxivParagraphs',
    'CSVArxivSections',
    'CSVArxivTable',
    'CSVArxivParagraphCitation'
]
