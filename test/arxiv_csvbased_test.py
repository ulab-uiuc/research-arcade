import sys
import os

import unittest
from unittest.mock import Mock, patch

from dotenv import load_dotenv  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_arcade.research_arcade import ResearchArcade
load_dotenv()


class TestArxivArcadeNodeCRUD(unittest.TestCase):
    """Test suite for Node CRUD operations"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.config = os.getenv('CSV_DATASET_FOLDER_PATH')
        self.config = {'csv_dir': './arxiv_csv'}
        self.db_type = "csv"
        
        # Mock all CSV database classes
        with patch('research_arcade.csv_database.csv_arxiv_authors.CSVArxivAuthors'), \
            patch('research_arcade.csv_database.csv_arxiv_categories.CSVArxivCategory'), \
            patch('research_arcade.csv_database.csv_arxiv_citations.CSVArxivCitation'), \
            patch('research_arcade.csv_database.csv_arxiv_figures.CSVArxivFigure'), \
            patch('research_arcade.csv_database.csv_arxiv_tables.CSVArxivTable'), \
            patch('research_arcade.csv_database.csv_arxiv_papers.CSVArxivPapers'), \
            patch('research_arcade.csv_database.csv_arxiv_paragraphs.CSVArxivParagraphs'), \
            patch('research_arcade.csv_database.csv_arxiv_sections.CSVArxivSections'), \
            patch('research_arcade.csv_database.csv_arxiv_paper_authors.CSVArxivPaperAuthor'), \
            patch('research_arcade.csv_database.csv_arxiv_paper_categories.CSVArxivPaperCategory'), \
            patch('research_arcade.csv_database.csv_arxiv_paper_figures.CSVArxivPaperFigure'), \
            patch('research_arcade.csv_database.csv_arxiv_paper_tables.CSVArxivPaperTable'), \
            patch('research_arcade.csv_database.csv_arxiv_paragraph_references.CSVArxivParagraphReference'):
            self.arcade = ResearchArcade(db_type=self.db_type, config=self.config)

    # -------------------------
    # insert_node tests
    # -------------------------
    def test_insert_node_authors(self):
        """Test inserting an author node"""
        node_features = {'author_id': '1', 'name': 'John Doe'}
        self.arcade.arxiv_authors.insert_author = Mock(return_value='author_1')
        
        result = self.arcade.insert_node('arxiv_authors', node_features)
        
        self.arcade.arxiv_authors.insert_author.assert_called_once_with(**node_features)
        self.assertEqual(result, 'author_1')
    
    def test_insert_node_categories(self):
        """Test inserting a category node"""
        node_features = {'category_id': 'cs.AI', 'description': 'Artificial Intelligence'}
        self.arcade.arxiv_categories.insert_category = Mock(return_value='cat_1')
        
        result = self.arcade.insert_node('arxiv_categories', node_features)
        
        self.arcade.arxiv_categories.insert_category.assert_called_once_with(**node_features)
        self.assertEqual(result, 'cat_1')

    def test_insert_node_figures(self):
        """Test inserting a figure node"""
        node_features = {'figure_id': 'fig1', 'caption': 'Test figure'}
        self.arcade.arxiv_figures.insert_figure = Mock(return_value='fig_1')
        
        result = self.arcade.insert_node('arxiv_figures', node_features)
        
        self.arcade.arxiv_figures.insert_figure.assert_called_once_with(**node_features)
        self.assertEqual(result, 'fig_1')
    
    def test_insert_node_tables(self):
        """Test inserting a table node"""
        node_features = {'table_id': 'tab1', 'caption': 'Test table'}
        self.arcade.arxiv_tables.insert_table = Mock(return_value='tab_1')
        
        result = self.arcade.insert_node('arxiv_tables', node_features)
        
        self.arcade.arxiv_tables.insert_table.assert_called_once_with(**node_features)
        self.assertEqual(result, 'tab_1')
    
    def test_insert_node_papers(self):
        """Test inserting a paper node"""
        node_features = {'paper_id': 'arxiv123', 'title': 'Test Paper'}
        self.arcade.arxiv_papers.insert_paper = Mock(return_value='paper_1')
        
        result = self.arcade.insert_node('arxiv_papers', node_features)
        
        self.arcade.arxiv_papers.insert_paper.assert_called_once_with(**node_features)
        self.assertEqual(result, 'paper_1')
    
    def test_insert_node_paragraphs(self):
        """Test inserting a paragraph node"""
        node_features = {'paragraph_id': 'para1', 'text': 'Sample text'}
        self.arcade.arxiv_paragraphs.insert_paragraph = Mock(return_value='para_1')
        
        result = self.arcade.insert_node('arxiv_paragraphs', node_features)
        
        self.arcade.arxiv_paragraphs.insert_paragraph.assert_called_once_with(**node_features)
        self.assertEqual(result, 'para_1')
    
    def test_insert_node_sections(self):
        """Test inserting a section node"""
        node_features = {'section_id': 'sec1', 'title': 'Introduction'}
        self.arcade.arxiv_sections.insert_section = Mock(return_value='sec_1')
        
        result = self.arcade.insert_node('arxiv_sections', node_features)
        
        self.arcade.arxiv_sections.insert_section.assert_called_once_with(**node_features)
        self.assertEqual(result, 'sec_1')
    
    def test_insert_node_invalid_table(self):
        """Test inserting node with invalid table name"""
        node_features = {'id': '1'}
        result = self.arcade.insert_node('invalid_table', node_features)
        
        self.assertIsNone(result)
    
    # -------------------------
    # delete_node_by_id tests
    # -------------------------
    def test_delete_node_authors(self):
        """Test deleting an author node"""
        primary_key = {'author_id': '1'}
        self.arcade.arxiv_authors.delete_author_by_id = Mock(return_value=True)
        
        result = self.arcade.delete_node_by_id('arxiv_authors', primary_key)
        
        self.arcade.arxiv_authors.delete_author_by_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_node_categories(self):
        """Test deleting a category node"""
        primary_key = {'category_id': 'cs.AI'}
        self.arcade.arxiv_categories.delete_category_by_id = Mock(return_value=True)
        
        result = self.arcade.delete_node_by_id('arxiv_categories', primary_key)
        
        self.arcade.arxiv_categories.delete_category_by_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_node_figures(self):
        """Test deleting a figure node"""
        primary_key = {'figure_id': 'fig1'}
        self.arcade.arxiv_figures.delete_figure_by_id = Mock(return_value=True)
        
        result = self.arcade.delete_node_by_id('arxiv_figures', primary_key)
        
        self.arcade.arxiv_figures.delete_figure_by_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_node_tables(self):
        """Test deleting a table node"""
        primary_key = {'table_id': 'tab1'}
        self.arcade.arxiv_tables.delete_table_by_id = Mock(return_value=True)
        
        result = self.arcade.delete_node_by_id('arxiv_tables', primary_key)
        
        self.arcade.arxiv_tables.delete_table_by_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_node_papers(self):
        """Test deleting a paper node"""
        primary_key = {'paper_id': 'arxiv123'}
        self.arcade.arxiv_papers.delete_paper_by_id = Mock(return_value=True)
        
        result = self.arcade.delete_node_by_id('arxiv_papers', primary_key)
        
        self.arcade.arxiv_papers.delete_paper_by_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_node_paragraphs(self):
        """Test deleting a paragraph node"""
        primary_key = {'paragraph_id': 'para1'}
        self.arcade.arxiv_paragraphs.delete_paragraph_by_id = Mock(return_value=True)
        
        result = self.arcade.delete_node_by_id('arxiv_paragraphs', primary_key)
        
        self.arcade.arxiv_paragraphs.delete_paragraph_by_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_node_sections(self):
        """Test deleting a section node"""
        primary_key = {'section_id': 'sec1'}
        self.arcade.arxiv_sections.delete_section_by_id = Mock(return_value=True)
        
        result = self.arcade.delete_node_by_id('arxiv_sections', primary_key)
        
        self.arcade.arxiv_sections.delete_section_by_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_node_invalid_table(self):
        """Test deleting node with invalid table name"""
        primary_key = {'id': '1'}
        result = self.arcade.delete_node_by_id('invalid_table', primary_key)
        
        self.assertIsNone(result)
    
    # -------------------------
    # update_node tests
    # -------------------------
    def test_update_node_authors(self):
        """Test updating an author node"""
        node_features = {'author_id': '1', 'name': 'Jane Doe'}
        self.arcade.arxiv_authors.update_author = Mock(return_value=True)
        
        result = self.arcade.update_node('arxiv_authors', node_features)
        
        self.arcade.arxiv_authors.update_author.assert_called_once_with(**node_features)
        self.assertTrue(result)
    
    def test_update_node_categories(self):
        """Test updating a category node"""
        node_features = {'category_id': 'cs.AI', 'description': 'Updated AI'}
        self.arcade.arxiv_categories.update_category = Mock(return_value=True)
        
        result = self.arcade.update_node('arxiv_categories', node_features)
        
        self.arcade.arxiv_categories.update_category.assert_called_once_with(**node_features)
        self.assertTrue(result)
    
    def test_update_node_figures(self):
        """Test updating a figure node"""
        node_features = {'figure_id': 'fig1', 'caption': 'Updated caption'}
        self.arcade.arxiv_figures.update_figure = Mock(return_value=True)
        
        result = self.arcade.update_node('arxiv_figures', node_features)
        
        self.arcade.arxiv_figures.update_figure.assert_called_once_with(**node_features)
        self.assertTrue(result)
    
    def test_update_node_tables(self):
        """Test updating a table node"""
        node_features = {'table_id': 'tab1', 'caption': 'Updated table'}
        self.arcade.arxiv_tables.update_table = Mock(return_value=True)
        
        result = self.arcade.update_node('arxiv_tables', node_features)
        
        self.arcade.arxiv_tables.update_table.assert_called_once_with(**node_features)
        self.assertTrue(result)
    
    def test_update_node_papers(self):
        """Test updating a paper node"""
        node_features = {'paper_id': 'arxiv123', 'title': 'Updated Title'}
        self.arcade.arxiv_papers.update_paper = Mock(return_value=True)
        
        result = self.arcade.update_node('arxiv_papers', node_features)
        
        self.arcade.arxiv_papers.update_paper.assert_called_once_with(**node_features)
        self.assertTrue(result)
    
    def test_update_node_paragraphs(self):
        """Test updating a paragraph node"""
        node_features = {'paragraph_id': 'para1', 'text': 'Updated text'}
        self.arcade.arxiv_paragraphs.update_paragraph = Mock(return_value=True)
        
        result = self.arcade.update_node('arxiv_paragraphs', node_features)
        
        self.arcade.arxiv_paragraphs.update_paragraph.assert_called_once_with(**node_features)
        self.assertTrue(result)
    
    def test_update_node_sections(self):
        """Test updating a section node"""
        node_features = {'section_id': 'sec1', 'title': 'Updated Section'}
        self.arcade.arxiv_sections.update_section = Mock(return_value=True)
        
        result = self.arcade.update_node('arxiv_sections', node_features)
        
        self.arcade.arxiv_sections.update_section.assert_called_once_with(**node_features)
        self.assertTrue(result)
    
    def test_update_node_invalid_table(self):
        """Test updating node with invalid table name"""
        node_features = {'id': '1'}
        result = self.arcade.update_node('invalid_table', node_features)
        
        self.assertIsNone(result)
    
    # -------------------------
    # get_node_features_by_id tests
    # -------------------------
    def test_get_node_features_authors(self):
        """Test getting author features by ID"""
        primary_key = {'author_id': '1'}
        expected = {'author_id': '1', 'name': 'John Doe'}
        self.arcade.arxiv_authors.get_author_by_id = Mock(return_value=expected)
        
        result = self.arcade.get_node_features_by_id('arxiv_authors', primary_key)
        
        self.arcade.arxiv_authors.get_author_by_id.assert_called_once_with(**primary_key)
        self.assertEqual(result, expected)

    def test_get_node_features_categories(self):
        """Test getting category features by ID"""
        primary_key = {'category_id': 'cs.AI'}
        expected = {'category_id': 'cs.AI', 'description': 'AI'}
        self.arcade.arxiv_categories.get_category_by_id = Mock(return_value=expected)
        
        result = self.arcade.get_node_features_by_id('arxiv_categories', primary_key)
        
        self.arcade.arxiv_categories.get_category_by_id.assert_called_once_with(**primary_key)
        self.assertEqual(result, expected)

    def test_get_node_features_figures(self):
        """Test getting figure features by ID"""
        primary_key = {'figure_id': 'fig1'}
        expected = {'figure_id': 'fig1', 'caption': 'Test'}
        self.arcade.arxiv_figures.get_figure_by_id = Mock(return_value=expected)
        
        result = self.arcade.get_node_features_by_id('arxiv_figures', primary_key)
        
        self.arcade.arxiv_figures.get_figure_by_id.assert_called_once_with(**primary_key)
        self.assertEqual(result, expected)
    
    def test_get_node_features_tables(self):
        """Test getting table features by ID"""
        primary_key = {'table_id': 'tab1'}
        expected = {'table_id': 'tab1', 'caption': 'Test'}
        self.arcade.arxiv_tables.get_table_by_id = Mock(return_value=expected)
        
        result = self.arcade.get_node_features_by_id('arxiv_tables', primary_key)
        
        self.arcade.arxiv_tables.get_table_by_id.assert_called_once_with(**primary_key)
        self.assertEqual(result, expected)
    
    def test_get_node_features_papers(self):
        """Test getting paper features by ID"""
        primary_key = {'paper_id': 'arxiv123'}
        expected = {'paper_id': 'arxiv123', 'title': 'Test'}
        self.arcade.arxiv_papers.get_paper_by_id = Mock(return_value=expected)
        
        result = self.arcade.get_node_features_by_id('arxiv_papers', primary_key)
        
        self.arcade.arxiv_papers.get_paper_by_id.assert_called_once_with(**primary_key)
        self.assertEqual(result, expected)
    
    def test_get_node_features_paragraphs(self):
        """Test getting paragraph features by ID"""
        primary_key = {'paragraph_id': 'para1'}
        expected = {'paragraph_id': 'para1', 'text': 'Test'}
        self.arcade.arxiv_paragraphs.get_paragraph_by_id = Mock(return_value=expected)
        
        result = self.arcade.get_node_features_by_id('arxiv_paragraphs', primary_key)
        
        self.arcade.arxiv_paragraphs.get_paragraph_by_id.assert_called_once_with(**primary_key)
        self.assertEqual(result, expected)
    
    def test_get_node_features_sections(self):
        """Test getting section features by ID"""
        primary_key = {'section_id': 'sec1'}
        expected = {'section_id': 'sec1', 'title': 'Intro'}
        self.arcade.arxiv_sections.get_section_by_id = Mock(return_value=expected)
        
        result = self.arcade.get_node_features_by_id('arxiv_sections', primary_key)
        
        self.arcade.arxiv_sections.get_section_by_id.assert_called_once_with(**primary_key)
        self.assertEqual(result, expected)
    
    def test_get_node_features_invalid_table(self):
        """Test getting features with invalid table name"""
        primary_key = {'id': '1'}
        result = self.arcade.get_node_features_by_id('invalid_table', primary_key)
        
        self.assertIsNone(result)
    
    # -------------------------
    # get_all_node_features tests
    # -------------------------
    def test_get_all_node_features_authors(self):
        """Test getting all author features"""
        expected = [{'author_id': '1'}, {'author_id': '2'}]
        self.arcade.arxiv_authors.get_all_authors = Mock(return_value=expected)
        
        result = self.arcade.get_all_node_features('arxiv_authors')
        
        self.arcade.arxiv_authors.get_all_authors.assert_called_once_with(is_all_features=True)
        self.assertEqual(result, expected)
    
    def test_get_all_node_features_categories(self):
        """Test getting all category features"""
        expected = [{'category_id': 'cs.AI'}, {'category_id': 'cs.ML'}]
        self.arcade.arxiv_categories.get_all_categories = Mock(return_value=expected)
        
        result = self.arcade.get_all_node_features('arxiv_categories')
        
        self.arcade.arxiv_categories.get_all_categories.assert_called_once_with(is_all_features=True)
        self.assertEqual(result, expected)
    
    def test_get_all_node_features_figures(self):
        """Test getting all figure features"""
        expected = [{'figure_id': 'fig1'}, {'figure_id': 'fig2'}]
        self.arcade.arxiv_figures.get_all_figures = Mock(return_value=expected)
        
        result = self.arcade.get_all_node_features('arxiv_figures')
        
        self.arcade.arxiv_figures.get_all_figures.assert_called_once_with(is_all_features=True)
        self.assertEqual(result, expected)
    
    def test_get_all_node_features_tables(self):
        """Test getting all table features"""
        expected = [{'table_id': 'tab1'}, {'table_id': 'tab2'}]
        self.arcade.arxiv_tables.get_all_tables = Mock(return_value=expected)
        
        result = self.arcade.get_all_node_features('arxiv_tables')
        
        self.arcade.arxiv_tables.get_all_tables.assert_called_once_with(is_all_features=True)
        self.assertEqual(result, expected)
    
    def test_get_all_node_features_papers(self):
        """Test getting all paper features"""
        expected = [{'paper_id': 'p1'}, {'paper_id': 'p2'}]
        self.arcade.arxiv_papers.get_all_papers = Mock(return_value=expected)
        
        result = self.arcade.get_all_node_features('arxiv_papers')
        
        self.arcade.arxiv_papers.get_all_papers.assert_called_once_with(is_all_features=True)
        self.assertEqual(result, expected)
    
    def test_get_all_node_features_paragraphs(self):
        """Test getting all paragraph features"""
        expected = [{'paragraph_id': 'para1'}, {'paragraph_id': 'para2'}]
        self.arcade.arxiv_paragraphs.get_all_paragraphs = Mock(return_value=expected)
        
        result = self.arcade.get_all_node_features('arxiv_paragraphs')
        
        self.arcade.arxiv_paragraphs.get_all_paragraphs.assert_called_once_with(is_all_features=True)
        self.assertEqual(result, expected)
    
    def test_get_all_node_features_sections(self):
        """Test getting all section features"""
        expected = [{'section_id': 'sec1'}, {'section_id': 'sec2'}]
        self.arcade.arxiv_sections.get_all_sections = Mock(return_value=expected)
        
        result = self.arcade.get_all_node_features('arxiv_sections')
        
        self.arcade.arxiv_sections.get_all_sections.assert_called_once_with(is_all_features=True)
        self.assertEqual(result, expected)
    
    def test_get_all_node_features_invalid_table(self):
        """Test getting all features with invalid table name"""
        result = self.arcade.get_all_node_features('invalid_table')
        
        self.assertIsNone(result)


class TestArxivArcadeEdgeCRUD(unittest.TestCase):
    """Test suite for Edge CRUD operations"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.config = {'test_config': 'value'}
        
        with patch('arxiv_arcade.CSVArxivAuthors'), \
             patch('arxiv_arcade.CSVArxivCategory'), \
             patch('arxiv_arcade.CSVArxivFigure'), \
             patch('arxiv_arcade.CSVArxivTable'), \
             patch('arxiv_arcade.CSVArxivPapers'), \
             patch('arxiv_arcade.CSVArxivParagraphs'), \
             patch('arxiv_arcade.CSVArxivSections'), \
             patch('arxiv_arcade.CSVArxivCitation'), \
             patch('arxiv_arcade.CSVArxivPaperAuthor'), \
             patch('arxiv_arcade.CSVArxivPaperCategory'), \
             patch('arxiv_arcade.CSVArxivPaperFigure'), \
             patch('arxiv_arcade.CSVArxivPaperTable'), \
             patch('arxiv_arcade.CSVArxivParagraphReference'):
            self.arcade = ResearchArcade(self.config)
    
    # -------------------------
    # insert_edge tests
    # -------------------------
    def test_insert_edge_citation(self):
        """Test inserting a citation edge"""
        edge_features = {'citing_paper_id': 'p1', 'cited_paper_id': 'p2'}
        self.arcade.arxiv_citation.insert_citation = Mock(return_value=True)
        
        result = self.arcade.insert_edge('arxiv_citation', edge_features)
        
        self.arcade.arxiv_citation.insert_citation.assert_called_once_with(**edge_features)
        self.assertTrue(result)
    
    def test_insert_edge_paper_author(self):
        """Test inserting a paper-author edge"""
        edge_features = {'paper_id': 'p1', 'author_id': 'a1'}
        self.arcade.arxiv_paper_author.insert_paper_author = Mock(return_value=True)
        
        result = self.arcade.insert_edge('arxiv_paper_author', edge_features)
        
        self.arcade.arxiv_paper_author.insert_paper_author.assert_called_once_with(**edge_features)
        self.assertTrue(result)
    
    def test_insert_edge_paper_category(self):
        """Test inserting a paper-category edge"""
        edge_features = {'paper_id': 'p1', 'category_id': 'cs.AI'}
        self.arcade.arxiv_paper_category.insert_paper_category = Mock(return_value=True)
        
        result = self.arcade.insert_edge('arxiv_paper_category', edge_features)
        
        self.arcade.arxiv_paper_category.insert_paper_category.assert_called_once_with(**edge_features)
        self.assertTrue(result)
    
    def test_insert_edge_paper_figure(self):
        """Test inserting a paper-figure edge"""
        edge_features = {'paper_id': 'p1', 'figure_id': 'fig1'}
        self.arcade.arxiv_paper_figure.insert_paper_figure = Mock(return_value=True)
        
        result = self.arcade.insert_edge('arxiv_paper_figure', edge_features)
        
        self.arcade.arxiv_paper_figure.insert_paper_figure.assert_called_once_with(**edge_features)
        self.assertTrue(result)
    
    def test_insert_edge_paper_table(self):
        """Test inserting a paper-table edge"""
        edge_features = {'paper_id': 'p1', 'table_id': 'tab1'}
        self.arcade.arxiv_paper_table.insert_paper_table = Mock(return_value=True)
        
        result = self.arcade.insert_edge('arxiv_paper_table', edge_features)
        
        self.arcade.arxiv_paper_table.insert_paper_table.assert_called_once_with(**edge_features)
        self.assertTrue(result)
    
    def test_insert_edge_paragraph_reference(self):
        """Test inserting a paragraph-reference edge"""
        edge_features = {'paragraph_id': 'para1', 'reference_id': 'ref1'}
        self.arcade.arxiv_paragraph_reference.insert_paragraph_reference = Mock(return_value=True)
        
        result = self.arcade.insert_edge('arxiv_paragraph_reference', edge_features)
        
        self.arcade.arxiv_paragraph_reference.insert_paragraph_reference.assert_called_once_with(**edge_features)
        self.assertTrue(result)
    
    def test_insert_edge_invalid_table(self):
        """Test inserting edge with invalid table name"""
        edge_features = {'id': '1'}
        result = self.arcade.insert_edge('invalid_table', edge_features)
        
        self.assertIsNone(result)
    
    # -------------------------
    # delete_edge_by_id tests - citation
    # -------------------------
    def test_delete_edge_citation_both_keys(self):
        """Test deleting citation with both citing and cited paper IDs"""
        primary_key = {'citing_paper_id': 'p1', 'cited_paper_id': 'p2'}
        self.arcade.arxiv_citation.delete_citation_by_id = Mock(return_value=True)
        
        result = self.arcade.delete_edge_by_id('arxiv_citation', primary_key)
        
        self.arcade.arxiv_citation.delete_citation_by_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_edge_citation_citing_only(self):
        """Test deleting citation with only citing paper ID"""
        primary_key = {'citing_paper_id': 'p1'}
        self.arcade.arxiv_citation.delete_citation_by_citing_id = Mock(return_value=True)
        
        result = self.arcade.delete_edge_by_id('arxiv_citation', primary_key)
        
        self.arcade.arxiv_citation.delete_citation_by_citing_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_edge_citation_cited_only(self):
        """Test deleting citation with only cited paper ID"""
        primary_key = {'cited_paper_id': 'p2'}
        self.arcade.arxiv_citation.delete_citation_by_cited_id = Mock(return_value=True)
        
        result = self.arcade.delete_edge_by_id('arxiv_citation', primary_key)
        
        self.arcade.arxiv_citation.delete_citation_by_cited_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_edge_citation_no_keys(self):
        """Test deleting citation with no valid keys"""
        primary_key = {'invalid_key': 'value'}
        result = self.arcade.delete_edge_by_id('arxiv_citation', primary_key)
        
        self.assertIsNone(result)
    
    # -------------------------
    # delete_edge_by_id tests - paper_author
    # -------------------------
    def test_delete_edge_paper_author_both_keys(self):
        primary_key = {'paper_id': 'p1', 'author_id': 'a1'}
        self.arcade.arxiv_paper_author.delete_paper_author_by_id = Mock(return_value=True)
        
        result = self.arcade.delete_edge_by_id('arxiv_paper_author', primary_key)
        
        self.arcade.arxiv_paper_author.delete_paper_author_by_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)
    
    def test_delete_edge_paper_author_paper_only(self):
        primary_key = {'paper_id': 'p1'}
        self.arcade.arxiv_paper_author.delete_paper_author_by_paper_id = Mock(return_value=True)
        
        result = self.arcade.delete_edge_by_id('arxiv_paper_author', primary_key)
        
        self.arcade.arxiv_paper_author.delete_paper_author_by_paper_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)

    def test_delete_edge_paper_author_author_only(self):
        """Test deleting paper-author with only author ID"""
        primary_key = {'author_id': 'a1'}
        self.arcade.arxiv_paper_author.delete_paper_author_by_author_id = Mock(return_value=True)
        
        result = self.arcade.delete_edge_by_id('arxiv_paper_author', primary_key)
        
        self.arcade.arxiv_paper_author.delete_paper_author_by_author_id.assert_called_once_with(**primary_key)
        self.assertTrue(result)

    def test_delete_edge_paper_author_invalid_keys(self):
        """Test deleting paper-author with invalid keys"""
        primary_key = {'invalid_key': 'val'}
        result = self.arcade.delete_edge_by_id('arxiv_paper_author', primary_key)
        
        self.assertIsNone(result)


taadc = TestArxivArcadeNodeCRUD()

taadc.setUp()
taadc.test_insert_node_authors()