"""
Try the PaperGraphProcessor functionalities
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_constructor.paragraph_processor import ParagraphProcessor

from graph_constructor.node_processor import NodeConstructor


data_dir = "download/output/endpoints"
figures_dir = "download/output/figures"
output_dir = "download/output/paragraphs"
# figures_dir, output_dir
pp = ParagraphProcessor(data_dir=data_dir, figures_dir=figures_dir, output_dir=output_dir)

pp.extract_paragraphs()

nc = NodeConstructor()

nc.create_tables()

nc.process_paragraphs("download")