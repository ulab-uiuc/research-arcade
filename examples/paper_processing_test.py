import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_constructor.node_processor import NodeConstructor
from multi_input.multi_download import MultiDownload

nc = NodeConstructor()
md = MultiDownload()

# First download the paper
# Why no citation?
arxiv_id = "2507.02822v1"
dir_path = "download"

# md.download_arxiv(input=arxiv_id, input_type="id", output_type="both", dest_dir=dir_path)
md.build_paper_graph(input=arxiv_id, input_type="id", dest_dir=dir_path)

# nc.drop_tables()

# At least I need to first download the paper
# nc.create_tables()
nc.process_paper(arxiv_id=arxiv_id, dir_path=dir_path)

