import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from graph_constructor.node_processor import NodeConstructor

nc = NodeConstructor()

arxiv_id1 = "2508.00224"
arxiv_id2 = "2508.00224v1"


arxiv_id = nc.arxiv_id_processor(arxiv_id=arxiv_id1)
print(arxiv_id)

arxiv_id = nc.arxiv_id_processor(arxiv_id=arxiv_id2)
print(arxiv_id)
