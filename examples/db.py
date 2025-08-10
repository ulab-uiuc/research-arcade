"""
Insert None into INT entry
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_constructor.database import Database

db = Database()
arxiv_id = '2412.17767'

db.insert_paper(arxiv_id=arxiv_id, base_arxiv_id=arxiv_id, version=None, title="ResearchTown")