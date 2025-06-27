import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semanticscholar import SemanticScholar

"""
Test searching papers with arxiv id on semantic scholar
If this does not work well, we might need alternativ approaches

The result shows that we can only search paper authors with base arxiv id, instead of the full one with version (v1, v2, v3...)
"""

arxiv_id1 = "2501.02725v3"
arxiv_id2 = "2501.02725"

sch = SemanticScholar()
paper_sch = sch.get_paper(f"ARXIV:{arxiv_id2}")
authors = paper_sch.authors
print(authors)
