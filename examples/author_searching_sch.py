"""
This program records the time needed to search a paper on semantic scholar and find its authors given the arxiv id.

It seems that, in the node processor method, searching authors takes a long time, even longer than searching arxiv id given the titles.

Using an API solves the issue.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from graph_constructor.node_processor import NodeConstructor
import time


NUMBER_OF_LOOPS = 10

arxiv_id = "2501.02725v3"


times = []

nc = NodeConstructor()

for i in range(NUMBER_OF_LOOPS):

    print(f"Start fetching paper, the loop {i}")

    t0 = time.perf_counter()
    # Add the author into paper directory if the paper is on semantic scholar
    base_arxiv_id, version = nc.arxiv_id_processor(arxiv_id)
    try:
        paper_sch = nc.sch.get_paper(f"ARXIV:{base_arxiv_id}")
    except Exception as e:
        print(f"Paper with arxiv id {base_arxiv_id} is not found on semantic scholar: {e}")

    authors = paper_sch.authors

    # Dont really need to test time of adding authors into database since it does not take that much time

    times.append(time.perf_counter() - t0)
    print(f"Time of finding authors and adding authors to database: {times[i]}")

print(times)