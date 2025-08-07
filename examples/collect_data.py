from paper_collector.graph_construction import build_citation_graph_thread
from paper_collector.utils import None_constraint


arxiv_list = ["2501.02725v2"]
build_citation_graph_thread(
    arxiv_list,
    "arxiv_papers_with_source",
    "arxiv_papers_with_source/working_folder",
    "arxiv_papers_with_source/output",
    None,
    None_constraint,
    len(arxiv_list),
    1000,
    True,
    len(arxiv_list),
)
