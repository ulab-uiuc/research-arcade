import re
from typing import List, Tuple
import hashlib

def arxiv_id_processor(arxiv_id):
    """
    If the arxiv id is in the format a.bvc, we extract the a.b and c
    If it is in the format a.b, we only extract the a.b
    """
    ARXIV_RE = re.compile(r'^(?P<id>(?:[a-z\-]+\/\d{7}|\d{4}\.\d{4,5}))(?:v(?P<v>\d+))?$', re.IGNORECASE)

    m = ARXIV_RE.match(arxiv_id.strip())
    if not m:
        raise ValueError(f"Not a valid arXiv id: {arxiv_id!r}")
    base = m.group('id')
    v = m.group('v')
    return base, (int(v) if v is not None else None)

def figure_iteration_recursive(figure_json):

    # Create a set of figures along with the
    # list represents (path, caption, label)
    path_to_info: List[Tuple[str, str, str]] = []

    # First iterate through parent, then go into the children

    def figure_iteration(figure_json):
        nonlocal path_to_info

        if not figure_json:
            return
        if figure_json['figure_paths']:
            path = figure_json['figure_paths'][0]
            caption = figure_json['caption']
            label = figure_json['label']
            path_to_info.append((path, caption, label))
        subfigures = figure_json['subfigures']
        
        for subfigure in subfigures:
            figure_iteration(subfigure)
    
    figure_iteration(figure_json=figure_json)
    return path_to_info
def get_paragraph_num(pid):
    pattern = re.compile(r'^text_(\d+)$')
    m = pattern.match(pid)
    if not m:
        raise ValueError(f"Bad paragraph id format: {pid!r}")
    return int(m.group(1))

def arxiv_ids_hashing(arxiv_ids):
    # convert a list of arxiv ids into a unique hashing
    # arxiv ids up to permutation: for the same set of arxiv ids, we produce the same result after hashing, regardless how we order them in the list

    arxiv_ids_set = set(arxiv_ids)
    sorted_ids = sorted(arxiv_ids_set)
    combined = "|".join(sorted_ids)
    return hashlib.sha256(combined.encode()).hexdigest()