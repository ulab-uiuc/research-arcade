import os
import re

def figure_latex_path_to_path(path, arxiv_id, latex_path):
    # 2. Replace any forward or backward slash with underscore
    latex_path = re.sub(r'[/]', '_', latex_path)
    # 3. Build the filename with the arXiv ID prefix
    return f"{path}/output/figures/{arxiv_id}/{latex_path}"

