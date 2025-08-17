import os
import re

def figure_latex_path_to_path(path, arxiv_id, latex_path):
    # 2. Replace any forward or backward slash with underscore
    latex_path = re.sub(r'[/]', '_', latex_path)
    # 3. Build the filename with the arXiv ID prefix
    return f"{path}/output/figures/{arxiv_id}/{latex_path}"

def figure_label_add_latex_format(name):
    return f"\\label{{{name}}}"

def figure_label_remove_latex_format(name):
    # Reverse method of figure_label_add_latex_format
    """
    Extracts the name from a LaTeX \\label{...} string.
    Example: "\\label{fig1}" -> "fig1"
    """
    match = re.match(r"\\label\{(.+?)\}", name.strip())
    if match:
        return match.group(1)
    return None
