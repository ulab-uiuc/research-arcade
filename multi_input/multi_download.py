import requests

import os
from multi_input import MultiInput
import arxiv
import json
from typing import Optional
# # Search for the paper by its arXiv ID
# search = arxiv.Search(id_list=["2503.12600"])
# paper = next(arxiv.Client().results(search))

# # Download the PDF and source latex code to the current directory
# paper.download_pdf()
# paper.download_source()

class MultiDownload:
    """
    This class supports downloading arxiv latex code, pdf and html webpage using arxiv id, link or bib
    """

    def download_arxiv(self, input: str, input_type: str, output_type: str, dest_dir: Optional[str] = None):
        
        mi = MultiInput()
        input_type = input_type.lower()
        arxiv_id = ""
        if input_type == "id" or input_type == "arxiv_id":
            arxiv_id = input

        elif input_type == "bib" or input_type == "arxiv_bib":
            bib_dict = mi.extract_bib_from_string(input)
            arxiv_id = mi.extract_arxiv_id(bib_dict)

        elif input_type == "url" or input_type == "link":
            arxiv_id = mi.arxiv_url_to_id(input)

        else:
            # Raise error for unknown input_type
            raise ValueError(f"Unknown input_type '{input_type}'. "
                            f"Expected one of: 'id', 'arxiv_id', 'bib', 'arxiv_bib', 'url', 'link'.")

        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(arxiv.Client().results(search))


        # Save metadata
        if dest_dir:
            metadata = {
                'id': arxiv_id,
                'title': paper.title,
                'abstract': paper.summary,
                'authors': [a.name for a in paper.authors],
                'published': str(paper.published),
                'categories': paper.categories,
                'url': paper.entry_id,
            }
            os.makedirs(dest_dir, exist_ok=True)
            with open(f"{dest_dir}/{arxiv_id}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        if output_type == "pdf":
            paper.download_pdf()

        if output_type == "latex":
            paper.download_source(dirpath = dest_dir)
        
        if output_type == "both":
            paper.download_source(dirpath = dest_dir)
            paper.download_pdf(dirpath = dest_dir)




id_string = "2112.10911"
dest_path = "./download"

mo = MultiDownload()

mo.download_arxiv(id_string, "id", "both", dest_path)

