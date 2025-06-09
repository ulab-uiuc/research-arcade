import requests

from multi_input import multi_input
import arxiv

# # Search for the paper by its arXiv ID
# search = arxiv.Search(id_list=["2503.12600"])
# paper = next(arxiv.Client().results(search))

# # Download the PDF and source latex code to the current directory
# paper.download_pdf()
# paper.download_source()

class multi_download:
    """
    This class supports downloading arxiv latex code, pdf and html webpage using arxiv id, link or bib
    """

    
    def download_arxiv(self, input: str, input_type: str, output_type: str, dest_dir: str = None):
        
        mi = multi_input()
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

        if output_type == "pdf":
            paper.download_pdf()

        if output_type == "latex":
            paper.download_source(dirpath = dest_dir)
        
        if output_type == "both":
            paper.download_source(dirpath = dest_dir)
            paper.download_pdf(dirpath = dest_dir)




id_string = "2112.10911"
dest_path = "./download"

mo = multi_download()

mo.download_arxiv(id_string, "id", "both", dest_path)

