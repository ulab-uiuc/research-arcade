import arxiv

# Search for the paper by its arXiv ID
search = arxiv.Search(id_list=["2503.12600"])
paper = next(arxiv.Client().results(search))

# Download the PDF and source latex code to the current directory
paper.download_pdf()
paper.download_source()

# Then, how shall I work on the web extraction?