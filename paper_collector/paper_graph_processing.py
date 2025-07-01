from paper_graph_processor import PaperGraphProcessor, clean_latex_code


data_dir_path = "/Users/chongshan0lin/Documents/Research/UIUC/paper-crawler/arxiv_papers_with_source/2501.01149v2"
figures_dir_path = "/Users/chongshan0lin/Documents/Research/UIUC/paper-crawler/arxiv_papers_with_source/2501.01149v2/images"
output_dir_path = "/Users/chongshan0lin/Documents/Research/UIUC/paper-crawler/arxiv_papers_extracted_json"

# It seems that nothing is extracted. Why?
# process_all_papers reads json files only. I need to first convert the papers into json format from latex code. How can I do that?

pgp = PaperGraphProcessor(data_dir=data_dir_path, figures_dir=figures_dir_path, output_dir=output_dir_path)

pgp.process_all_papers()