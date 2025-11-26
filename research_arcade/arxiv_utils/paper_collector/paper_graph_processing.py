from paper_graph_processor import PaperGraphProcessor


data_dir_path = ""
figures_dir_path = ""
output_dir_path = ""

# It seems that nothing is extracted. Why?
# process_all_papers reads json files only. I need to first convert the papers into json format from latex code. How can I do that?

pgp = PaperGraphProcessor(data_dir=data_dir_path, figures_dir=figures_dir_path, output_dir=output_dir_path)

pgp.process_all_papers()