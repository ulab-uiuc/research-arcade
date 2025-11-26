"""
This program extracts the paragraphs from a paper and store them in the database
"""


from paper_collector.paper_graph_processor import PaperGraphProcessor

class ParagraphProcessor:

    def __init__(self, data_dir, figures_dir, output_dir):

        self.paper_graph_processor = PaperGraphProcessor(data_dir=data_dir, figures_dir=figures_dir, output_dir=output_dir)


    def extract_paragraphs(self):
        
        # First process all the json files
        self.paper_graph_processor.process_all_papers()
        # After that, we extract the corresponding paragraphs and store the paragraphs into database

        # Then we go through all the extracted information
        


    pass

