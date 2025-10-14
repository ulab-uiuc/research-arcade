from pdf_utils import extract_paragraphs_from_pdf_new
import json

pdf_path = "original.pdf"
filter_list = ["Under review as a conference paper at ICLR 2025"]
output_file = "original_content.json"

structured_content = extract_paragraphs_from_pdf_new(pdf_path, filter_list)

with open(output_file, "w") as f:
    json.dump(structured_content, f, indent=4)