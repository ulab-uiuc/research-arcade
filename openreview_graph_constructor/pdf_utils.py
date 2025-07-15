import re
import difflib
import requests
from tqdm import tqdm
from typing import List, Optional
from pathlib import Path
from pdfminer.high_level import extract_text

# get pdf based on openreview_id
def get_pdf(id, pdf_name):
    # pdf url
    pdf_url = "https://openreview.net/notes/edits/attachment?id="+id+"&name=pdf"
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(pdf_url, headers=headers)
    if response.status_code == 200:
        with open(pdf_name, "wb") as f:
            f.write(response.content)
        print("✅ PDF is downloaded as "+pdf_name)
    else:
        print("❌ Failure, Status Code: ", response.status_code)

# extract text from pdf
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# compare the differences between two pdfs
def compare_texts(text1, text2):
    diff = difflib.unified_diff(
        text1.splitlines(),
        text2.splitlines(),
        fromfile='Original',
        tofile='Modified',
        lineterm=''
    )
    return '\n'.join(diff)

# format the differences
def parse_diff(diff_text):
    lines = diff_text.splitlines()
    
    all_diff = []
    current_diff = None
    for line in tqdm(lines[2:]):
        # Check for diff change markers
        if line.startswith('@@'):
            if current_diff is not None:
                # Add the previous diff to the corresponding list
                all_diff.append(current_diff)
            # Start a new diff block
            current_diff = {
                'context_before': "",
                'context_after': "",
                'original_lines': "",
                'modified_lines': "",
            }
        elif line.startswith('-'):
            current_diff['original_lines'] = current_diff['original_lines'] + line[1:].strip() + " "
        elif line.startswith('+'):
            current_diff['modified_lines'] = current_diff['modified_lines'] + line[1:].strip() + " "
        elif line.strip() != "" and (current_diff['original_lines'] == "" and current_diff['modified_lines'] == ""):
            current_diff['context_before'] = current_diff['context_before'] + line.strip() + " "
        elif line.strip() != "" and (current_diff['original_lines'] != "" or current_diff['modified_lines'] != ""):
            current_diff['context_after'] = current_diff['context_after'] + line.strip() + " "
            
    print("successfully build current_diff")
    
    return all_diff

# more than 3 math-related symbols and less than 10 characters
def check_str_regex(s: str) -> bool: 
    math_symbol_pattern = r'[0-9+\-*/=]'
    math_count = len(re.findall(math_symbol_pattern, s))
    letter_count = len(re.findall(r'[A-Za-z]', s))
    return (math_count >= 3) and (letter_count < 10)

# primarily format lines into paragraphs
def preprocess_lines_in_paragraphs(lines: list) -> list:
    formatted_lines = []
    buffer = []
    for line in lines:
        if line.strip(): # Non-empty -> the same paragraph
            if line[-1] == '-':
                buffer.append(line[:-1])
            else:
                buffer.append(line)
        else: # Empty -> Next paragraph
            if buffer: # Combine the content in buffer
                formatted_lines.append("".join(buffer))
                buffer = [] # Clean buffer
    if buffer:
        formatted_lines.append("".join(buffer))
        
    return formatted_lines

# finally format pdf into paragraphs
def extract_paragraphs_from_pdf(pdf_path: Path, filter_list: Optional[List[str]] = None):
    # extract all the text from pdf
    full_text = extract_text(pdf_path)
    
    # split the text into lines
    lines = full_text.splitlines()
    
    # construct paragraphs based on the empty lines
    formatted_lines = preprocess_lines_in_paragraphs(lines)
    
    # only extract paragraphs between abstract and appendix
    start = 0
    try:
        start = formatted_lines.index("Abstract")
    except:
        try:
            start = formatted_lines.index("ABSTRACT")
        except:
            print("can not find abstract")
    end = len(formatted_lines)
    try:
        end = formatted_lines.index("Appendix")
    except:
        try:
            end = formatted_lines.index("APPENDIX")
        except:
            print("can not find appendix")

    # the structured content and insert the title
    structured_content = {
        "Title": [formatted_lines[1]],
    }
    
    # start constructing the structured content
    before_context = ""
    current_section_idx = 0
    current_subsection_idx = 0
    # before_section = ""
    current_section = ""
    current_image_table = []
    is_chapter = False
    num_paragraph = 0
    num_image_table = 0
    for line in formatted_lines[start:end]:
        if check_str_regex(line): # no more than 3 digits or at least 10 characters
            continue
        
        # check if before_context contains invalid content
        is_filter = False
        if filter_list is not None:
            for text in filter_list:
                if text in before_context:
                    is_filter = True
                    break
        
        # before_context add into formatted context
        if is_filter:
            pass
        elif is_chapter:
            num_image_table = 0
            num_paragraph = -1
            is_chapter = False
        elif before_context.startswith("Figure") or before_context.startswith("Table"):
            current_image_table.append(before_context)
        elif before_context != "":
            if not before_context.isdigit() and before_context != "": # get rid of pure digit
                if len(structured_content[current_section]) == 0:
                    num_image_table = 0
                    num_paragraph += 1
                    structured_content[current_section].append(before_context)
                else:
                    char_end = structured_content[current_section][num_paragraph-num_image_table][-1]
                    
                    is_append = True
                    if char_end != ".":
                        is_append = False
                    elif char_end == "." and (before_context[0].isdigit() and before_context[1] == "."): # 1. 2.
                        is_append = False
                    elif char_end == "." and (before_context[0] == "•"): # •
                        is_append = False
                    elif char_end == "." and (before_context[0].isdigit() and before_context[1] == ")"): # 1) 2)
                        is_append = False
                    elif char_end == "." and before_context[0] == "(": # (1), (information)
                        is_append = False

                    if is_append:
                        num_image_table = 0
                        num_paragraph += 1
                        structured_content[current_section].append(before_context)
                        if len(current_image_table) != 0:
                            num_image_table = len(current_image_table)
                            structured_content[current_section].extend(current_image_table)
                            current_image_table = []
                    else:
                        structured_content[current_section][num_paragraph-num_image_table] = structured_content[current_section][num_paragraph-num_image_table] + " " + before_context
                
        # abstract
        if line == "Abstract" or line == "ABSTRACT":
            is_chapter = True
            current_section = "Abstract"
            structured_content[current_section] = []
        # reference
        if line == "REFERENCES" or line == "References":
            is_chapter = True
            # before_section = current_section
            current_section = "References"
            structured_content[current_section] = []
        # chapter
        if before_context.isdigit() and len(line) <= 20:
            is_chapter = True
            # before_section = current_section
            current_section = before_context+" "+line
            structured_content[current_section] = []
            current_section_idx = current_section_idx + 1
            current_subsection_idx = 0
        if not line.isdigit() and line[0].isdigit() and line[1] == " ":
            is_chapter = True
            # before_section = current_section
            current_section = line
            structured_content[current_section] = []
            current_section_idx = current_section_idx + 1
            current_subsection_idx = 0
        # sub-chapter
        if not line.isdigit() and line[0] == str(current_section_idx) and line[1] == "." and line[2] == str(current_subsection_idx+1):
            is_chapter = True
            # before_section = current_section
            current_section = line
            structured_content[current_section] = []
            current_subsection_idx = current_subsection_idx + 1

        before_context = line
    
    return structured_content

# connect the differences with the paragraphs
def connect_diffs_and_paragraphs(original_pdf_path: Path, modified_pdf_path: Path, filter_list: Optional[List[str]] = None):
    # extract text
    original_text = extract_text_from_pdf(original_pdf_path)
    print("Successfully extract text from the original pdf")
    modified_text = extract_text_from_pdf(modified_pdf_path)
    print("Successfully extract text from the modified pdf")

    # get the differences
    all_diff_result = compare_texts(original_text, modified_text)
    print("Successfully extract differences between original pdf and modified pdf")
    
    # get formatted differences
    formatted_diff_result = parse_diff(all_diff_result)
    print("Successfully get formatted differences")
    
    # get the structured paragraphs from modified paper
    structured_paragraphs_from_modified = extract_paragraphs_from_pdf(modified_pdf_path, filter_list)
    print("Successfully extract paragraphs from the modified pdf")
    
    # connect differences with paragraphs
    all_diff_loc = []
    for diff in formatted_diff_result:
        diff_before_context = diff["context_before"]
        diff_after_context = diff["context_after"]
        idx = 0
        before_list = []
        after_list = []
        for key, val in zip(structured_paragraphs_from_modified.keys(), structured_paragraphs_from_modified.values()):
            for paragraph in val:
                idx = idx + 1
                if diff_before_context[:10] in paragraph:
                    diff_sample = {}
                    diff_sample["context_before"] = diff["context_before"]
                    diff_sample["context_after"] = diff["context_after"]
                    diff_sample["original_lines"] = diff["original_lines"]
                    diff_sample["modified_lines"] = diff["modified_lines"]
                    diff_sample["section"] = key
                    diff_sample["paragraph_idx"] = idx
                    before_list.append(diff_sample)
                if diff_after_context[:10] in paragraph:
                    diff_sample = {}
                    diff_sample["context_before"] = diff["context_before"]
                    diff_sample["context_after"] = diff["context_after"]
                    diff_sample["original_lines"] = diff["original_lines"]
                    diff_sample["modified_lines"] = diff["modified_lines"]
                    diff_sample["section"] = key
                    diff_sample["paragraph_idx"] = idx
                    after_list.append(diff_sample)
        if len(before_list) == 1:
            all_diff_loc.append(before_list[0])
        elif len(after_list) == 1:
            all_diff_loc.append(after_list[0])
        elif len(before_list) > 1 and len(after_list) > 1:
            is_find = False
            for before_dict in before_list:
                for after_dict in after_list:
                    if before_dict["paragraph_idx"] == after_dict["paragraph_idx"]:
                        all_diff_loc.append(before_dict)
                        is_find = True
                        break
                if is_find:
                    break    
                    
    print("Successfully connect differences with paragraphs")
                    
    return all_diff_loc