import os
import openreview
import requests
from tqdm import tqdm
import time
import re

def get_pdf_by_link(link: str, pdf_name: str, log_file: str):
    pdf_url = "https://openreview.net"+link
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(pdf_url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open(pdf_name, "wb") as f:
                f.write(response.content)
            print(f"✅ PDF downloaded: {pdf_name}")
        else:
            print(f"❌ Download failed ({response.status_code}) for ID: {id}")
            with open(log_file, "a") as log:
                log.write(f"{link}\n")
    except Exception as e:
        print(f"❌ Exception for ID {link}: {e}")
        with open(log_file, "a") as log:
            log.write(f"{link}\n")

def get_pdf_by_id(id: str, pdf_name: str, log_file: str): # for 2025 and 2024
    # pdf url
    pdf_url = "https://openreview.net/notes/edits/attachment?id="+id+"&name=pdf"
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(pdf_url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open(pdf_name, "wb") as f:
                f.write(response.content)
            print(f"✅ PDF downloaded: {pdf_name}")
        else:
            print(f"❌ Download failed ({response.status_code}) for ID: {id}")
            with open(log_file, "a") as log:
                log.write(f"{id}\n")
    except Exception as e:
        print(f"❌ Exception for ID {id}: {e}")
        with open(log_file, "a") as log:
            log.write(f"{id}\n")
            
def get_pdf_by_id_new(id: str, pdf_name: str, log_file: str): # for 2013, 2014, 2017, 2018, 2019, 2020, 2021, 2022, 2023
    # pdf url
    pdf_url = "https://openreview.net/references/pdf?id="+id
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(pdf_url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open(pdf_name, "wb") as f:
                f.write(response.content)
            print(f"✅ PDF downloaded: {pdf_name}")
        else:
            print(f"❌ Download failed ({response.status_code}) for ID: {id}")
            with open(log_file, "a") as log:
                log.write(f"{id}\n")
    except Exception as e:
        print(f"❌ Exception for ID {id}: {e}")
        with open(log_file, "a") as log:
            log.write(f"{id}\n")
            
def get_pdf_from_arxiv(arxiv_id: str, pdf_name: str, log_file: str):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(pdf_url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open(pdf_name, "wb") as f:
                f.write(response.content)
            print(f"✅ PDF downloaded: {pdf_name}")
        else:
            print(f"❌ Download failed ({response.status_code}) for arXiv ID: {arxiv_id}")
            with open(log_file, "a") as log:
                log.write(f"{arxiv_id}\n")
    except Exception as e:
        print(f"❌ Exception for arXiv ID {arxiv_id}: {e}")
        with open(log_file, "a") as log:
            log.write(f"{arxiv_id}\n")

client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        ) # for 2025 and 2024
# client = openreview.Client(baseurl='https://api.openreview.net') # for 2013, 2014, 2017, 2018, 2019, 2020, 2021, 2022, 2023

# parameters
# 2024, 2025
venue_id = 'ICLR.cc/2025/Conference'
pdf_dir = "/data/jingjunx/openreview_pdfs_2025/"
paper_log_file = "not_found_papers_2025.txt"
revision_log_file = "not_found_revisions_2025.txt"
submissions = client.get_all_notes(invitation=f'{venue_id}/-/Submission', details='revisions')

# 2018, 2019, 2020, 2021, 2022, 2023
# venue_id = 'ICLR.cc/2023/Conference'
# pdf_dir = "/data/jingjunx/openreview_pdfs_2023/"
# paper_log_file = "not_found_papers_2023.txt"
# revision_log_file = "not_found_revisions_2023.txt"
# submissions = client.get_all_notes(invitation=f'{venue_id}/-/Blind_Submission', details='revisions')

# 2013, 2014, 2017
# venue_id = 'ICLR.cc/2013/conference'
# pdf_dir = "/data/jingjunx/openreview_pdfs_2013/"
# paper_log_file = "not_found_papers_2013.txt"
# revision_log_file = "not_found_revisions_2013.txt"
# submissions = client.get_all_notes(invitation=f'{venue_id}/-/submission', details='revisions')

os.makedirs(pdf_dir, exist_ok=True)
for submission in tqdm(submissions): 
    # 2025 and 2024
    decision = submission.content["venueid"]["value"].split('/')[-1]
    if decision == "Withdrawn_Submission":
        continue
    else:
        paper_id = submission.id
        link = submission.content["pdf"]["value"]
        pdf_name = str(pdf_dir)+str(paper_id)+".pdf"
        get_pdf_by_link(link, pdf_name, paper_log_file)
        time.sleep(1)
        
        note_edits = client.get_note_edits(note_id=paper_id)
        if len(note_edits) <= 1:
            continue
        else:
            for note in note_edits:
                id = note.id
                pdf_name = str(pdf_dir)+str(id)+".pdf"
                if os.path.isfile(pdf_name):
                    continue
                else:
                    time.sleep(1)
                    get_pdf_by_id(id, pdf_name, revision_log_file)    
    
    
    # 2013, 2014, 2017, 2018, 2019, 2020, 2021, 2022, 2023
    # paper_id = submission.id
    # link = submission.content["pdf"]
    # pdf_name = str(pdf_dir)+str(paper_id)+".pdf"
    # get_pdf_by_link(link, pdf_name, paper_log_file)
    # time.sleep(1)
    
    # revisions = client.get_references(referent=paper_id, original=True)
    # time.sleep(1)
    # pdf_revisions_ids = []
    # for revision in revisions:
    #     if "pdf" in revision.content:
    #         pdf_revisions_ids.append(revision.id)
    
    # if len(pdf_revisions_ids) <= 1:
    #     continue
    # else:
    #     for pdf_revision_id in pdf_revisions_ids:
    #         pdf_name = str(pdf_dir)+str(pdf_revision_id)+".pdf"
    #         if os.path.isfile(pdf_name):
    #             continue
    #         else:
    #             get_pdf_by_id(pdf_revision_id, pdf_name, revision_log_file)
    #             time.sleep(1)