import os
import openreview
import requests
from tqdm import tqdm
import time
import re

def get_paper_pdf(link, pdf_path, log_file):
    pdf_url = "https://openreview.net"+link
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(pdf_url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"✅ PDF downloaded: {pdf_path}")
        else:
            print(f"❌ Download failed ({response.status_code}) for ID: {id}")
            with open(log_file, "a") as log:
                log.write(f"{link}\n")
    except Exception as e:
        print(f"❌ Exception for ID {link}: {e}")
        with open(log_file, "a") as log:
            log.write(f"{link}\n")

def get_revision_pdf(venue, id, pdf_path, log_file):
    if "2024" in venue or "2025" in venue:
        pdf_url = "https://openreview.net/notes/edits/attachment?id="+id+"&name=pdf"
    elif "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue or "2017" in venue or "2014" in venue or "2013" in venue:
        pdf_url = "https://openreview.net/references/pdf?id="+id
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(pdf_url, headers=headers, timeout=15)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"✅ PDF downloaded: {pdf_path}")
        else:
            print(f"❌ Download failed ({response.status_code}) for ID: {id}")
            with open(log_file, "a") as log:
                log.write(f"{id}\n")
    except Exception as e:
        print(f"❌ Exception for ID {id}: {e}")
        with open(log_file, "a") as log:
            log.write(f"{id}\n")

if __name__ == "__main__":
    client_v1 = openreview.Client(baseurl='https://api.openreview.net')
    client_v2 = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    venue = 'ICLR.cc/2017/conference'
    pdf_dir = "/data/jingjunx/openreview_pdfs_2017/"
    log_file = "failed_ids_revisions_2017.txt"
    start_idx = 0
    end_idx = 5
    
    if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue or "2017" in venue or "2014" in venue or "2013" in venue:
        if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue:
            submissions = client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='revisions')
        elif "2017" in venue or "2014" in venue or "2013" in venue:
            submissions = client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='revisions')
            
        if submissions is None:
            print(f"No submissions found for venue: {venue}")
        else:
            for submission in tqdm(submissions[start_idx:end_idx]):
                # get paper openreview id
                paper_id = submission.id
                revisions = client_v1.get_references(referent=paper_id, original=True)
                time.sleep(1)
                
                pdf_revisions_ids = []
                for revision in revisions:
                    if "pdf" in revision.content:
                        pdf_revisions_ids.append(revision.id)
                
                if len(pdf_revisions_ids) <= 1:
                    continue
                else:
                    for pdf_revision_id in pdf_revisions_ids:
                        pdf_path = str(pdf_dir)+str(pdf_revision_id)+".pdf"
                        if os.path.isfile(pdf_path):
                            continue
                        else:
                            get_revision_pdf(venue, pdf_revision_id, pdf_path, log_file)
                            time.sleep(1)
    else:
        submissions = client_v2.get_all_notes(invitation=f'{venue}/-/Submission', details='revisions')
        if submissions is None:
            print(f"No submissions found for venue: {venue}")
        else:
            for submission in tqdm(submissions[start_idx:end_idx]):
                decision = submission.content["venueid"]["value"].split('/')[-1]
                if decision == "Withdrawn_Submission":
                    continue
                else:
                    # get paper openreview id
                    paper_id = submission.id
                    revisions = client_v2.get_note_edits(note_id=paper_id)
                    if len(revisions) <= 1:
                        continue
                    else:
                        for revision in revisions:
                            pdf_revision_id = revision.id
                            pdf_path = str(pdf_dir)+str(pdf_revision_id)+".pdf"
                            if os.path.isfile(pdf_path):
                                continue
                            else:
                                time.sleep(1)
                                get_revision_pdf(venue, pdf_revision_id, pdf_path, log_file)
                                time.sleep(1)