import os
import openreview
import requests
from tqdm import tqdm

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

client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        )

# parameters
venue_id = 'ICLR.cc/2025/Conference'
pdf_dir = "/Users/xujingjun/SUSTech/Research/LLM_copy/Graph_LLM/Code/all_pdf/"
submissions = client.get_all_notes(invitation=f'{venue_id}/-/Submission')

# alr_num = 643+1216+430+21+1628+873+451
for submission in tqdm(submissions[-2487:-2485]):
    decision = submission.content["venueid"]["value"].split('/')[-1]
    if decision == "Withdrawn_Submission":
        continue
    else:
        # get paper openreview id
        paper_id = submission.id
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
                    get_pdf(id, pdf_name)