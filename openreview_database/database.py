import os
import re
import ast
import time
import json
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import openreview
import arxiv
from arxiv import UnexpectedEmptyPageError
from .sqlDatabase import sqlDatabase
from .pdf_utils import connect_diffs_and_paragraphs, extract_paragraphs_from_pdf_new

class Database:
    def __init__(self, host: str = "localhost", dbname: str = "iclr_openreview_database", user: str = "jingjunx", password: str = "", port: str = "5432") -> None:
        self.client_v1 = openreview.Client(
            baseurl='https://api.openreview.net'
        )
        
        self.client_v2 = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        )
        
        self.db = sqlDatabase(host=host, dbname=dbname, user=user, password=password, port=port)

    def construct_review_table(self, venue: str) -> None:
        # create sql table
        self.db.create_review_table()
        if "2025" or "2024" in venue:
            # get all reviews
            reviews = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission', details='replies')
            # remove withdrawn papers
            for review in tqdm(reviews):
                decision = review.content["venueid"]["value"].split('/')[-1]
                if decision == "Withdrawn_Submission":
                    continue
                else:
                    # # get paper openreview id
                    # paper_id = review.id
                    # get all the replies
                    replies = review.details["replies"]
                    for reply in replies:
                        # get review openreview id
                        reply_id = reply["id"]
                        # get replyto openreview id
                        replyto_id = reply["replyto"]
                        # get writer id
                        writer = reply["signatures"][0].split('/')[-1]
                        # get time
                        time = datetime.fromtimestamp(reply['tmdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                        # classify the type of comment
                        if reply["content"].get("summary") is not None: # reviewer initial comment
                            title = "Official Review by " + reply["signatures"][0].split('/')[-1]
                            content = {
                                "Summary": reply["content"]["summary"]["value"],
                                "Soundness": reply["content"]["soundness"]["value"],
                                "Presentation": reply["content"]["presentation"]["value"],
                                "Contribution": reply["content"]["contribution"]["value"],
                                "Strengths": reply["content"]["strengths"]["value"],
                                "Weaknesses": reply["content"]["weaknesses"]["value"],
                                "Questions": reply["content"]["questions"]["value"],
                                "Rating": reply["content"]["rating"]["value"],
                                "Confidence": reply["content"]["confidence"]["value"],
                            }
                        elif reply["content"].get("comment") is not None and reply["content"].get("decision") is None: # author to reviewer / reviewer to author / authors initial rebuttal / author revised paper
                            # set title
                            if writer == "Authors":
                                title = "Response by Authors"
                            else:
                                title = "Response by Reviewer"
                            # set content
                            if reply["content"].get("title") is not None:
                                subtitle = reply["content"]["title"]["value"]
                            else:
                                subtitle = ""
                            content = {
                                "Title": subtitle,
                                "Comment": reply["content"]["comment"]["value"]
                            }
                        elif reply["content"].get("metareview") is not None: # meta review by area chair
                            title = "Meta Review of " + reply["signatures"][0].split('/')[-2] + " by " + reply["signatures"][0].split('/')[-1]
                            meta_review = ""
                            additional_comments = ""
                            try:
                                meta_review = reply["content"]["metareview"]["value"]
                            except:
                                pass
                            try:
                                additional_comments = reply["content"]["additional_comments_on_reviewer_discussion"]["value"]
                            except:
                                pass
                            content = {
                                "Meta Review": meta_review,
                                "Additional Comments On Reviewer Discussion": additional_comments,
                            }
                        elif reply["content"].get("decision") is not None: # Paper Decision by Program Chair
                            # set title
                            if reply["content"].get("title") is not None:
                                title = reply["content"]["title"]["value"]
                            else:
                                title = "Paper Decision"
                            # set content
                            try:
                                comment = reply["content"]["comment"]["value"]
                            except:
                                comment = ""
                            content = {
                                "Decision": reply["content"]["decision"]["value"],
                                "Comment": comment,
                            }
                        self.db.insert_review(venue, reply_id, replyto_id, writer, title, content, time)
        elif "2023" or "2022" or "2021" or "2020" or "2019" or "2018" or "2017" or "2014" or "2013" in venue:
            # get all reviews
            if "2023" or "2022" or "2021" or "2020" or "2019" or "2018" in venue:
                reviews = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='replies')
            elif "2017" or "2014" or "2013" in venue:
                reviews = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='replies')
            # remove withdrawn papers
            for review in tqdm(reviews):
                # # get paper openreview id
                # paper_id = review.id
                # get all the replies
                replies = review.details["replies"]
                for reply in replies:
                    # get review openreview id
                    reply_id = reply["id"]
                    # get replyto openreview id
                    replyto_id = reply["replyto"]
                    # get writer id
                    writer = reply["signatures"][0].split('/')[-1]
                    # get time
                    time = datetime.fromtimestamp(reply['tmdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                    # classify the type of comment
                    if reply["content"].get("summary_of_the_paper") is not None: # reviewer initial comment
                        title = "Official Review by " + reply["signatures"][0].split('/')[-1]
                        content = {
                            "Summary of the Paper": reply["content"]["summary_of_the_paper"],
                            "Main Review": reply["content"]["main_review"], # 2022
                            # "Strength and Weaknesses": reply["content"]["strength_and_weaknesses"], # 2023
                            # "Clarity, quality, novelty and reproducibility": reply["content"]["clarity,_quality,_novelty_and_reproducibility"], # 2023
                            "Summary of the review": reply["content"]["summary_of_the_review"],
                            # "Correctness": reply["content"]["correctness"], # 2023
                            "Technical novelty and significance": reply["content"]["technical_novelty_and_significance"],
                            "Empirical novelty and significance": reply["content"]["empirical_novelty_and_significance"],
                            "Recommendation": reply["content"]["recommendation"],
                            "Confidence": reply["content"]["confidence"],
                        }
                    elif reply["content"].get("comment") is not None and reply["content"].get("decision") is None: # author to reviewer / reviewer to author / authors initial rebuttal / author revised paper
                        # set title
                        if writer == "Authors":
                            title = "Response by Authors"
                        else:
                            title = "Response by Reviewer"
                        # set content
                        if reply["content"].get("title") is not None:
                            subtitle = reply["content"]["title"]
                        else:
                            subtitle = ""
                        content = {
                            "Title": subtitle,
                            "Comment": reply["content"]["comment"]
                        }
                    elif reply["content"].get("decision") is not None: # Paper Decision by Program Chair
                        # set title
                        if reply["content"].get("title") is not None:
                            title = reply["content"]["title"]
                        else:
                            title = "Paper Decision"
                        # set content
                        content = reply["content"]
                        content = {
                            "Decision": reply["content"]["decision"],
                            "Comment": reply["content"]["comment"], # 2022
                            # "Metareview: summary, strengths and weaknesses": reply["content"]["metareview:_summary,_strengths_and_weaknesses"], # 2023
                            # "Justification for why not higher score": reply["content"]["justification_for_why_not_higher_score"], # 2023
                            # "Justification for why not lower score": reply["content"]["justification_for_why_not_lower_score"] # 2023
                        }
                    self.db.insert_review(venue, reply_id, replyto_id, writer, title, content, time)
                        
    def construct_author_table(self, venue: str) -> None: # author_list contains authors' openreview ids
        # create sql table
        self.db.create_author_table()
        
        if "2025" or "2024" in venue:
            author_set = set()
            # retrieve all the authors in this venue, skip the authors in the withdrawn submissions
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
            for submission in tqdm(submissions):
                # get paper decision and remove withdrawn papers
                decision = submission.content["venueid"]["value"].split('/')[-1]
                if decision == "Withdrawn_Submission":
                    continue
                else:
                    # get author openreview ids
                    author_ids = submission.content["authorids"]["value"]
                    author_set.update(author_ids)
            # get profiles
            author_profiles = openreview.tools.get_profiles(self.client_v2, author_set)
            for profile in tqdm(author_profiles):
                # get author fullname and author openreview id
                all_names = profile.content["names"]
                author_id = ""
                fullname = ""
                for name in all_names:
                    if name.get("preferred") is not None:
                        if name["preferred"] == True:
                            author_id = name["username"]
                            fullname = name["fullname"]
                            break
                    else:
                        if name.get("username") is not None:
                            author_id = name["username"]
                            try:
                                fullname = name["fullname"]
                                break
                            except:
                                fullname = author_id
                        else:
                            pass
                if author_id == "": # remove the author with no username
                    author_id = profile.id
                # get email
                try:
                    email = profile.content["preferredEmail"]
                except:
                    try:
                        email = profile.content["emailsConfirmed"][0]
                    except:
                        try:
                            email = profile.content["emails"][0]
                        except:
                            email = ""
                # get affiliation
                try:
                    affiliation = profile.content["history"][0]["institution"]["name"]
                except:
                    affiliation = ""
                # get homepage
                if profile.content.get("homepage") is not None:
                    homepage = profile.content["homepage"]
                else:
                    homepage = ""
                # get dblp
                if profile.content.get("dblp") is not None:
                    dblp = profile.content["dblp"]
                else:
                    dblp = ""
                # add to author table
                self.db.insert_author(venue, author_id, fullname, email, affiliation, homepage, dblp)
        elif "2023" or "2022" or "2021" or "2020" or "2019" or "2018" or "2017" or "2014" or "2013" in venue:
            author_set = set()
            # retrieve all the authors in this venue, skip the authors in the withdrawn submissions
            if "2023" or "2022" or "2021" or "2020" or "2019" or "2018" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission')
            elif "2017" or "2014" or "2013" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission')
            for submission in tqdm(submissions):
                # get author openreview ids
                author_ids = submission.content["authorids"]
                author_set.update(author_ids)
            # get profiles
            author_profiles = openreview.tools.get_profiles(self.client_v1, author_set)
            for profile in tqdm(author_profiles):
                # get author fullname and author openreview id
                all_names = profile.content["names"]
                author_id = ""
                fullname = ""
                for name in all_names:
                    if name.get("preferred") is not None:
                        if name["preferred"] == True:
                            author_id = name["username"]
                            fullname = name["fullname"]
                            break
                    else:
                        if name.get("username") is not None:
                            author_id = name["username"]
                            try:
                                fullname = name["fullname"]
                                break
                            except:
                                fullname = author_id
                        else:
                            pass
                if author_id == "": # remove the author with no username
                    author_id = profile.id
                # get email
                try:
                    email = profile.content["preferredEmail"]
                except:
                    try:
                        email = profile.content["emailsConfirmed"][0]
                    except:
                        try:
                            email = profile.content["emails"][0]
                        except:
                            email = ""
                # get affiliation
                try:
                    affiliation = profile.content["history"][0]["institution"]["name"]
                except:
                    affiliation = ""
                # get homepage
                if profile.content.get("homepage") is not None:
                    homepage = profile.content["homepage"]
                else:
                    homepage = ""
                # get dblp
                if profile.content.get("dblp") is not None:
                    dblp = profile.content["dblp"]
                else:
                    dblp = ""
                # add to author table
                self.db.insert_author(venue, author_id, fullname, email, affiliation, homepage, dblp)
                
            
    def construct_paper_table(self, venue: str, with_pdf: bool = True, filter_list: list = [], pdf_dir: str = "./", log_file: str = "not_found_papers.txt") -> None:
        # create sql table
        self.db.create_papers_table()
        if with_pdf:
            self.db.create_paragraphs_table()
        
        if "2025" or "2024" in venue:
            # get all submissions
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
            for submission in tqdm(submissions):
                # get paper decision and remove withdrawn papers
                decision = submission.content["venueid"]["value"].split('/')[-1]
                if decision == "Withdrawn_Submission":
                    continue
                else:
                    if decision == "Conference":
                        decision = submission.content["venue"]["value"]
                    # get paper openreview id
                    paper_id = submission.id
                    
                    # insert paragraphs
                    if with_pdf:
                        pdf_path = str(pdf_dir)+str(paper_id)+".pdf"
                        if os.path.exists(pdf_path):
                            try:
                                structured_content = extract_paragraphs_from_pdf_new(pdf_path, filter_list)
                                
                                paragraph_counter = 1
                                for section, paragraphs in structured_content.items():
                                    for paragraph in paragraphs:
                                        self.db.insert_paragraph(venue, paper_id, paragraph_counter, section, paragraph)
                                        paragraph_counter += 1
                                
                                # delete the original pdf
                                os.remove(pdf_path)
                                print(pdf_path+" Deleted")
                            except:
                                with open(log_file, "a") as log:
                                    log.write(f"{pdf_path}\n")
                        else:
                            with open(log_file, "a") as log:
                                log.write(f"{pdf_path}\n")
                    
                    # # get revisions and their time
                    # revisions = {}
                    # note_edits = self.client_v2.get_note_edits(note_id=paper_id)
                    # try:
                    #     for note in note_edits:
                    #         revisions[note.id] = {
                    #             "Time": datetime.fromtimestamp(note.tmdate / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                    #             "Title": note.invitation.split('/')[-1]
                    #         }
                    # except:
                    #     pass
                    # get title
                    title = submission.content["title"]["value"]
                    # get abstract
                    abstract = submission.content["abstract"]["value"]
                    # # get author openreview ids
                    # author_ids = submission.content["authorids"]["value"]
                    # # get author full names
                    # fullnames = submission.content["authors"]["value"]
                    # get paper's pdf
                    pdf = submission.content["pdf"]["value"]
                    # add to paper table
                    self.db.insert_paper(venue, paper_id, title, abstract, decision, pdf)
        elif "2023" or "2022" or "2021" or "2020" or "2019" or "2018" or "2017" or "2014" or "2013" in venue:
            # get all submissions
            if "2023" or "2022" or "2021" or "2020" or "2019" or "2018" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission')
            elif "2017" or "2014" or "2013" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission')
            for submission in tqdm(submissions):
                # get paper decision and remove withdrawn papers
                try:
                    decision = submission.content["venue"]
                except:
                    decision = ""
                
                # get paper openreview id
                paper_id = submission.id
                
                # insert paragraphs
                pdf_path = str(pdf_dir)+str(paper_id)+".pdf"
                if os.path.exists(pdf_path):
                    try:
                        structured_content = extract_paragraphs_from_pdf_new(pdf_path, filter_list)
                        
                        paragraph_counter = 1
                        for section, paragraphs in structured_content.items():
                            for paragraph in paragraphs:
                                self.db.insert_paragraph(venue, paper_id, paragraph_counter, section, paragraph)
                                paragraph_counter += 1
                        
                        # delete the original pdf
                        os.remove(pdf_path)
                        print(pdf_path+" Deleted")
                    except:
                        with open(log_file, "a") as log:
                            log.write(f"{pdf_path}\n")
                else:
                    with open(log_file, "a") as log:
                        log.write(f"{pdf_path}\n")
                
                 # get title
                title = submission.content["title"]
                # get abstract
                abstract = submission.content["abstract"]
                # get paper's pdf
                pdf = submission.content["pdf"]
                # insert paper
                self.db.insert_paper(venue, paper_id, title, abstract, decision, pdf)
    
    def construct_revision_table(self, venue: str, filter_list: list, pdf_dir: str = "./", log_file: str = "not_found_pdfs.txt") -> None:
        # create sql table
        self.db.create_revisions_table()
        self.db.create_paragraphs_table()
        import time
        if "2025" or "2024" in venue:
            # get all submissions
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
            for submission in tqdm(submissions):
                # get paper decision and remove withdrawn papers
                decision = submission.content["venueid"]["value"].split('/')[-1]
                if decision == "Withdrawn_Submission":
                    continue
                else:
                    # get paper openreview id
                    paper_id = submission.id
                    # get revisions and their time
                    revisions = {}
                    # all_diffs = []
                    note_edits = self.client_v2.get_note_edits(note_id=paper_id)
                    time.sleep(1)
                    if len(note_edits) <= 1:
                        continue
                    else:
                        for note in note_edits:
                            revisions[note.id] = {
                                "Time": datetime.fromtimestamp(note.tmdate / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                                "Title": note.invitation.split('/')[-1]
                            }
                        # sorted by time
                        sorted_revisions = sorted(revisions.items(), key=lambda x: datetime.strptime(x[1]["Time"], "%Y-%m-%d %H:%M:%S"))
                        num_revision = len(sorted_revisions)
                        
                        if num_revision <= 1:
                            continue
                        else:
                            # original_pdf = "original.pdf"
                            # modified_pdf = "modified.pdf"
                            original_id = None
                            modified_id = None
                            
                            for idx, revision in enumerate(sorted_revisions):

                                original_id = modified_id
                                modified_id = revision[0]
                                original_pdf = str(pdf_dir)+str(original_id)+".pdf"
                                modified_pdf = str(pdf_dir)+str(modified_id)+".pdf"
                                if idx > 1:
                                    time = revision[1]["Time"]
                                    
                                    # get_pdf(original_id, original_pdf)
                                    # get_pdf(modified_id, modified_pdf)
                                    try:
                                        content = connect_diffs_and_paragraphs(original_pdf, modified_pdf, filter_list)
                                    except:
                                        continue
                                    
                                    self.db.insert_revision(venue, original_id, modified_id, content, time)
                                    if os.path.exists(original_pdf):
                                        try:
                                            # insert paragraphs
                                            structured_content = extract_paragraphs_from_pdf_new(original_pdf, filter_list)
                        
                                            paragraph_counter = 1
                                            for section, paragraphs in structured_content.items():
                                                for paragraph in paragraphs:
                                                    self.db.insert_paragraph(venue, original_id, paragraph_counter, section, paragraph)
                                                    paragraph_counter += 1
                                            
                                            # delete the original pdf
                                            os.remove(original_pdf)
                                            print(original_pdf+" Deleted")
                                        except:
                                            with open(log_file, "a") as log:
                                                log.write(f"{original_pdf}\n")
                                    else:
                                        with open(log_file, "a") as log:
                                            log.write(f"{original_pdf}\n")
                                        print("File not exist")
                                    if idx == num_revision - 1:
                                        if os.path.exists(modified_pdf):
                                            # insert paragraphs
                                            try:
                                                structured_content = extract_paragraphs_from_pdf_new(modified_pdf, filter_list)
                            
                                                paragraph_counter = 1
                                                for section, paragraphs in structured_content.items():
                                                    for paragraph in paragraphs:
                                                        self.db.insert_paragraph(venue, modified_id, paragraph_counter, section, paragraph)
                                                        paragraph_counter += 1
                                                        
                                                # delete the modified pdf
                                                os.remove(modified_pdf)
                                                print(modified_pdf+" Deleted")
                                            except:
                                                with open(log_file, "a") as log:
                                                    log.write(f"{modified_pdf}\n")
                                        else:
                                            with open(log_file, "a") as log:
                                                log.write(f"{modified_pdf}\n")
                                            print("File not exist")
        elif "2023" or "2022" or "2021" or "2020" or "2019" or "2018" or "2017" or "2014" or "2013" in venue:
            # get all submissions
            if "2023" or "2022" or "2021" or "2020" or "2019" or "2018" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='revisions')
            elif "2017" or "2014" or "2013" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='revisions')
            for submission in tqdm(submissions):
                # get paper openreview id
                paper_id = submission.id
                # get revisions and their time
                revisions = {}
                # get revisions and their time
                note_edits = self.client_v1.get_references(referent=paper_id, original=True)
                time.sleep(1)
                
                filtered_notes = []
                for note in note_edits:
                    if "pdf" in note.content:
                        filtered_notes.append(note)
                
                if len(filtered_notes) <= 1:
                    continue
                else:
                    for note in filtered_notes:
                        revisions[note.id] = {
                            "Time": datetime.fromtimestamp(note.tmdate / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                            "Title": "Paper Revision"
                        }
                    # sorted by time
                    sorted_revisions = sorted(revisions.items(), key=lambda x: datetime.strptime(x[1]["Time"], "%Y-%m-%d %H:%M:%S"))
                    num_revision = len(sorted_revisions)
                    
                    original_id = None
                    modified_id = None
                    
                    for idx, revision in enumerate(sorted_revisions):
                        original_id = modified_id
                        modified_id = revision[0]
                        original_pdf = str(pdf_dir)+str(original_id)+".pdf"
                        modified_pdf = str(pdf_dir)+str(modified_id)+".pdf"
                        if idx < 1:
                            continue
                        else:
                            date = revision[1]["Time"]
                            try:
                                content = connect_diffs_and_paragraphs(original_pdf, modified_pdf, filter_list)
                            except:
                                continue
                            
                            self.db.insert_revision(venue, original_id, modified_id, content, date)
                            # if original_id != paper_id:
                            if os.path.exists(original_pdf):
                                try:
                                    # insert paragraphs
                                    structured_content = extract_paragraphs_from_pdf_new(original_pdf, filter_list)
                
                                    paragraph_counter = 1
                                    for section, paragraphs in structured_content.items():
                                        for paragraph in paragraphs:
                                            self.db.insert_paragraph(venue, original_id, paragraph_counter, section, paragraph)
                                            paragraph_counter += 1
                                    
                                    # delete the original pdf
                                    os.remove(original_pdf)
                                    print(original_pdf+" Deleted")
                                except:
                                    with open(log_file, "a") as log:
                                        log.write(f"{original_pdf}\n")
                            else:
                                with open(log_file, "a") as log:
                                    log.write(f"{original_pdf}\n")
                                print("File not exist")
                            if idx == num_revision - 1:
                                if os.path.exists(modified_pdf):
                                    # insert paragraphs
                                    try:
                                        structured_content = extract_paragraphs_from_pdf_new(modified_pdf, filter_list)
                    
                                        paragraph_counter = 1
                                        for section, paragraphs in structured_content.items():
                                            for paragraph in paragraphs:
                                                self.db.insert_paragraph(venue, modified_id, paragraph_counter, section, paragraph)
                                                paragraph_counter += 1
                                                
                                        # delete the modified pdf
                                        os.remove(modified_pdf)
                                        print(modified_pdf+" Deleted")
                                    except:
                                        with open(log_file, "a") as log:
                                            log.write(f"{modified_pdf}\n")
                                else:
                                    with open(log_file, "a") as log:
                                        log.write(f"{modified_pdf}\n")
                                    print("File not exist")
    
    def construct_papers_authors_table(self, venue: str) -> None:
        # create sql table
        self.db.create_papers_authors_table()
        if "2025" or "2024" in venue:
            # get all submissions
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
            for submission in tqdm(submissions):
                # get paper decision and remove withdrawn papers
                decision = submission.content["venueid"]["value"].split('/')[-1]
                if decision == "Withdrawn_Submission":
                    continue
                else:
                    # get paper openreview id
                    paper_id = submission.id
                    # get author openreview ids
                    author_ids = set(submission.content["authorids"]["value"])
                    # add to papers authors table
                    for author_id in author_ids:
                        self.db.insert_paper_authors(venue, paper_id, author_id)
        elif "2023" or "2022" or "2021" or "2020" or "2019" or "2018" or "2017" or "2014" or "2013" in venue:
            # get all submissions
            if "2023" or "2022" or "2021" or "2020" or "2019" or "2018" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission')
            elif "2017" or "2014" or "2013" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission')
            for submission in tqdm(submissions):
                paper_id = submission.id
                # get author openreview ids
                author_ids = set(submission.content["authorids"])
                # add to papers authors table
                for author_id in author_ids:
                    self.db.insert_paper_authors(venue, paper_id, author_id)
                
    def construct_papers_revisions_table(self, venue: str) -> None:
        # create sql table
        import time
        self.db.create_papers_revisions_table()
        if "2025" or "2024" in venue:
            # get all submissions
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
            for submission in tqdm(submissions):
                # get paper decision and remove withdrawn papers
                decision = submission.content["venueid"]["value"].split('/')[-1]
                if decision == "Withdrawn_Submission":
                    continue
                else:
                    # get paper openreview id
                    paper_id = submission.id
                    # get revisions
                    note_edits = self.client_v2.get_note_edits(note_id=paper_id)
                    time.sleep(1)
                    if len(note_edits) <= 1:
                        continue
                    else:
                        for note in note_edits:
                            revision_openreview_id = note.id
                            title = note.invitation.split('/')[-1]
                            date = datetime.fromtimestamp(note.tmdate / 1000).strftime("%Y-%m-%d %H:%M:%S")
                            self.db.insert_paper_revisions(venue, paper_id, revision_openreview_id, title, date)
        elif "2023" or "2022" or "2021" or "2020" or "2019" or "2018" or "2017" or "2014" or "2013" in venue:
            # get all submissions
            if "2023" or "2022" or "2021" or "2020" or "2019" or "2018" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='revisions')
            elif "2017" or "2014" or "2013" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='revisions')
            for submission in tqdm(submissions):
                # get paper openreview id
                paper_id = submission.id
                # get revisions
                revisions = self.client_v1.get_references(referent=paper_id, original=True)
                time.sleep(1)

                filtered_revisions = []
                for revision in revisions:
                    if "pdf" in revision.content:
                        filtered_revisions.append(revision)
                
                if len(filtered_revisions) <= 1:
                    continue
                else:
                    for revision in filtered_revisions:
                        revision_openreview_id = revision.id
                        title = "Paper Revision"
                        date = datetime.fromtimestamp(revision.tmdate / 1000).strftime("%Y-%m-%d %H:%M:%S")
                        self.db.insert_paper_revisions(venue, paper_id, revision_openreview_id, title, date)
                
    def construct_papers_reviews_table(self, venue: str) -> None:
        # create sql table
        self.db.create_papers_reviews_table()
        if "2025" or "2024" in venue:
            # get all submissions with reviews
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission', details='replies')
            # remove withdrawn papers
            for submission in tqdm(submissions):
                decision = submission.content["venueid"]["value"].split('/')[-1]
                if decision == "Withdrawn_Submission":
                    continue
                else:
                    # get paper openreview id
                    paper_id = submission.id
                    # get reviews
                    replies = submission.details["replies"]
                    for reply in replies:
                        # get review openreview id
                        reply_id = reply["id"]
                        # get time
                        time = datetime.fromtimestamp(reply['tmdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                        # get writer id
                        writer = reply["signatures"][0].split('/')[-1]
                        if reply["content"].get("summary") is not None: # reviewer initial comment
                            title = "Official Review by " + reply["signatures"][0].split('/')[-1]
                        elif reply["content"].get("comment") is not None and reply["content"].get("decision") is None: # author to reviewer / reviewer to author / authors initial rebuttal / author revised paper
                            # set title
                            if writer == "Authors":
                                title = "Response by Authors"
                            else:
                                title = "Response by Reviewer"
                        elif reply["content"].get("metareview") is not None: # meta review by area chair
                            title = "Meta Review of " + reply["signatures"][0].split('/')[-2] + " by " + reply["signatures"][0].split('/')[-1]
                        elif reply["content"].get("decision") is not None: # Paper Decision by Program Chair
                            # set title
                            if reply["content"].get("title") is not None:
                                title = reply["content"]["title"]["value"]
                            else:
                                title = "Paper Decision"
                        self.db.insert_paper_reviews(venue, paper_id, reply_id, title, time)
        elif "2023" or "2022" or "2021" or "2020" or "2019" or "2018" or "2017" or "2014" or "2013" in venue:
            # get all reviews
            if "2023" or "2022" or "2021" or "2020" or "2019" or "2018" in venue:
                reviews = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='replies')
            elif "2017" or "2014" or "2013" in venue:
                reviews = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='replies')
            # remove withdrawn papers
            for review in tqdm(reviews):
                paper_id = review.id
                # get all the replies
                replies = review.details["replies"]
                for reply in replies:
                    # get review openreview id
                    reply_id = reply["id"]
                    # get time
                    time = datetime.fromtimestamp(reply['tmdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                    # get writer id
                    writer = reply["signatures"][0].split('/')[-1]
                    # classify the type of comment
                    if reply["content"].get("summary_of_the_paper") is not None: # reviewer initial comment
                        title = "Official Review by " + writer
                    elif reply["content"].get("comment") is not None and reply["content"].get("decision") is None: # author to reviewer / reviewer to author / authors initial rebuttal / author revised paper
                        # set title
                        if writer == "Authors":
                            title = "Response by Authors"
                        else:
                            title = "Response by Reviewer"
                    elif reply["content"].get("decision") is not None: # Paper Decision by Program Chair
                        # set title
                        if reply["content"].get("title") is not None:
                            title = reply["content"]["title"]
                        else:
                            title = "Paper Decision"
                    self.db.insert_paper_reviews(venue, paper_id, reply_id, title, time)
    
    def construct_revisions_reviews_table(self, venue: str) -> None:
        # create sql table
        self.db.create_revisions_reviews_table()
        
        # get related edge
        papers_reviews_df = self.get_edge_features_by_venue("papers_reviews", venue)
        papers_revisions_df = self.get_edge_features_by_venue("papers_revisions", venue)
        
        # get unique paper ids
        unique_paper_ids = papers_revisions_df['paper_openreview_id'].unique()
        
        for paper_id in tqdm(unique_paper_ids):
            # get the revision ids
            paper_revision_edges = papers_revisions_df[papers_revisions_df['paper_openreview_id'] == paper_id].sort_values(by='time', ascending=True)
            
            # get the review ids
            paper_review_edges = papers_reviews_df[papers_reviews_df['paper_openreview_id'] == paper_id].sort_values(by='time', ascending=True)
            
            start_idx = 0
            for revision in paper_revision_edges.itertuples():
                # get the revision time
                revision_time = revision.time
                # get the revision id
                revision_id = revision.revision_openreview_id
                
                # get the review ids
                for review in paper_review_edges.iloc[start_idx:].itertuples():
                    # get the review time
                    review_time = review.time
                    if review_time > revision_time:
                        break
                    
                    # get the review id
                    review_id = review.review_openreview_id
                    
                    # insert the edge
                    self.db.insert_revision_reviews(venue, revision_id, review_id)
                    
                    start_idx += 1
                    
    def _title_cleaner(self, title: str) -> str:
        """
        Remove all symbols (non-alphanumeric, non-space characters) from the title.
        Collapses multiple spaces down to one and trims ends.
        """
        # Remove anything that isn't a letter, number, or whitespace
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', title)
        # Collapse multiple spaces and strip leading/trailing spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.strip().lower()
    
    def _search_title_with_name(self, title, name, max_result=20):
        query = f"ti:{title} AND au:{name}"
        search = arxiv.Search(
            query=query,
            max_results=max_result,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        try:
            for result in search.results():
                if (self._title_cleaner(result.title) == self._title_cleaner(title)):
                    return result.entry_id
        except UnexpectedEmptyPageError:
            return None
    
    def construct_openreview_arxiv_table(self, venue: str) -> None:
        # create sql table
        self.db.create_openreview_arxiv_table()
        
        # get papers from papers table
        # submissions = self.db.get_papers()
        
        # get papers through openreview api
        submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
        # match openreview_id with arxiv_id
        for submission in tqdm(submissions):
            # get paper decision and remove withdrawn papers
            decision = submission.content["venueid"]["value"].split('/')[-1]
            if decision == "Withdrawn_Submission":
                continue
            else:
                # get paper openreview id
                openreview_id = submission.id
                # openreview_id = submission[0]
                
                # get title
                title = submission.content["title"]["value"]
                # title = submission[1]
                
                # get author full names
                author_names = submission.content["authors"]["value"]
                # author_names = submission[2]
                
                # get arxiv id based on title and the first author
                arxiv_id = self._search_title_with_name(title, author_names[0])
                # insert into openreview_arxiv table
                self.db.insert_openreview_arxiv(openreview_id, arxiv_id, title, author_names)
    
    def construct_paper_table_by_csv(self, csv_file_path: str, with_pdf: bool = True, filter_list: list = [], pdf_dir: str = "./", log_file: str = "not_found_papers.txt") -> None:
        # create sql table
        self.db.create_papers_table()
        if with_pdf:
            self.db.create_paragraphs_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into papers table
        for index, row in df.iterrows():
            row.to_dict()
            self.db.insert_paper(**row)
            
            venue = row['venue']
            paper_id = row['paper_openreview_id']
            if with_pdf:
                pdf_path = str(pdf_dir)+str(paper_id)+".pdf"
                if os.path.exists(pdf_path):
                    try:
                        structured_content = extract_paragraphs_from_pdf_new(pdf_path, filter_list)
                        
                        paragraph_counter = 1
                        for section, paragraphs in structured_content.items():
                            for paragraph in paragraphs:
                                self.db.insert_paragraph(venue, paper_id, paragraph_counter, section, paragraph)
                                paragraph_counter += 1
                        
                        # delete the original pdf
                        os.remove(pdf_path)
                        print(pdf_path+" Deleted")
                    except:
                        with open(log_file, "a") as log:
                            log.write(f"{pdf_path}\n")
                else:
                    with open(log_file, "a") as log:
                        log.write(f"{pdf_path}\n")
        print("Paper table constructed successfully")
                        
    def construct_revision_table_by_csv(self, csv_file_path: str, with_pdf: bool = True, filter_list: list = [], pdf_dir: str = "./", log_file: str = "not_found_revisions.txt") -> None:
        # create sql table
        self.db.create_revisions_table()
        self.db.create_paragraphs_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into revisions table
        for index, row in df.iterrows():
            row.to_dict()
            row["content"] = ast.literal_eval(row["content"])
            print(row)
            self.db.insert_revision(**row)
            
            venue = row['venue']
            revision_id = row['revision_openreview_id']
            original_id = row['original_openreview_id']
            if with_pdf:
                pdf_path = str(pdf_dir)+str(revision_id)+".pdf"
                if os.path.exists(pdf_path):
                    try:
                        structured_content = extract_paragraphs_from_pdf_new(pdf_path, filter_list)
                        
                        paragraph_counter = 1
                        for section, paragraphs in structured_content.items():
                            for paragraph in paragraphs:
                                self.db.insert_paragraph(venue, revision_id, paragraph_counter, section, paragraph)
                                paragraph_counter += 1
                                
                        # delete the original pdf
                        os.remove(pdf_path)
                        print(pdf_path+" Deleted")
                    except:
                        with open(log_file, "a") as log:
                            log.write(f"{pdf_path}\n")
                else:
                    with open(log_file, "a") as log:
                        log.write(f"{pdf_path}\n")
                        
                pdf_path = str(pdf_dir)+str(original_id)+".pdf"
                if os.path.exists(pdf_path):
                    try:
                        structured_content = extract_paragraphs_from_pdf_new(pdf_path, filter_list)
                        
                        paragraph_counter = 1
                        for section, paragraphs in structured_content.items():
                            for paragraph in paragraphs:
                                self.db.insert_paragraph(venue, original_id, paragraph_counter, section, paragraph)
                                paragraph_counter += 1
                                
                        # delete the original pdf
                        os.remove(pdf_path)
                        print(pdf_path+" Deleted")
                    except:
                        with open(log_file, "a") as log:
                            log.write(f"{pdf_path}\n")
                else:
                    with open(log_file, "a") as log:
                        log.write(f"{pdf_path}\n")
        print("Revision table constructed successfully")
    
    def construct_author_table_by_csv(self, csv_file_path: str) -> None:
        # create sql table
        self.db.create_author_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into authors table
        for index, row in df.iterrows():
            row.to_dict()
            self.db.insert_author(**row)
        print("Author table constructed successfully")
            
    def construct_review_table_by_csv(self, csv_file_path: str) -> None:
        # create sql table
        self.db.create_review_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into reviews table
        for index, row in df.iterrows():
            row.to_dict()
            row["content"] = ast.literal_eval(row["content"])
            self.db.insert_review(**row)
        print("Review table constructed successfully")
        
    def construct_papers_authors_table_by_csv(self, csv_file_path: str) -> None:
        # create sql table
        self.db.create_papers_authors_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into papers_authors table
        for index, row in df.iterrows():
            self.db.insert_paper_author(**row)
        print("Papers authors table constructed successfully")
    
    def construct_papers_revisions_table_by_csv(self, csv_file_path: str) -> None:
        # create sql table
        self.db.create_papers_revisions_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into papers_revisions table
        for index, row in df.iterrows():
            self.db.insert_paper_revision(**row)
        print("Papers revisions table constructed successfully")
    
    def construct_papers_reviews_table_by_csv(self, csv_file_path: str) -> None:
        # create sql table
        self.db.create_papers_reviews_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into papers_reviews table
        for index, row in df.iterrows():
            self.db.insert_paper_review(**row)
        print("Papers reviews table constructed successfully")

    def construct_papers_revisions_reviews_table_by_csv(self, csv_file_path: str) -> None:
        # create sql table
        self.db.create_papers_revisions_reviews_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into papers_revisions_reviews table
        for index, row in df.iterrows():
            self.db.insert_paper_revision_review(**row)
        print("Papers revisions reviews table constructed successfully")
            
    
    def construct_revisions_reviews_table_by_csv(self, csv_file_path: str) -> None:
        # create sql table
        self.db.create_revisions_reviews_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into revisions_reviews table
        for index, row in df.iterrows():
            self.db.insert_revision_review(**row)
        print("Revisions reviews table constructed successfully")
            
    def construct_openreview_arxiv_table_by_csv(self, csv_file_path: str) -> None:
        # create sql table
        self.db.create_openreview_arxiv_table()
        
        # read csv file
        df = pd.read_csv(csv_file_path)
        
        # insert into openreview_arxiv table
        for index, row in df.iterrows():
            self.db.insert_openreview_arxiv(**row)
        print("Openreview arxiv table constructed successfully")
            
    def construct_review_table_by_json(self, json_file_path: str) -> None:
        # create sql table
        self.db.create_review_table()
        
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into reviews table
        for index, row in enumerate(dataset):
            self.db.insert_review(**row)
        print("Review table constructed successfully")
            
    def construct_author_table_by_json(self, json_file_path: str) -> None:
        # create sql table
        self.db.create_author_table()
        
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into authors table
        for index, row in enumerate(dataset):
            self.db.insert_author(**row)
        print("Author table constructed successfully")
            
    def construct_paper_table_by_json(self, json_file_path: str, with_pdf: bool = True, filter_list: list = [], pdf_dir: str = "./", log_file: str = "not_found_papers.txt") -> None:
        # create sql table
        self.db.create_papers_table()
        if with_pdf:
            self.db.create_paragraphs_table()
            
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into papers table
        for index, row in enumerate(dataset):
            self.db.insert_paper(**row)
            
            venue = row['venue']
            paper_id = row['paper_openreview_id']
            if with_pdf:
                pdf_path = str(pdf_dir)+str(paper_id)+".pdf"
                if os.path.exists(pdf_path):
                    try:
                        structured_content = extract_paragraphs_from_pdf_new(pdf_path, filter_list)
                        
                        paragraph_counter = 1
                        for section, paragraphs in structured_content.items():
                            for paragraph in paragraphs:
                                self.db.insert_paragraph(venue, paper_id, paragraph_counter, section, paragraph)
                                paragraph_counter += 1
                                
                        # delete the original pdf
                        os.remove(pdf_path)
                        print(pdf_path+" Deleted")
                    except:
                        with open(log_file, "a") as log:
                            log.write(f"{pdf_path}\n")
                else:
                    with open(log_file, "a") as log:
                        log.write(f"{pdf_path}\n")
        print("Paper table constructed successfully")
        
    def construct_revision_table_by_json(self, json_file_path: str, with_pdf: bool = True, filter_list: list = [], pdf_dir: str = "./", log_file: str = "not_found_revisions.txt") -> None:
        # create sql table
        self.db.create_revisions_table()
        self.db.create_paragraphs_table()
        
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into revisions table
        for index, row in enumerate(dataset):
            self.db.insert_revision(**row)
            
            venue = row['venue']
            revision_id = row['revision_openreview_id']
            original_id = row['original_openreview_id']
            if with_pdf:
                pdf_path = str(pdf_dir)+str(revision_id)+".pdf"
                if os.path.exists(pdf_path):
                    try:
                        structured_content = extract_paragraphs_from_pdf_new(pdf_path, filter_list)
                        
                        paragraph_counter = 1
                        for section, paragraphs in structured_content.items():
                            for paragraph in paragraphs:
                                self.db.insert_paragraph(venue, revision_id, paragraph_counter, section, paragraph)
                                paragraph_counter += 1
                                
                        # delete the original pdf
                        os.remove(pdf_path)
                        print(pdf_path+" Deleted")
                    except:
                        with open(log_file, "a") as log:
                            log.write(f"{pdf_path}\n")
                else:
                    with open(log_file, "a") as log:
                        log.write(f"{pdf_path}\n")
        print("Revision table constructed successfully")
                        
    def construct_papers_authors_table_by_json(self, json_file_path: str) -> None:
        # create sql table
        self.db.create_papers_authors_table()
        
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into papers_authors table
        for index, row in enumerate(dataset):
            self.db.insert_paper_author(**row)
        print("Papers authors table constructed successfully")
    
    def construct_papers_revisions_table_by_json(self, json_file_path: str) -> None:
        # create sql table
        self.db.create_papers_revisions_table()
        
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into papers_revisions table
        for index, row in enumerate(dataset):
            self.db.insert_paper_revision(**row)
        print("Papers revisions table constructed successfully")
            
    def construct_papers_reviews_table_by_json(self, json_file_path: str) -> None:
        # create sql table
        self.db.create_papers_reviews_table()
        
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into papers_reviews table
        for index, row in enumerate(dataset):
            self.db.insert_paper_review(**row)
        print("Papers reviews table constructed successfully")
    
    def construct_papers_revisions_reviews_table_by_json(self, json_file_path: str) -> None:
        # create sql table
        self.db.create_papers_revisions_reviews_table()
        
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into papers_revisions_reviews table
        for index, row in enumerate(dataset):
            self.db.insert_paper_revision_review(**row)
        print("Papers revisions reviews table constructed successfully")
            
    def construct_revisions_reviews_table_by_json(self, json_file_path: str) -> None:
        # create sql table
        self.db.create_revisions_reviews_table()
        
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into revisions_reviews table
        for index, row in enumerate(dataset):
            self.db.insert_revision_review(**row)
        print("Revisions reviews table constructed successfully")
            
    def construct_openreview_arxiv_table_by_json(self, json_file_path: str) -> None:
        # create sql table
        self.db.create_openreview_arxiv_table()
        
        # read json file
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        # insert into openreview_arxiv table
        for index, row in enumerate(dataset):
            self.db.insert_openreview_arxiv(**row)
        print("Openreview arxiv table constructed successfully")
            
    # def construct_papers_revisions_reviews_table(self, venue):
    #     # create sql table
    #     self.db.create_papers_revisions_reviews_table()
        
    #     # get reviews
    #     # papers_reviews_df = self.db.get_all_papers_reviews()
    #     papers_reviews_df = self.db.get_papers_reviews_by_venue(venue)
        
    #     # get revisions
    #     # paper_revisions_df = self.db.get_all_papers_revisions()
    #     paper_revisions_df = self.db.get_papers_revisions_by_venue(venue)

    #     # get all unique paper_id in revisions_df
    #     unique_paper_ids = paper_revisions_df['paper_openreview_id'].unique()

    #     # match revisions with reviews
    #     for paper_id in tqdm(unique_paper_ids):
    #         revisions = paper_revisions_df[paper_revisions_df['paper_openreview_id'] == paper_id]
    #         revisions_sorted = revisions.sort_values(by='time', ascending=True)
            
    #         reviews = papers_reviews_df[papers_reviews_df['paper_openreview_id'] == paper_id]
    #         reviews_sorted = reviews.sort_values(by='time', ascending=True)
            
    #         start_idx = 0
    #         for revision in revisions_sorted.itertuples():
    #             # venue = revision.venue
                
    #             revision_id = revision.revision_openreview_id
    #             revision_time = revision.time
    #             _revision_time = datetime.strptime(revision_time, "%Y-%m-%d %H:%M:%S")
    #             for review in reviews_sorted.iloc[start_idx:].itertuples():
    #                 review_id = review.review_openreview_id
    #                 review_time = review.time
    #                 _review_time = datetime.strptime(review_time, "%Y-%m-%d %H:%M:%S")
    #                 if _review_time <= _revision_time:
    #                     self.db.insert_paper_revision_review(venue, paper_id, revision_id, review_id, revision_time, review_time)
    #                     # print(venue, paper_id, revision_id, review_id, revision_time, review_time)
    #                     start_idx += 1
    #                 else:
    #                     break
    
    # node          
    def insert_node(self, table: str, node_features: dict) -> None:
        if table == "papers":
            try:
                self.db.insert_paper(**node_features)
                paper_openreview_id = node_features["paper_openreview_id"]
                print(f"Paper with paper_openreview_id {paper_openreview_id} inserted successfully.")
            except: # venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions
                print(f'''The node in 'papers' table requires the following node features:
                      venue: str,
                      paper_openreview_id: str,
                      title: str, 
                      abstract: str, 
                      paper_decision: str,
                      paper_pdf_link: str,
                    
                      And the node features you provided:
                      {node_features}
                      are not qualified
                      ''')
        elif table == "reviews":
            try:
                self.db.insert_review(**node_features)
                review_openreview_id = node_features["review_openreview_id"]
                print(f"Review with review_openreview_id {review_openreview_id} inserted successfully.")
            except: # venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time
                print(f'''The node in 'reviews' table requires the following node features:
                      venue: str,
                      review_openreview_id: str, 
                      replyto_openreview_id: str, 
                      writer: str, 
                      title: str,
                      content: dict,
                      time: str

                      And the node features you provided:
                      {node_features}
                      are not qualified
                      ''')
        elif table == "authors":
            try:
                self.db.insert_author(**node_features)
                author_openreview_id = node_features["author_openreview_id"]
                print(f"Author with author_openreview_id {author_openreview_id} inserted successfully.")
            except: # venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp
                print(f'''The node in 'authors' table requires the following node features:
                      venue: str,
                      author_openreview_id: str,
                      author_full_name: str, 
                      email: str, 
                      affiliation: str,
                      homepage: str,
                      dblp: str

                      And the node features you provided:
                      {node_features}
                      are not qualified
                      ''')
        elif table == "revisions":
            try:
                self.db.insert_revision(**node_features)
                modified_openreview_id = node_features["modified_openreview_id"]
                print(f"Revision with modified_openreview_id {modified_openreview_id} inserted successfully.")
            except: # venue, paper_openreview_id, original_openreview_id, modified_openreview_id, content, time
                print(f'''The node in 'revisions' table requires the following node features:
                      venue: str,
                      original_openreview_id: str, 
                      modified_openreview_id: str, 
                      content: dict, 
                      time: str

                      And the node features you provided:
                      {node_features}
                      are not qualified
                      ''')
        else:
            print(f"The table {table} is not exist in this database")
    
    def insert_node_by_csv(self, table: str, csv_file_path: str) -> None:
        df = pd.read_csv(csv_file_path)
        for index, row in df.iterrows():
            node_features = row.to_dict()
            self.insert_node(table, node_features)
            print(f"Node with {table} {index} inserted successfully.")
    
    def insert_node_by_json(self, table: str, json_file_path: str) -> None:
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for idx, node_features in enumerate(dataset):
            self.insert_node(table, node_features)
            print(f"Node with {table} {idx} inserted successfully.")
    
    def delete_node_by_id(self, table: str, primary_key: dict) -> None | pd.DataFrame:
        if table == "papers":
            try:
                return self.db.delete_paper_by_id(**primary_key)
            except: # venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions
                print(f'''The primary key in 'paper' table is
                      
                      paper_openreview_id: str

                      And the primary key you provided:
                      {primary_key}
                      are not qualified
                      ''')
                return None
        elif table == "reviews":
            try:
                return self.db.delete_review_by_id(**primary_key)
            except: # venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time
                print(f'''The node in 'reviews' table requires the following node features:

                      review_openreview_id: str

                      And the primary key you provided:
                      {primary_key}
                      are not qualified
                      ''')
                return None
        elif table == "authors":
            try:
                return self.db.delete_author_by_id(**primary_key)
            except: # venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp
                print(f'''The node in 'authors' table requires the following node features:

                      author_openreview_id: str

                      And the primary key you provided:
                      {primary_key}
                      are not qualified
                      ''')
                return None
        elif table == "revisions":
            try:
                return self.db.delete_revision_by_id(**primary_key)
            except: # venue, paper_openreview_id, original_openreview_id, modified_openreview_id, content, time
                print(f'''The node in 'revisions' table requires the following node features:

                      modified_openreview_id: str, 

                      And the primary key you provided:
                      {primary_key}
                      are not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
    
    def delete_node_by_venue(self, table: str, venue: str) -> None | pd.DataFrame:
        if table == "papers":
            try:
                return self.db.delete_papers_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "reviews":
            try:
                return self.db.delete_reviews_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "authors":
            try:
                return self.db.delete_authors_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "revisions":
            try:
                return self.db.delete_revisions_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
            
    def get_node_features_by_id(self, table: str, primary_key: dict) -> None | pd.DataFrame:
        if table == "papers":
            try:
                return self.db.get_paper_by_id(**primary_key)
            except: # venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions
                print(f'''The primary key in 'paper' table is
                      
                      paper_openreview_id: str

                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        elif table == "reviews":
            try:
                return self.db.get_review_by_id(**primary_key)
            except: # venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time
                print(f'''The node in 'reviews' table requires the following node features:

                      review_openreview_id: str

                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        elif table == "authors":
            try:
                return self.db.get_author_by_id(**primary_key)
            except: # venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp
                print(f'''The node in 'authors' table requires the following node features:

                      author_openreview_id: str

                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        elif table == "revisions":
            try:
                return self.db.get_revision_by_id(**primary_key)
            except: # venue, paper_openreview_id, original_openreview_id, modified_openreview_id, content, time
                print(f'''The node in 'revisions' table requires the following node features:

                      revision_openreview_id: str, 

                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        elif table == "paragraphs":
            try:
                return self.db.get_paragraph_by_id(**primary_key)
            except: # venue, paper_openreview_id, paragraph_number
                print(f'''The node in 'paragraphs' table requires the following node features:

                      paper_openreview_id: str,

                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def get_node_features_by_venue(self, table: str, venue: str) -> None | pd.DataFrame:
        if table == "papers":
            try:
                return self.db.get_papers_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "reviews":
            try:
                return self.db.get_reviews_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "authors":
            try:
                return self.db.get_authors_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "revisions":
            try:
                return self.db.get_revisions_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "paragraphs":
            try:
                return self.db.get_paragraphs_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
    
    def update_node(self, table: str, node_features: dict) -> None | pd.DataFrame:
        if table == "papers":
            try:
                return self.db.update_paper(**node_features)
            except: # venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions
                print(f'''The node in 'papers' table requires the following node features:
                      venue: str,
                      paper_openreview_id: str,
                      title: str, 
                      abstract: str, 
                      author_openreview_ids: set, 
                      author_full_names: set,
                      paper_decision: str,
                      paper_pdf_link: str,
                      revisions: dict

                      And the node features you provided:
                      {node_features}
                      are not qualified
                      ''')
                return None
        elif table == "reviews":
            try:
                return self.db.update_review(**node_features)
            except: # venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time
                print(f'''The node in 'reviews' table requires the following node features:
                      venue: str,
                      paper_openreview_id: str,
                      review_openreview_id: str, 
                      replyto_openreview_id: str, 
                      writer: str, 
                      title: str,
                      content: dict,
                      time: str

                      And the node features you provided:
                      {node_features}
                      are not qualified
                      ''')
                return None
        elif table == "authors":
            try:
                return self.db.update_author(**node_features)
            except: # venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp
                print(f'''The node in 'authors' table requires the following node features:
                      venue: str,
                      author_openreview_id: str,
                      author_full_name: str, 
                      email: str, 
                      affiliation: str,
                      homepage: str,
                      dblp: str

                      And the node features you provided:
                      {node_features}
                      are not qualified
                      ''')
                return None
        elif table == "revisions":
            try:
                return self.db.update_revision(**node_features)
            except: # venue, paper_openreview_id, original_openreview_id, modified_openreview_id, content, time
                print(f'''The node in 'revisions' table requires the following node features:
                      venue: str,
                      paper_openreview_id: str,
                      original_openreview_id: str, 
                      modified_openreview_id: str, 
                      content: dict, 
                      time: str

                      And the node features you provided:
                      {node_features}
                      are not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
            
    def get_all_node_features(self, table: str) -> None | pd.DataFrame:
        if table == "papers":
            return self.db.get_all_papers(is_all_features=True)
        elif table == "reviews":
            return self.db.get_all_reviews(is_all_features=True)
        elif table == "authors":
            return self.db.get_all_authors(is_all_features=True)
        elif table == "revisions":
            return self.db.get_all_revisions(is_all_features=True)
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def get_all_nodes(self, table: str) -> None | pd.DataFrame:
        if table == "papers":
            return self.db.get_all_papers(is_all_features=False)
        elif table == "reviews":
            return self.db.get_all_reviews(is_all_features=False)
        elif table == "authors":
            return self.db.get_all_authors(is_all_features=False)
        elif table == "revisions":
            return self.db.get_all_revisions(is_all_features=False)
        else:
            print(f"The table {table} is not exist in this database")
            return None
    
    # edge
    def get_edge_features_by_id(self, table: str, primary_key: dict) -> None | pd.DataFrame:
        if table == "papers_authors":
            try:
                return self.db.get_paper_author_by_id(**primary_key)
            except:
                print(f'''The primary key in 'papers_authors' table is
                      
                      paper_openreview_id: str,
                      author_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                    ''')
                return None
        elif table == "papers_reviews":
            try:
                return self.db.get_paper_review_by_id(**primary_key)
            except:
                print(f'''The primary key in 'papers_reviews' table is
                      
                      paper_openreview_id: str,
                      review_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        elif table == "papers_revisions":
            try:
                return self.db.get_paper_revision_by_id(**primary_key)
            except:
                print(f'''The primary key in 'papers_revisions' table is
                      
                      paper_openreview_id: str,
                      revision_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        elif table == "revisions_reviews":
            try:
                return self.db.get_revision_review_by_id(**primary_key)
            except:
                print(f'''The primary key in 'revisions_reviews' table is
                      
                      revision_openreview_id: str,
                      review_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def get_edge_features_by_venue(self, table: str, venue: str) -> None | pd.DataFrame:
        if table == "papers_authors":
            try:
                return self.db.get_papers_authors_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "papers_reviews":
            try:
                return self.db.get_papers_reviews_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "papers_revisions":
            try:
                return self.db.get_papers_revisions_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "revisions_reviews":
            try:
                return self.db.get_revisions_reviews_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "openreview_arxiv":
            try:
                return self.db.get_openreview_arxiv_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def get_all_edge_features(self, table: str) -> None | pd.DataFrame:
        if table == "papers_authors":
            return self.db.get_all_papers_authors()
        elif table == "papers_reviews":
            return self.db.get_all_papers_reviews()
        elif table == "papers_revisions":
            return self.db.get_all_papers_revisions()
        elif table == "revisions_reviews":
            return self.db.get_all_revisions_reviews()
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def delete_edge_by_id(self, table: str, primary_key: dict) -> None | pd.DataFrame:
        if table == "papers_authors":
            try:
                return self.db.delete_paper_author_by_id(**primary_key)
            except:
                print(f'''The primary key in 'papers_authors' table is
                      
                      paper_openreview_id: str,
                      author_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                    ''')
                return None
        elif table == "papers_reviews":
            try:
                return self.db.delete_paper_review_by_id(**primary_key)
            except:
                print(f'''The primary key in 'papers_reviews' table is
                      
                      paper_openreview_id: str,
                      review_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        elif table == "papers_revisions":
            try:
                return self.db.delete_paper_revision_by_id(**primary_key)
            except:
                print(f'''The primary key in 'papers_revisions' table is
                      
                      paper_openreview_id: str,
                      revision_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        elif table == "revisions_reviews":
            try:
                return self.db.delete_revision_review_by_id(**primary_key)
            except:
                print(f'''The primary key in 'revisions_reviews' table is
                      
                      revision_openreview_id: str,
                      review_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def delete_edge_by_venue(self, table: str, venue: str) -> None | pd.DataFrame:
        if table == "papers_authors":
            try:
                return self.db.delete_papers_authors_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "papers_reviews":
            try:
                return self.db.delete_papers_reviews_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "papers_revisions":
            try:
                return self.db.delete_papers_revisions_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "revisions_reviews":
            try:
                return self.db.delete_revisions_reviews_by_venue(venue)
            except:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def insert_edge(self, table: str, edge_features: dict) -> None:
        if table == "papers_authors":
            try:
                self.db.insert_paper_authors(**edge_features)
                paper_openreview_id = edge_features["paper_openreview_id"]
                author_openreview_id = edge_features["author_openreview_id"]
                print(f"Paper {paper_openreview_id} and author {author_openreview_id} are connected successfully.")
            except:
                print(f'''The edge in 'papers_authors' table requires the following edge features:
                      venue: str,
                      paper_openreview_id: str,
                      author_openreview_id: str
                      
                      And the edge features you provided:
                      {edge_features}
                      are not qualified
                      ''')
        elif table == "papers_reviews":
            try:
                self.db.insert_paper_reviews(**edge_features)
                paper_openreview_id = edge_features["paper_openreview_id"]
                review_openreview_id = edge_features["review_openreview_id"]
                print(f"Paper {paper_openreview_id} and review {review_openreview_id} are connected successfully.")
            except:
                print(f'''The edge in 'papers_reviews' table requires the following edge features:
                      venue: str,
                      paper_openreview_id: str,
                      review_openreview_id: str,
                      title: str,
                      time: str
                      
                      And the edge features you provided:
                      {edge_features}
                      are not qualified
                      ''')
        elif table == "papers_revisions":
            try:
                self.db.insert_paper_revisions(**edge_features)
                paper_openreview_id = edge_features["paper_openreview_id"]
                revision_openreview_id = edge_features["revision_openreview_id"]
                print(f"Paper {paper_openreview_id} and revision {revision_openreview_id} are connected successfully.")
            except:
                print(f'''The edge in 'papers_reviews' table requires the following edge features:
                      venue: str,
                      paper_openreview_id: str,
                      revision_openreview_id: str,
                      title: str,
                      time: str
                      
                      And the edge features you provided:
                      {edge_features}
                      are not qualified
                      ''')
        elif table == "revisions_reviews":
            try:
                self.db.insert_revision_reviews(**edge_features)
                revision_openreview_id = edge_features["revision_openreview_id"]
                review_openreview_id = edge_features["review_openreview_id"]
                print(f"Revision {revision_openreview_id} and review {review_openreview_id} are connected successfully.")
            except:
                print(f'''The edge in 'revisions_reviews' table requires the following edge features:
                      venue: str,
                      revision_openreview_id: str,
                      review_openreview_id: str,
                      
                      And the edge features you provided:
                      {edge_features}
                      are not qualified
                      ''')
        else:
            print(f"The table {table} is not exist in this database")
            return None
    
    def insert_edge_by_csv(self, table: str, csv_file_path: str) -> None:
        df = pd.read_csv(csv_file_path)
        for index, row in df.iterrows():
            edge_features = row.to_dict()
            self.insert_edge(table, edge_features)
            print(f"Edge with {table} {index} inserted successfully.")
            
    def insert_edge_by_json(self, table: str, json_file_path: str) -> None:
        with open(json_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for idx, edge_features in enumerate(dataset):
            self.insert_edge(table, edge_features)
            print(f"Edge with {table} {idx} inserted successfully.")
    
    def get_neighborhood_by_id(self, table: str, primary_key: dict) -> None | pd.DataFrame:
        if table == "papers_authors":
            try:
                if "paper_openreview_id" in primary_key:
                    return self.db.get_paper_neighboring_authors(**primary_key)
                elif "author_openreview_id" in primary_key:
                    return self.db.get_author_neighboring_papers(**primary_key)
                else:
                    print(f'''To find neighborhood in table 'papers_authors',
                        the primary key should only include
                        
                        paper_openreview_id: str
                        or
                        author_openreview_id: str
                        
                        The primary key you provided
                        {primary_key}
                        is not qualified
                        ''')
                    return None
            except:
                print(f'''To find neighborhood in table 'papers_authors',
                    the primary key should only include
                    
                    paper_openreview_id: str
                    or
                    author_openreview_id: str
                    
                    The primary key you provided
                    {primary_key}
                    is not qualified
                    ''')
                return None
        elif table == "papers_reviews":
            try:
                if "paper_openreview_id" in primary_key:
                    return self.db.get_paper_neighboring_reviews(**primary_key)
                elif "review_openreview_id" in primary_key:
                    return self.db.get_review_neighboring_papers(**primary_key)
                else:
                    print(f'''To find neighborhood in table 'papers_reviews',
                        the primary key should only include
                        
                        paper_openreview_id: str
                        or
                        review_openreview_id: str
                        
                        The primary key you provided
                        {primary_key}
                        is not qualified
                        ''')
                    return None
            except:
                print(f'''To find neighborhood in table 'papers_authors',
                        the primary key should only include
                        
                        paper_openreview_id: str
                        or
                        review_openreview_id: str
                        
                        The primary key you provided
                        {primary_key}
                        is not qualified
                        ''')
                return None
        elif table == "papers_revisions":
            try:
                if "paper_openreview_id" in primary_key:
                    return self.db.get_paper_neighboring_revisions(**primary_key)
                elif "revision_openreview_id" in primary_key:
                    return self.db.get_revision_neighboring_papers(**primary_key)
                else:
                    print(f'''To find neighborhood in table 'papers_revisions',
                        the primary key should only include
                        
                        paper_openreview_id: str
                        or
                        revision_openreview_id: str
                        
                        The primary key you provided
                        {primary_key}
                        is not qualified
                        ''')
                    return None
            except:
                print(f'''To find neighborhood in table 'papers_revisions',
                        the primary key should only include
                        
                        paper_openreview_id: str
                        or
                        revision_openreview_id: str
                        
                        The primary key you provided
                        {primary_key}
                        is not qualified
                        ''')
                return None
        elif table == "revisions_reviews":
            try:
                if "revision_openreview_id" in primary_key:
                    return self.db.get_revision_neighboring_reviews(**primary_key)
                elif "review_openreview_id" in primary_key:
                    return self.db.get_review_neighboring_revisions(**primary_key)
                else:
                    print(f'''To find neighborhood in table 'revisions_reviews',
                        the primary key should only include
                        
                        revision_openreview_id: str
                        or
                        review_openreview_id: str
                        
                        The primary key you provided
                        {primary_key}
                        is not qualified
                        ''')
                    return None
            except:
                print(f'''To find neighborhood in table 'revisions_reviews',
                    the primary key should only include
                    
                    revision_openreview_id: str
                    or
                    review_openreview_id: str
                    
                    The primary key you provided
                    {primary_key}
                    is not qualified
                    ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None