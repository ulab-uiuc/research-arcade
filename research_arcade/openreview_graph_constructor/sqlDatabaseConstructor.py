import openreview
import arxiv
import os
import re
from arxiv import UnexpectedEmptyPageError
from datetime import datetime
from tqdm import tqdm
from database import Database
from pdf_utils import connect_diffs_and_paragraphs, extract_paragraphs_from_pdf_new

class sqlDatabaseConstructor:
    def __init__(self):
        self.client_v1 = openreview.Client(
            baseurl='https://api.openreview.net'
        )
        
        self.client_v2 = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        )
        
        self.db = Database()

    def construct_review_table(self, venue, version = "v2"):
        # create sql table
        self.db.create_review_table()
        if version == "v2":
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
                        time = datetime.fromtimestamp(reply['cdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
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
                            except Exception:
                                pass
                            try:
                                additional_comments = reply["content"]["additional_comments_on_reviewer_discussion"]["value"]
                            except Exception:
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
                            except Exception:
                                comment = ""
                            content = {
                                "Decision": reply["content"]["decision"]["value"],
                                "Comment": comment,
                            }
                        self.db.insert_review(venue, reply_id, replyto_id, writer, title, content, time)
        elif version == "v1":
            # get all reviews
            reviews = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='replies')
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
                    time = datetime.fromtimestamp(reply['cdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                    # classify the type of comment
                    if reply["content"].get("summary_of_the_paper") is not None: # reviewer initial comment
                        title = "Official Review by " + reply["signatures"][0].split('/')[-1]
                        content = {
                            "Summary of the Paper": reply["content"]["summary_of_the_paper"],
                            "Strength and Weaknesses": reply["content"]["strength_and_weaknesses"],
                            "Clarity, quality, novelty and reproducibility": reply["content"]["clarity,_quality,_novelty_and_reproducibility"],
                            "Summary of the review": reply["content"]["summary_of_the_review"],
                            "Correctness": reply["content"]["correctness"],
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
                        content = {
                            "Decision": reply["content"]["decision"],
                            "Metareview: summary, strengths and weaknesses": reply["content"]["metareview:_summary,_strengths_and_weaknesses"],
                            "Justification for why not higher score": reply["content"]["justification_for_why_not_higher_score"],
                            "Justification for why not lower score": reply["content"]["justification_for_why_not_lower_score"]
                        }
                    self.db.insert_review(venue, reply_id, replyto_id, writer, title, content, time)
                        
    def construct_author_table(self, venue, version = "v2"): # author_list contains authors' openreview ids
        # create sql table
        self.db.create_author_table()
        
        if version == "v2":
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
                        if name["preferred"]:
                            author_id = name["username"]
                            fullname = name["fullname"]
                            break
                    else:
                        if name.get("username") is not None:
                            author_id = name["username"]
                            try:
                                fullname = name["fullname"]
                                break
                            except Exception:
                                fullname = author_id
                        else:
                            pass
                if author_id == "": # remove the author with no username
                    author_id = profile.id
                # get email
                try:
                    email = profile.content["preferredEmail"]
                except Exception:
                    try:
                        email = profile.content["emailsConfirmed"][0]
                    except Exception:
                        try:
                            email = profile.content["emails"][0]
                        except Exception:
                            email = ""
                # get affiliation
                try:
                    affiliation = profile.content["history"][0]["institution"]["name"]
                except Exception:
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
        elif version == "v1":
            author_set = set()
            # retrieve all the authors in this venue, skip the authors in the withdrawn submissions
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission')
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
                        if name["preferred"]:
                            author_id = name["username"]
                            fullname = name["fullname"]
                            break
                    else:
                        if name.get("username") is not None:
                            author_id = name["username"]
                            try:
                                fullname = name["fullname"]
                                break
                            except Exception:
                                fullname = author_id
                        else:
                            pass
                if author_id == "": # remove the author with no username
                    author_id = profile.id
                # get email
                try:
                    email = profile.content["preferredEmail"]
                except Exception:
                    try:
                        email = profile.content["emailsConfirmed"][0]
                    except Exception:
                        try:
                            email = profile.content["emails"][0]
                        except Exception:
                            email = ""
                # get affiliation
                try:
                    affiliation = profile.content["history"][0]["institution"]["name"]
                except Exception:
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
                
            
    def construct_paper_table(self, venue, filter_list, pdf_dir = "/data/jingjunx/openreview_pdfs/", log_file="not_found_papers_2024.txt", version="v2"):
        # create sql table
        self.db.create_papers_table()
        self.db.create_paragraphs_table()
        
        if version == "v2":
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
                        except Exception:
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
                    #             "Time": datetime.fromtimestamp(note.cdate / 1000).strftime("%Y-%m-%d %H:%M:%S"),
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
        elif version == "v1":
            # get all submissions
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission')
            for submission in tqdm(submissions):
                # get paper decision and remove withdrawn papers
                try:
                    decision = submission.content["venue"]
                except Exception:
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
                    except Exception:
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
    
    def construct_revision_table(self, venue, filter_list, pdf_dir = "/data/jingjunx/openreview_pdfs/", log_file="not_found_pdfs.txt", version = "v2"):
        # create sql table
        self.db.create_revisions_table()
        self.db.create_paragraphs_table()
        if version == "v2":
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
                    if len(note_edits) <= 1:
                        continue
                    else:
                        for note in note_edits:
                            revisions[note.id] = {
                                "Time": datetime.fromtimestamp(note.cdate / 1000).strftime("%Y-%m-%d %H:%M:%S"),
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
                                    except Exception:
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
                                        except Exception:
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
                                            except Exception:
                                                with open(log_file, "a") as log:
                                                    log.write(f"{modified_pdf}\n")
                                        else:
                                            with open(log_file, "a") as log:
                                                log.write(f"{modified_pdf}\n")
                                            print("File not exist")
        elif version == "v1":
            # get all submissions
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='revisions')
            for submission in tqdm(submissions):
                # get paper openreview id
                paper_id = submission.id
                # get revisions and their time
                revisions = {}
                # get revisions and their time
                note_edits = self.client_v1.get_references(referent=paper_id, original=True)
                if len(note_edits) <= 1:
                    continue
                else:
                    filtered_notes = []
                    for note in note_edits:
                        if "pdf" in note.content:
                            filtered_notes.append(note)
                    if len(filtered_notes) <= 1:
                        continue
                    else:
                        for note in filtered_notes:
                            revisions[note.id] = {
                                "Time": datetime.fromtimestamp(note.cdate / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                                "Title": "Paper Revision"
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
                                    except Exception:
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
                                        except Exception:
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
                                            except Exception:
                                                with open(log_file, "a") as log:
                                                    log.write(f"{modified_pdf}\n")
                                        else:
                                            with open(log_file, "a") as log:
                                                log.write(f"{modified_pdf}\n")
                                            print("File not exist")
    
    def construct_papers_authors_table(self, venue, version = "v2"):
        # create sql table
        self.db.create_papers_authors_table()
        if version == "v2":
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
        elif version == "v1":
            # get all submissions
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission')
            for submission in tqdm(submissions):
                paper_id = submission.id
                # get author openreview ids
                author_ids = set(submission.content["authorids"])
                # add to papers authors table
                for author_id in author_ids:
                    self.db.insert_paper_authors(venue, paper_id, author_id)
                
    def construct_papers_revisions_table(self, venue, version = "v2"):
        # create sql table
        self.db.create_papers_revisions_table()
        if version == "v2":
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
                    if len(note_edits) <= 1:
                        continue
                    else:
                        for note in note_edits:
                            revision_openreview_id = note.id
                            title = note.invitation.split('/')[-1]
                            time = datetime.fromtimestamp(note.cdate / 1000).strftime("%Y-%m-%d %H:%M:%S")
                            self.db.insert_paper_revisions(venue, paper_id, revision_openreview_id, title, time)
        elif version == "v1":
            # get all submissions
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='revisions')
            for submission in tqdm(submissions):
                # get paper openreview id
                paper_id = submission.id
                # get revisions
                revisions = self.client_v1.get_references(referent=paper_id, original=True)
                if len(revisions) <= 1:
                    continue
                else:
                    filtered_revisions = []
                    for revision in revisions:
                        if "pdf" in revision.content:
                            filtered_revisions.append(revision)
                    if len(filtered_revisions) <= 1:
                        continue
                    else:
                        revision_openreview_id = revision.id
                        title = "Paper Revision"
                        time = datetime.fromtimestamp(revision.cdate / 1000).strftime("%Y-%m-%d %H:%M:%S")
                        self.db.insert_paper_revisions(venue, paper_id, revision_openreview_id, title, time)
                
    def construct_papers_reviews_table(self, venue, version = "v2"):
        # create sql table
        self.db.create_papers_reviews_table()
        if version == "v2":
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
                        time = datetime.fromtimestamp(reply['cdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
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
        elif version == "v1":
            # get all reviews
            reviews = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='replies')
            # remove withdrawn papers
            for review in tqdm(reviews):
                paper_id = review.id
                # get all the replies
                replies = review.details["replies"]
                for reply in replies:
                    # get review openreview id
                    reply_id = reply["id"]
                    # get time
                    time = datetime.fromtimestamp(reply['cdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                    # classify the type of comment
                    if reply["content"].get("summary_of_the_paper") is not None: # reviewer initial comment
                        title = "Official Review by " + reply["signatures"][0].split('/')[-1]
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
    
    def construct_openreview_arxiv_table(self, venue):
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
    
    def construct_papers_revisions_reviews_table(self):
        # create sql table
        self.db.create_papers_revisions_reviews_table()
        
        # get reviews
        papers_reviews_df = self.db.get_all_papers_reviews()
        
        # get revisions
        paper_revisions_df = self.db.get_all_papers_revisions()

        # get all unique paper_id in revisions_df
        unique_paper_ids = paper_revisions_df['paper_openreview_id'].unique()

        # match revisions with reviews
        for paper_id in tqdm(unique_paper_ids):
            revisions = paper_revisions_df[paper_revisions_df['paper_openreview_id'] == paper_id]
            revisions_sorted = revisions.sort_values(by='time', ascending=True)
            
            reviews = papers_reviews_df[papers_reviews_df['paper_openreview_id'] == paper_id]
            reviews_sorted = reviews.sort_values(by='time', ascending=True)
            
            start_idx = 0
            for revision in revisions_sorted.itertuples():
                venue = revision.venue
                
                revision_id = revision.revision_openreview_id
                revision_time = revision.time
                _revision_time = datetime.strptime(revision_time, "%Y-%m-%d %H:%M:%S")
                for review in reviews_sorted.iloc[start_idx:].itertuples():
                    review_id = review.review_openreview_id
                    review_time = review.time
                    _review_time = datetime.strptime(review_time, "%Y-%m-%d %H:%M:%S")
                    if _review_time <= _revision_time:
                        self.db.insert_paper_revision_review(venue, paper_id, revision_id, review_id, revision_time, review_time)
                        # print(venue, paper_id, revision_id, review_id, revision_time, review_time)
                        start_idx += 1
                    else:
                        break
    
    # node          
    def insert_node(self, table: str, node_features: dict):
        if table == "papers":
            try:
                self.db.insert_paper(**node_features)
                paper_openreview_id = node_features["paper_openreview_id"]
                print(f"Paper with paper_openreview_id {paper_openreview_id} inserted successfully.")
            except Exception: # venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions
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
            except Exception: # venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time
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
            except Exception: # venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp
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
            except Exception: # venue, paper_openreview_id, original_openreview_id, modified_openreview_id, content, time
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
    
    def delete_node_by_id(self, table: str, primary_key: dict):
        if table == "papers":
            try:
                return self.db.delete_paper_by_id(**primary_key)
            except Exception: # venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions
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
            except Exception: # venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time
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
            except Exception: # venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp
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
            except Exception: # venue, paper_openreview_id, original_openreview_id, modified_openreview_id, content, time
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
    
    def delete_node_by_venue(self, table: str, venue: str):
        if table == "papers":
            try:
                return self.db.delete_papers_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "reviews":
            try:
                return self.db.delete_reviews_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "authors":
            try:
                return self.db.delete_authors_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "revisions":
            try:
                return self.db.delete_revisions_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
            
    def get_node_features_by_id(self, table: str, primary_key: dict):
        if table == "papers":
            try:
                return self.db.get_paper_by_id(**primary_key)
            except Exception: # venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions
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
            except Exception: # venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time
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
            except Exception: # venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp
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
            except Exception: # venue, paper_openreview_id, original_openreview_id, modified_openreview_id, content, time
                print(f'''The node in 'revisions' table requires the following node features:

                      modified_openreview_id: str, 

                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def get_node_features_by_venue(self, table: str, venue: str):
        if table == "papers":
            try:
                return self.db.get_papers_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "reviews":
            try:
                return self.db.get_reviews_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "authors":
            try:
                return self.db.get_authors_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "revisions":
            try:
                return self.db.get_revisions_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
    
    def update_node(self, table: str, node_features: dict):
        if table == "papers":
            try:
                return self.db.update_paper(**node_features)
            except Exception: # venue, paper_openreview_id, title, abstract, author_openreview_ids, author_full_names, paper_decision, paper_pdf_link, revisions
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
            except Exception: # venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, writer, title, content, time
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
            except Exception: # venue, author_openreview_id, author_full_name, email, affiliation, homepage, dblp
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
            except Exception: # venue, paper_openreview_id, original_openreview_id, modified_openreview_id, content, time
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
            
    def get_all_node_features(self, table: str):
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
        
    def get_all_nodes(self, table: str):
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
    def get_edge_features_by_id(self, table: str, primary_key: dict):
        if table == "papers_authors":
            try:
                return self.db.get_paper_author_by_id(**primary_key)
            except Exception:
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
            except Exception:
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
            except Exception:
                print(f'''The primary key in 'papers_revisions' table is
                      
                      paper_openreview_id: str,
                      revision_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def get_edge_features_by_venue(self, table: str, venue: str):
        if table == "papers_authors":
            try:
                return self.db.get_papers_authors_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "papers_reviews":
            try:
                return self.db.get_papers_reviews_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "papers_revisions":
            try:
                return self.db.get_papers_revisions_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def get_all_edge_features(self, table: str):
        if table == "papers_authors":
            return self.db.get_all_papers_authors()
        elif table == "papers_reviews":
            return self.db.get_all_papers_reviews()
        elif table == "papers_revisions":
            return self.db.get_all_papers_revisions()
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def delete_edge_by_id(self, table: str, primary_key: dict):
        if table == "papers_authors":
            try:
                return self.db.delete_paper_author_by_id(**primary_key)
            except Exception:
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
            except Exception:
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
            except Exception:
                print(f'''The primary key in 'papers_revisions' table is
                      
                      paper_openreview_id: str,
                      revision_openreview_id: str
                      
                      And the primary key you provided:
                      {primary_key}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def delete_edge_by_venue(self, table: str, venue: str):
        if table == "papers_authors":
            try:
                return self.db.delete_papers_authors_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "papers_reviews":
            try:
                return self.db.delete_papers_reviews_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        elif table == "papers_revisions":
            try:
                return self.db.delete_papers_revisions_by_venue(venue)
            except Exception:
                print(f'''The venue you provided:
                      {venue}
                      is not qualified
                      ''')
                return None
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def insert_edge(self, table: str, edge_features: dict):
        if table == "papers_authors":
            try:
                self.db.insert_paper_authors(**edge_features)
                paper_openreview_id = edge_features["paper_openreview_id"]
                author_openreview_id = edge_features["author_openreview_id"]
                print(f"Paper {paper_openreview_id} and author {author_openreview_id} are connected successfully.")
            except Exception:
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
            except Exception:
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
            except Exception:
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
        else:
            print(f"The table {table} is not exist in this database")
            return None
        
    def get_neighborhood_by_id(self, table: str, primary_key: dict):
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
            except Exception:
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
            except Exception:
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
            except Exception:
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
        else:
            print(f"The table {table} is not exist in this database")
            return None