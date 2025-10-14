import openreview
import os
from .pdf_utils import extract_paragraphs_from_pdf_new

class OpenReviewCrawler:
    def __init__(self):
        self.client_v1 = openreview.Client(
            baseurl='https://api.openreview.net'
        )
        
        self.client_v2 = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        )
        
    def crawl_paper_data_from_api(self, venue: str):
        paper_data = []
        if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue:
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='replies')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    # get paper decision
                    decision = None
                    if "2019" in venue:
                        reviews = submission.details["replies"]
                        for review in reviews:
                            if "recommendation" in review['content']:
                                decision = review['content']['recommendation']
                                break
                    elif "2018" in venue or "2020" in venue or "2021" or "2022" in venue or "2023" in venue:
                        reviews = submission.details["replies"]
                        for review in reviews:
                            if "decision" in review['content']:
                                decision = review['content']['decision']
                                break
                    # get paper openreview id
                    paper_id = submission.id
                    # get title
                    title = submission.content["title"]
                    # get abstract
                    abstract = submission.content["abstract"]
                    # get paper's pdf
                    pdf = submission.content["pdf"]
                    paper_data.append({
                            "paper_openreview_id": paper_id,
                            "title": title,
                            "abstract": abstract,
                            "pdf_link": pdf,
                            "paper_decision": decision
                        })
                return paper_data
        elif "2017" in venue or "2014" in venue or "2013" in venue:
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='replies')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    # get paper decision
                    decision = None
                    if "2017" in venue:
                        reviews = submission.details["replies"]
                        for review in reviews:
                            if "decision" in review['content']:
                                decision = review['content']['decision']
                                break
                    # get paper openreview id
                    paper_id = submission.id
                    # get title
                    title = submission.content["title"]
                    # get abstract
                    abstract = submission.content["abstract"]
                    # get paper's pdf
                    pdf = submission.content["pdf"]
                    paper_data.append({
                            "venue": venue,
                            "paper_openreview_id": paper_id,
                            "title": title,
                            "abstract": abstract,
                            "pdf_link": pdf,
                            "paper_decision": decision
                        })
                return paper_data
        else:
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
            if submissions is not None:
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
                        # get title
                        title = submission.content["title"]["value"]
                        # get abstract
                        abstract = submission.content["abstract"]["value"]
                        # get paper's pdf
                        pdf = submission.content["pdf"]["value"]
                        paper_data.append({
                            "venue": venue,
                            "paper_openreview_id": paper_id,
                            "title": title,
                            "abstract": abstract,
                            "pdf_link": pdf,
                            "paper_decision": decision
                        })
                return paper_data
            else:
                print(f"No submissions found for venue: {venue}")
                return []
    
    def crawl_author_data_from_api(self, venue: str):
        author_data = []
        author_set = set()
        if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue:
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission')
        elif "2017" in venue or "2014" in venue or "2013" in venue:
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission')
        else:
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
        if submissions is None:
            print(f"No submissions found for venue: {venue}")
            return []
        else:
            for submission in tqdm(submissions):
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
                    
                author_data.append({
                    "venue": venue,
                    "author_openreview_id": author_id,
                    "author_full_name": fullname,
                    "email": email,
                    "affiliation": affiliation,
                    "homepage": homepage,
                    "dblp": dblp
                })
            return author_data
        
    def crawl_review_data_from_api(self, venue: str):
        review_data = []
        if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue:
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='replies')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    reviews = submission.details["replies"]
                    for review in reviews:
                        # get review openreview id
                        reply_id = review["id"]
                        # get replyto openreview id
                        replyto_id = review["replyto"]
                        # get time
                        time = datetime.fromtimestamp(review['tmdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                        # get writer id
                        writer = review["signatures"][0].split('/')[-1]
                        # get title
                        if "summary_of_the_paper" in review["content"] or "rating" in review["content"]:
                            title = "Official Review by " + writer
                        elif "decision" in review["content"]:
                            title = "Paper Decision"
                        else:
                            if "reviewer" in review["signatures"][0].split('/')[-1].lower():
                                title = "Response by " + writer
                            else:
                                title = "Response by Authors"
                        # get content
                        content = review["content"]
                        review_data.append({
                            "venue": venue,
                            "review_openreview_id": reply_id,
                            "replyto_openreview_id": replyto_id,
                            "title": title,
                            "writer": writer,
                            "content": content,
                            "time": time
                        })
                return review_data
        elif "2017" in venue or "2014" in venue or "2013" in venue:
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='replies')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    reviews = submission.details["replies"]
                    for review in reviews:
                        # get review openreview id
                        reply_id = review["id"]
                        # get replyto openreview id
                        replyto_id = review["replyto"]
                        # get time
                        time = datetime.fromtimestamp(review['tmdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                        # get writer id
                        writer = review["signatures"][0].split('/')[-1]
                        # get title
                        if "rating" in review["content"]:
                            title = "Official Review by " + writer
                        elif "decision" in review["content"]:
                            title = "Paper Decision"
                        else:
                            if "reviewer" in review["signatures"][0].split('/')[-1].lower():
                                title = "Response by " + writer
                            else:
                                title = "Response by Authors"
                        # get content
                        content = review["content"]
                        review_data.append({
                            "venue": venue,
                            "review_openreview_id": reply_id,
                            "replyto_openreview_id": replyto_id,
                            "title": title,
                            "writer": writer,
                            "content": content,
                            "time": time
                        }) 
                return review_data
        else:
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission', details='replies')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    if review.content["venueid"]["value"].split('/')[-1] == "Withdrawn_Submission":
                        continue
                    else:
                        reviews = submission.details["replies"]
                        for review in reviews:
                            # get review openreview id
                            reply_id = review["id"]
                            # get replyto openreview id
                            replyto_id = review["replyto"]
                            # get time
                            time = datetime.fromtimestamp(review['tmdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                            # get writer id
                            writer = review["signatures"][0].split('/')[-1].lower()
                            # get title
                            if "summary" in review["content"]:
                                title = "Official Review by " + writer
                            elif "metareview" in review["content"]:
                                title = "Meta Review by " + writer
                            elif "decision" in review["content"]:
                                title = "Paper Decision"
                            else:
                                if "reviewer" in review["signatures"][0].split('/')[-1].lower():
                                    title = "Response by " + writer
                                else:
                                    title = "Response by Authors"
                            # get content
                            content = review["content"]
                            review_data.append({
                                "venue": venue,
                                "review_openreview_id": reply_id,
                                "replyto_openreview_id": replyto_id,
                                "title": title,
                                "writer": writer,
                                "content": content,
                                "time": time
                            })
                return review_data

    def crawl_revision_data_from_api(self, venue: str, filter_list: list, pdf_dir: str, log_file: str):
        import time
        revision_data = []
        if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue or "2017" in venue or "2014" in venue or "2013" in venue:
            if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='revisions')
            elif "2017" in venue or "2014" in venue or "2013" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='revisions')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
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
                        
                        if num_revision <= 1:
                            continue
                        else:
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
                                        revision_data.append({
                                            "venue": venue,
                                            "original_openreview_id": original_id,
                                            "revision_openreview_id": modified_id,
                                            "content": content,
                                            "time": date
                                            })
                                    except:
                                        continue
                                    self.crawl_paragraph_data_from_pdf(original_pdf, filter_list, log_file, is_pdf_delete=True)
                                    if idx == num_revision - 1:
                                        self.crawl_paragraph_data_from_pdf(modified_pdf, filter_list, log_file, is_pdf_delete=True) 
                return revision_data
        else:
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission', details='revisions')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
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
                                original_id = None
                                modified_id = None
                                
                                for idx, revision in enumerate(sorted_revisions):

                                    original_id = modified_id
                                    modified_id = revision[0]
                                    original_pdf = str(pdf_dir)+str(original_id)+".pdf"
                                    modified_pdf = str(pdf_dir)+str(modified_id)+".pdf"
                                    if idx > 1:
                                        time = revision[1]["Time"]
                                        try:
                                            content = connect_diffs_and_paragraphs(original_pdf, modified_pdf, filter_list)
                                        except:
                                            continue
                                        revision_data.append({
                                            "venue": venue,
                                            "original_openreview_id": original_id,
                                            "revision_openreview_id": modified_id,
                                            "content": content,
                                            "time": date
                                            })
                                        self.crawl_paragraph_data_from_pdf(original_pdf, filter_list, log_file, is_pdf_delete=True)
                                    if idx == num_revision - 1:
                                        self.crawl_paragraph_data_from_pdf(modified_pdf, filter_list, log_file, is_pdf_delete=True)
                return revision_data
    
    def crawl_paragraph_data_from_pdf(self, pdf_path: str, filter_list: list, log_file: str, is_pdf_delete: bool = True):
        paragraph_data = []
        if os.path.exists(pdf_path):
            try:
                structured_content = extract_paragraphs_from_pdf_new(pdf_path, filter_list)    
                paragraph_counter = 1
                for section, paragraphs in structured_content.items():
                    for paragraph in paragraphs:
                        paragraph_data.append({
                            "section": section,
                            "paragraph_idx": paragraph_counter,
                            "content": paragraph
                        })
                        paragraph_counter += 1
                if is_pdf_delete:
                    os.remove(pdf_path)
                    print(f"Deleted PDF file: {pdf_path}")
                return paragraph_data
            except:
                with open(log_file, "a") as log:
                    log.write(f"{pdf_path} Failed\n")
                print(f"Failed to extract content from PDF: {pdf_path}")
                return []
        else:
            with open(log_file, "a") as log:
                log.write(f"{pdf_path} Missing\n")
            print(f"PDF file does not exist: {pdf_path}")
            return []
        
    def crawl_papers_authors_data_from_api(self, venue: str):
        papers_authors_data = []
        if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue or "2017" in venue or "2014" in venue or "2013" in venue:
            if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='replies')
            elif "2017" in venue or "2014" in venue or "2013" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='replies')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    # get paper openreview id
                    paper_id = submission.id
                    # get author openreview ids
                    author_ids = set(submission.content["authorids"])
                    for author_id in author_ids:
                        papers_authors_data.append({
                            "venue": venue,
                            "paper_openreview_id": paper_id,
                            "author_openreview_id": author_id
                        })
                return papers_authors_data
        else:
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
            if submissions is not None:
                for submission in tqdm(submissions):
                    # get paper openreview id
                    paper_id = submission.id
                    # get author openreview ids
                    author_ids = set(submission.content["authorids"])
                    for author_id in author_ids:
                        papers_authors_data.append({
                            "venue": venue,
                            "paper_openreview_id": paper_id,
                            "author_openreview_id": author_id
                        })
                return papers_authors_data
            else:
                print(f"No submissions found for venue: {venue}")
                return []
            
    def crawl_papers_revisions_data_from_api(self, venue: str):
        import time
        papers_revisions_data = []
        if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue or "2017" in venue or "2014" in venue or "2013" in venue:
            if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='revisions')
            elif "2017" in venue or "2014" in venue or "2013" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='revisions')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
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
                            revision_openreview_id = revision.id
                            title = "Paper Revision"
                            date = datetime.fromtimestamp(revision.tmdate / 1000).strftime("%Y-%m-%d %H:%M:%S")
                            papers_revisions_data.append({
                                "venue": venue,
                                "paper_openreview_id": paper_id,
                                "revision_openreview_id": revision_openreview_id,
                                "title": title,
                                "date": date
                            })
                return papers_revisions_data
        else:
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission', details='revisions')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
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
                                revision_openreview_id = note.id
                                title = note.invitation.split('/')[-1]
                                date = datetime.fromtimestamp(note.tmdate / 1000).strftime("%Y-%m-%d %H:%M:%S")
                                papers_revisions_data.append({
                                    "venue": venue,
                                    "paper_openreview_id": paper_id,
                                    "revision_openreview_id": revision_openreview_id,
                                    "title": title,
                                    "date": date
                                })
                return papers_revisions_data

    def crawl_papers_reviews_from_api(self, venue: str):
        papers_reviews_data = []
        if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue:
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission', details='replies')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    reviews = submission.details["replies"]
                    for review in reviews:
                        # get review openreview id
                        reply_id = review["id"]
                        # get replyto openreview id
                        replyto_id = review["replyto"]
                        # get writer id
                        writer = review["signatures"][0].split('/')[-1]
                        # get title
                        if "summary_of_the_paper" in review["content"] or "rating" in review["content"]:
                            title = "Official Review by " + writer
                        elif "decision" in review["content"]:
                            title = "Paper Decision"
                        else:
                            if "reviewer" in review["signatures"][0].split('/')[-1].lower():
                                title = "Response by " + writer
                            else:
                                title = "Response by Authors"
                        papers_reviews_data.append({
                            "venue": venue,
                            "review_openreview_id": reply_id,
                            "replyto_openreview_id": replyto_id,
                            "title": title,
                            "content": content
                        })
                return papers_reviews_data
        elif "2017" in venue or "2014" in venue or "2013" in venue:
            submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission', details='replies')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    reviews = submission.details["replies"]
                    for review in reviews:
                        # get review openreview id
                        reply_id = review["id"]
                        # get replyto openreview id
                        replyto_id = review["replyto"]
                        # get time
                        time = datetime.fromtimestamp(review['tmdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                        # get writer id
                        writer = review["signatures"][0].split('/')[-1]
                        # get title
                        if "rating" in review["content"]:
                            title = "Official Review by " + writer
                        elif "decision" in review["content"]:
                            title = "Paper Decision"
                        else:
                            if "reviewer" in review["signatures"][0].split('/')[-1].lower():
                                title = "Response by " + writer
                            else:
                                title = "Response by Authors"
                        papers_reviews_data.append({
                            "venue": venue,
                            "review_openreview_id": reply_id,
                            "replyto_openreview_id": replyto_id,
                            "title": title,
                            "time": time
                        }) 
                return papers_reviews_data
        else:
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission', details='replies')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    if review.content["venueid"]["value"].split('/')[-1] == "Withdrawn_Submission":
                        continue
                    else:
                        reviews = submission.details["replies"]
                        for review in reviews:
                            # get review openreview id
                            reply_id = review["id"]
                            # get replyto openreview id
                            replyto_id = review["replyto"]
                            # get time
                            time = datetime.fromtimestamp(review['tmdate'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                            # get writer id
                            writer = review["signatures"][0].split('/')[-1].lower()
                            # get title
                            if "summary" in review["content"]:
                                title = "Official Review by " + writer
                            elif "metareview" in review["content"]:
                                title = "Meta Review by " + writer
                            elif "decision" in review["content"]:
                                title = "Paper Decision"
                            else:
                                if "reviewer" in review["signatures"][0].split('/')[-1].lower():
                                    title = "Response by " + writer
                                else:
                                    title = "Response by Authors"
                            papers_reviews_data.append({
                                "venue": venue,
                                "review_openreview_id": reply_id,
                                "replyto_openreview_id": replyto_id,
                                "title": title,
                                "time": time
                            })
                return papers_reviews_data
                
    def crawl_openreview_arxiv_data_from_api(self, venue: str):
        openreview_arxiv_data = []
        if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue or "2017" in venue or "2014" in venue or "2013" in venue:
            if "2023" in venue or "2022" in venue or "2021" in venue or "2020" in venue or "2019" in venue or "2018" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/Blind_Submission')
            elif "2017" in venue or "2014" in venue or "2013" in venue:
                submissions = self.client_v1.get_all_notes(invitation=f'{venue}/-/submission')
            if submissions is None:
                print(f"No submissions found for venue: {venue}")
                return []
            else:
                for submission in tqdm(submissions):
                    # get paper openreview id
                    openreview_id = submission.id
                    # get title
                    title = submission.content["title"]
                    # get arxiv id if exists
                    arxiv_id = self._search_title_with_name(title)
                    openreview_arxiv_data.append({
                        "venue": venue,
                        "paper_openreview_id": openreview_id,
                        "arxiv_id": arxiv_id,
                        "title": title
                    })
                return openreview_arxiv_data
        else:
            submissions = self.client_v2.get_all_notes(invitation=f'{venue}/-/Submission')
            if submissions is not None:
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
                        # get title
                        title = submission.content["title"]["value"]
                        # get arxiv id if exists
                        arxiv_id = self._search_title_with_name(title)
                        openreview_arxiv_data.append({
                            "venue": venue,
                            "paper_openreview_id": paper_id,
                            "arxiv_id": arxiv_id,
                            "title": title
                        })
                return openreview_arxiv_data
            else:
                print(f"No submissions found for venue: {venue}")
                return []
                
    def _title_cleaner(self, title: str):
        """
        Remove all symbols (non-alphanumeric, non-space characters) from the title.
        Collapses multiple spaces down to one and trims ends.
        """
        # Remove anything that isn't a letter, number, or whitespace
        cleaned = re.sub(r'[^A-Za-z0-9\s]', '', title)
        # Collapse multiple spaces and strip leading/trailing spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.strip().lower()
    
    def _search_title_with_name(self, title, max_result=5):
        query = f"ti:{title}"
        search = arxiv.Search(
            query=query,
            max_results=max_result,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        try:
            for result in search.results():
                if (title_cleaner(result.title) == title_cleaner(title)):
                    return result.entry_id
        except UnexpectedEmptyPageError:
            return None