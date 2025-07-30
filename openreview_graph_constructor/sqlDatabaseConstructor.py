import openreview
import arxiv
import os
import re
from arxiv import UnexpectedEmptyPageError
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from database import Database
from pdf_utils import get_pdf, connect_diffs_and_paragraphs

class sqlDatabaseConstructor:
    def __init__(self):
        self.client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net'
        )
        self.db = Database()

    def construct_review_table(self, venue_id):
        # create sql table
        self.db.create_review_table()
        # get all reviews
        reviews = self.client.get_all_notes(invitation=f'{venue_id}/-/Submission', details='replies')
        # remove withdrawn papers
        for review in tqdm(reviews):
            decision = review.content["venueid"]["value"].split('/')[-1]
            if decision == "Withdrawn_Submission":
                continue
            else:
                # get paper openreview id
                paper_id = review.id
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
                        content = {
                            "Meta Review": reply["content"]["metareview"]["value"],
                            "Additional Comments On Reviewer Discussion": reply["content"]["additional_comments_on_reviewer_discussion"]["value"],
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
                    self.db.insert_review(venue_id, paper_id, reply_id, replyto_id, writer, title, content, time)
                        
    def construct_author_table(self, venue_id): # author_list contains authors' openreview ids
        # create sql table
        self.db.create_author_table()
        
        author_set = set()
        # retrieve all the authors in this venue, skip the authors in the withdrawn submissions
        submissions = self.client.get_all_notes(invitation=f'{venue_id}/-/Submission')
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
        author_profiles = openreview.tools.get_profiles(self.client, author_set)
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
                else:
                    if name.get("username") is not None:
                        author_id = name["username"]
                        try:
                            fullname = name["fullname"]
                        except:
                            pass
                    else:
                        break
            if author_id == "": # remove the author with no username
                continue
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
            self.db.insert_author(venue_id, author_id, fullname, email, affiliation, homepage, dblp)
            
    def construct_paper_table(self, venue_id):
        # create sql table
        self.db.create_papers_table()
        # get all submissions
        submissions = self.client.get_all_notes(invitation=f'{venue_id}/-/Submission')
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
                # get revisions and their time
                revisions = {}
                note_edits = self.client.get_note_edits(note_id=paper_id)
                try:
                    for note in note_edits:
                        revisions[note.id] = {
                            "Time": datetime.fromtimestamp(note.cdate / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                            "Title": note.invitation.split('/')[-1]
                        }
                except:
                    pass
                # get title
                title = submission.content["title"]["value"]
                # get abstract
                abstract = submission.content["abstract"]["value"]
                # get author openreview ids
                author_ids = submission.content["authorids"]["value"]
                # get author full names
                fullnames = submission.content["authors"]["value"]
                # get paper's pdf
                pdf = submission.content["pdf"]["value"]
                # add to paper table
                self.db.insert_paper(venue_id, paper_id, title, abstract, author_ids, fullnames, decision, pdf, revisions)
    
    def construct_revision_table(self, venue_id, filter_list, pdf_dir):
        # create sql table
        self.db.create_revisions_table()
        # get all submissions
        submissions = self.client.get_all_notes(invitation=f'{venue_id}/-/Submission')
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
                note_edits = self.client.get_note_edits(note_id=paper_id)
                try:
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
                                content = connect_diffs_and_paragraphs(original_pdf, modified_pdf, filter_list)
                                
                                self.db.insert_revision(venue_id, paper_id, original_id, modified_id, content, time)
                                if os.path.exists(original_pdf):
                                    os.remove(original_pdf)
                                    print(original_pdf+" Deleted")
                                else:
                                    print("File not exist")
                                if idx == num_revision - 1:
                                    if os.path.exists(modified_pdf):
                                        os.remove(modified_pdf)
                                        print(modified_pdf+" Deleted")
                                    else:
                                        print("File not exist")
                except:
                    pass
    
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

        # print("Title of Cited Paper:")
        # print(title)
        try:
            # print("Result:")
            for result in search.results():
                # print(result.title)
                if (self._title_cleaner(result.title) == self._title_cleaner(title)):
                    return result.entry_id
        except UnexpectedEmptyPageError:
            return None
    
    def construct_openreview_arxiv_table(self, venue_id):
        # create sql table
        self.db.create_openreview_arxiv_table()
        
        # get papers from papers table
        # submissions = self.db.get_papers()
        
        # get papers through openreview api
        submissions = self.client.get_all_notes(invitation=f'{venue_id}/-/Submission')
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
    
    def construct_revisions_reviews_table(self):
        # create sql table
        self.db.create_revisions_reviews_table()
        
        # get reviews
        reviews = self.db.get_all_reviews() # id, venue, paper_openreview_id, review_openreview_id, replyto_openreview_id, time 
        reviews_df = pd.DataFrame(reviews, columns=["id", "venue", "paper_openreview_id", "review_openreview_id", "replyto_openreview_id", "time"])
        
        # get revisions
        revisions = self.db.get_all_revisions() # id, venue, paper_openreview_id, original_openreivew_id, modified_openreview_id, time
        revisions_df = pd.DataFrame(revisions, columns=["id", "venue", "paper_openreview_id", "original_openreivew_id", "modified_openreview_id", "time"])

        # get all unique paper_id in revisions_df
        unique_paper_ids = revisions_df['paper_openreview_id'].unique()

        # match revisions with reviews
        for paper_id in tqdm(unique_paper_ids):
            filtered_revisions_df = revisions_df[revisions_df['paper_openreview_id'] == paper_id]
            filtered_reviews_df = reviews_df[reviews_df['paper_openreview_id'] == paper_id]
            start_idx = 0
            for revision in filtered_revisions_df.itertuples():
                venue_id = revision.venue
                original_id = revision.original_openreivew_id
                modified_id = revision.modified_openreview_id
                time = revision.time
                reviews = []
                revision_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                for review in filtered_reviews_df.iloc[start_idx:].itertuples():
                    review_time = datetime.strptime(review.time, "%Y-%m-%d %H:%M:%S")
                    if review_time <= revision_time:
                        review_example = {
                            "review_openreview_id": review.review_openreview_id,
                            "replyto_openreview_id": review.replyto_openreview_id,
                            "time": review.time
                        }
                        reviews.append(review_example)
                        start_idx += 1
                    else:
                        break
                self.db.insert_revision_reviews(venue_id, paper_id, original_id, modified_id, reviews, time)