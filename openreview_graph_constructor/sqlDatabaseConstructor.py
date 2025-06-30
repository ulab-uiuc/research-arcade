import openreview
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from database import Database

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