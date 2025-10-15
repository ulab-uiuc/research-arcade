import sys
from pathlib import Path
# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from research_arcade import ResearchArcade 

db_type = "sql"
config = {
    "host": "localhost",
    "dbname": "iclr_openreview_database",
    "user": "jingjunx",
    "password": "",
    "port": "5432"
}

research_arcade = ResearchArcade(db_type=db_type, config=config)

########## openreview_authors ##########
# get_all_node_features
# openreview_authors_df = research_arcade.get_all_node_features("openreview_authors")
# print(len(openreview_authors_df))

# get_node_features_by_id
# author_id = {"author_openreview_id": "~ishmam_zabir1"}
# author_features = research_arcade.get_node_features_by_id("openreview_authors", author_id)
# print(author_features.to_dict(orient="records")[0])

# delete_node
# author_id = {"author_openreview_id": "~ishmam_zabir1"}
# author_features = research_arcade.delete_node_by_id("openreview_authors", author_id)
# print(author_features.to_dict(orient="records")[0])

# insert_node
# new_author = {'venue': 'ICLR.cc/2025/Conference', 
#               'author_openreview_id': '~ishmam_zabir1', 
#               'author_full_name': 'ishmam zabir', 
#               'email': '****@microsoft.com', 
#               'affiliation': 'Microsoft', 
#               'homepage': 'https://scholar.google.com/citations?user=X7bjzrUAAAAJ&hl=en&oi=ao', 
#               'dblp': ''}
# research_arcade.insert_node("openreview_authors", node_features=new_author)

# update_node
# new_author = {'venue': 'ICLR.cc/2025/Conference', 
#               'author_openreview_id': '~ishmam_zabir1', 
#               'author_full_name': 'ishmam zabir', 
#               'email': '****@microsoft.com', 
#               'affiliation': 'Microsoft', 
#               'homepage': 'https://scholar.google.com/citations?user=X7bjzrUAAAAJ&hl=en&oi=ao', 
#               'dblp': ''}
# author_id = {"author_openreview_id": "~ishmam_zabir1"}
# author_features = research_arcade.get_node_features_by_id("openreview_authors", author_id)
# print(author_features.to_dict(orient="records")[0])
# research_arcade.update_node("openreview_authors", node_features=new_author)
# author_id = {"author_openreview_id": "~ishmam_zabir1"}
# author_features = research_arcade.get_node_features_by_id("openreview_authors", author_id)
# print(author_features.to_dict(orient="records")[0])

########## openreview_papers ##########
# get_all_node_features
# openreview_papers_df = research_arcade.get_all_node_features("openreview_papers")
# print(len(openreview_papers_df))

# get_node_features_by_id
# paper_id = {"paper_openreview_id": "zGej22CBnS"}
# paper_features = research_arcade.get_node_features_by_id("openreview_papers", paper_id)
# print(paper_features.to_dict(orient="records")[0])

# delete_node
# paper_id = {"paper_openreview_id": "zGej22CBnS"}
# paper_features = research_arcade.delete_node_by_id("openreview_papers", paper_id)
# print(paper_features.to_dict(orient="records")[0])

# insert_node
# paper_features = {'venue': 'ICLR.cc/2025/Conference', 
#                   'paper_openreview_id': 'zGej22CBnS', 
#                   'title': 'Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles', 
#                   'abstract': "Tokenization is associated with many poorly understood shortcomings in language models (LMs), yet remains an important component for long sequence scaling purposes. This work studies  how tokenization impacts  model performance by analyzing and comparing the stochastic behavior of tokenized models with their byte-level, or token-free, counterparts. We discover that, even when the two models are statistically equivalent, their predictive distributions over the next byte can be substantially different, a phenomenon we term as ``tokenization bias''. To fully characterize this phenomenon, we  introduce the Byte-Token Representation Lemma, a framework that establishes a mapping between the learned token distribution and its equivalent byte-level distribution.  From this result, we develop a next-byte sampling algorithm  that eliminates tokenization bias without requiring further training or optimization. In other words, this enables zero-shot conversion of tokenized LMs into statistically equivalent token-free ones. We demonstrate its broad applicability with two use cases: fill-in-the-middle (FIM) tasks and model ensembles. In FIM tasks where input prompts may terminate mid-token, leading to out-of-distribution tokenization, our method mitigates performance degradation and achieves 18\\% improvement in FIM coding benchmarks, while consistently outperforming the standard token healing fix. For model ensembles where each model employs a distinct vocabulary, our approach enables seamless integration, resulting in improved performance up to 3.7\\% over individual models across various standard baselines in reasoning, knowledge, and coding. Code is available at:https: //github.com/facebookresearch/Exact-Byte-Level-Probabilities-from-Tokenized-LMs.", 
#                   'paper_decision': 'ICLR 2025 Poster', 
#                   'paper_pdf_link': '/pdf/cdd2212a20c4034029874cba11a05e081bfdb83e.pdf'}
# research_arcade.insert_node("openreview_papers", node_features=paper_features)

# update_node
# new_paper_features = {'venue': 'ICLR.cc/2025/Conference', 
#                   'paper_openreview_id': 'zGej22CBnS', 
#                   'title': 'Exact Byte-Level Probabilities from Tokenized Language Models for FIM-Tasks and Model Ensembles', 
#                   'abstract': "Tokenization is associated with many poorly understood shortcomings in language models (LMs), yet remains an important component for long sequence scaling purposes. This work studies  how tokenization impacts  model performance by analyzing and comparing the stochastic behavior of tokenized models with their byte-level, or token-free, counterparts. We discover that, even when the two models are statistically equivalent, their predictive distributions over the next byte can be substantially different, a phenomenon we term as ``tokenization bias''. To fully characterize this phenomenon, we  introduce the Byte-Token Representation Lemma, a framework that establishes a mapping between the learned token distribution and its equivalent byte-level distribution.  From this result, we develop a next-byte sampling algorithm  that eliminates tokenization bias without requiring further training or optimization. In other words, this enables zero-shot conversion of tokenized LMs into statistically equivalent token-free ones. We demonstrate its broad applicability with two use cases: fill-in-the-middle (FIM) tasks and model ensembles. In FIM tasks where input prompts may terminate mid-token, leading to out-of-distribution tokenization, our method mitigates performance degradation and achieves 18\\% improvement in FIM coding benchmarks, while consistently outperforming the standard token healing fix. For model ensembles where each model employs a distinct vocabulary, our approach enables seamless integration, resulting in improved performance up to 3.7\\% over individual models across various standard baselines in reasoning, knowledge, and coding. Code is available at:https: //github.com/facebookresearch/Exact-Byte-Level-Probabilities-from-Tokenized-LMs.", 
#                   'paper_decision': 'ICLR 2025 Poster', 
#                   'paper_pdf_link': '/pdf/cdd2212a20c4034029874cba11a05e081bfdb83e.pdf'}
# paper_id = {"paper_openreview_id": "zGej22CBnS"}
# paper_features = research_arcade.get_node_features_by_id("openreview_papers", paper_id)
# print(paper_features.to_dict(orient="records")[0])
# research_arcade.update_node("openreview_papers", node_features=new_paper_features)
# paper_id = {"paper_openreview_id": "zGej22CBnS"}
# paper_features = research_arcade.get_node_features_by_id("openreview_papers", paper_id)
# print(paper_features.to_dict(orient="records")[0])

########## openreview_reviews ##########
# get_all_node_features
# openreview_reviews_df = research_arcade.get_all_node_features("openreview_reviews")
# print(len(openreview_reviews_df))

# get_node_features_by_id
# review_id = {"review_openreview_id": "DHwZxFryth"}
# review_features = research_arcade.get_node_features_by_id("openreview_reviews", review_id)
# print(review_features.to_dict(orient="records")[0])

# delete_node
# review_id = {"review_openreview_id": "DHwZxFryth"}
# review_features = research_arcade.delete_node_by_id("openreview_reviews", review_id)
# print(review_features.to_dict(orient="records")[0])

# insert_node
# review_features = {'venue': 'ICLR.cc/2025/Conference', 
#                    'review_openreview_id': 'DHwZxFryth', 
#                    'replyto_openreview_id': 'Yqbllggrmw', 
#                    'writer': 'Authors', 
#                    'title': 'Response by Authors', 
#                    'content': {'Title': 'Response to Reviewer 7i95 (1/2)', 'Comment': '> The method does not improve much in the AlpacaEval 2.0 Score. The author should give a detailed explanation. And why not use metrics like length-controlled win rate?**Response:** Thank you for your careful observation and question. We would like to clarify that we are already using the length-controlled (LC) AlpacaEval 2.0 win-rate metric in our evaluations. We will make this clearer in the table header of Table 3.Regarding the fact that the AlpacaEval 2.0 scores on LLama-3 (8B) do not improve compared to the baselines, we believe this is because our base model, the instruction-finetuned LLama-3 (8B), is already trained to perform exceptionally well in terms of helpfulness, which is the focus of the AlpacaEval benchmark. Additionally, the preference dataset we used, UltraFeedback, may not provide significant further enhancement in the helpfulness aspect. This is supported by the slight decrease observed in the AlpacaEval score for the standard DPO baseline as well (see Table 3, results on LLama-3). Therefore, we think these AlpacaEval 2.0 results on LLama-3 (8B) may not indicate that SAIL is ineffective; it may be simply caused by an ill-suited combination of base model, finetuning dataset, and evaluation benchmark.We also further conducted experiments on the Zephyr (7B) model as the backbone, whose AlpacaEval 2.0 win-rate is lower. We still train on the UltraFeedback preference dataset and the other experiment setups are unchanged. In this experiment, we see a larger improvement of the SAIL method compared to the standard DPO baseline (Zephyr-7B-Beta).|             | AlpacaEval 2.0 (LC) Win-Rate ||--------------------|------------------------------|| Base (Zephyr-7B-SFT-Full) | 6.4 %                        || DPO (Zephyr-7B-Beta)   | 13.2 %                       || SAIL-PP  | 15.9 %                       |> Authors should compare more advanced preference optimization algorithms like ORPO and SimPO. And current results are not impressive for the alignment community.**Response:** Thank you for raising this insightful point. We see ORPO and SimPO are two recent work which propose a different objective than the standard RLHF, and achieve remarkable improvements in terms of alignment performance and efficiency.Our work focus more on bringing standard RLHF to a bilevel optimization framework and propose an effective and efficient approximate algorithm on top of it. We can see some new preference optimization methods including ORPO and SimPO have one fundamental difference from our approach: they do not explicitly incorporate the KL regularization term. The absence of the KL regularization term allows these methods to optimize more aggressively for the reward function by deviating significantly from the reference model. In contrast, our approach is specifically grounded in the standard RLHF, where the KL regularization term ensures that the model remains aligned with the reference distribution while optimizing for the reward function. This distinction makes direct comparisons with ORPO or SimPO less meaningful theoretically, as those methods omit the KL regularization and adopt a fundamentally different optimization objective design.However, we think our work, although developed adhering to the standard RLHF setup, can be compatible and combined with some recent advanced preference optimization algorithms, despite their differences in optimization setups and objectives. This is because we can reformulate their alignment problem as bilevel optimization, and go through the derivation as done in the paper. Taking SimPO as an example, we can treat their reward model definition (Equation (4) in the SimPO paper) as the solution of the upper level optimization (replacing Equation (4) in our manuscript), and adopt their modified Bradley-Terry objective with reward margin (Equation (5) in the SimPO paper) to replace the standard one (Equation (10) in our manuscript). By applying these changes and rederiving the extra gradient terms, we can formulate an adaptation of our method to the SimPO objective. We will implement this combined algorithm, which adapt our methodology to the SimPO objective, and compare with the SimPO as a baseline.Recently many different alignment objectives and algorithms have emerged; it is an interesting question to discuss the compatibility and combination of our method with each objective. We will add more relevant discussions to the appendices, but due to the fact that the compatibility problem with each design is a non-trivial question, this process may incur considerably more work, and we hope the reviewer understands that this effort cannot be fully reflected by the rebuttal period. But we will continue to expand the discussion as the wide compatibility to other designs also strengthens our contribution to the community. We thank the reviewer for raising this insightful point.'}, 
#                    'time': '2024-11-26 15:27:26'
# }
# research_arcade.insert_node("openreview_reviews", node_features=review_features)

# update_node

########## openreview_revisions ##########

########## openreview_arxiv ##########

########## openreview_papers_authors ##########

########## openreview_papers_reviews ##########

########## openreview_papers_revisions ##########

########## openreview_revisions_reviews ##########