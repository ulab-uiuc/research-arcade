import sys
from pathlib import Path
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
# # construct_from_api
# config = {"venue": "ICLR.cc/2025/Conference"}
# research_arcade.construct_table_from_api("openreview_authors", config)

# # construct_from_csv
# config = {"csv_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/csv_data/csv_openreview_author_example.csv"}
# research_arcade.construct_table_from_csv("openreview_authors", config)

# # construct_from_json
# config = {"json_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/json_data/json_openreview_author_example.json"}
# research_arcade.construct_table_from_json("openreview_authors", config)

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
# construct_from_api
# config = {"venue": "ICLR.cc/2025/Conference"}
# research_arcade.construct_table_from_api("openreview_papers", config)

# construct_from_csv
# config = {"csv_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/csv_data/csv_openreview_paper_example.csv"}
# research_arcade.construct_table_from_csv("openreview_papers", config)

# construct_from_json
# config = {"json_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/json_data/json_openreview_paper_example.json"}
# research_arcade.construct_table_from_json("openreview_papers", config)

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
# construct_from_api
# config = {"venue": "ICLR.cc/2013/conference"}
# research_arcade.construct_table_from_api("openreview_reviews", config)

# construct_from_csv
# config = {"csv_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/csv_data/csv_openreview_review_example.csv"}
# research_arcade.construct_table_from_csv("openreview_reviews", config)

# construct_from_json
# config = {"json_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/json_data/json_openreview_review_example.json"}
# research_arcade.construct_table_from_json("openreview_reviews", config)

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
# new_review_features = {'venue': 'ICLR.cc/2025/Conference', 
#                    'review_openreview_id': 'DHwZxFryth', 
#                    'replyto_openreview_id': 'Yqbllggrmw', 
#                    'writer': 'Authors', 
#                    'title': 'Response by Authors', 
#                    'content': {'Title': 'Response to Reviewer 7i95 (1/2)', 'Comment': '> The method does not improve much in the AlpacaEval 2.0 Score. The author should give a detailed explanation. And why not use metrics like length-controlled win rate?**Response:** Thank you for your careful observation and question. We would like to clarify that we are already using the length-controlled (LC) AlpacaEval 2.0 win-rate metric in our evaluations. We will make this clearer in the table header of Table 3.Regarding the fact that the AlpacaEval 2.0 scores on LLama-3 (8B) do not improve compared to the baselines, we believe this is because our base model, the instruction-finetuned LLama-3 (8B), is already trained to perform exceptionally well in terms of helpfulness, which is the focus of the AlpacaEval benchmark. Additionally, the preference dataset we used, UltraFeedback, may not provide significant further enhancement in the helpfulness aspect. This is supported by the slight decrease observed in the AlpacaEval score for the standard DPO baseline as well (see Table 3, results on LLama-3). Therefore, we think these AlpacaEval 2.0 results on LLama-3 (8B) may not indicate that SAIL is ineffective; it may be simply caused by an ill-suited combination of base model, finetuning dataset, and evaluation benchmark.We also further conducted experiments on the Zephyr (7B) model as the backbone, whose AlpacaEval 2.0 win-rate is lower. We still train on the UltraFeedback preference dataset and the other experiment setups are unchanged. In this experiment, we see a larger improvement of the SAIL method compared to the standard DPO baseline (Zephyr-7B-Beta).|             | AlpacaEval 2.0 (LC) Win-Rate ||--------------------|------------------------------|| Base (Zephyr-7B-SFT-Full) | 6.4 %                        || DPO (Zephyr-7B-Beta)   | 13.2 %                       || SAIL-PP  | 15.9 %                       |> Authors should compare more advanced preference optimization algorithms like ORPO and SimPO. And current results are not impressive for the alignment community.**Response:** Thank you for raising this insightful point. We see ORPO and SimPO are two recent work which propose a different objective than the standard RLHF, and achieve remarkable improvements in terms of alignment performance and efficiency.Our work focus more on bringing standard RLHF to a bilevel optimization framework and propose an effective and efficient approximate algorithm on top of it. We can see some new preference optimization methods including ORPO and SimPO have one fundamental difference from our approach: they do not explicitly incorporate the KL regularization term. The absence of the KL regularization term allows these methods to optimize more aggressively for the reward function by deviating significantly from the reference model. In contrast, our approach is specifically grounded in the standard RLHF, where the KL regularization term ensures that the model remains aligned with the reference distribution while optimizing for the reward function. This distinction makes direct comparisons with ORPO or SimPO less meaningful theoretically, as those methods omit the KL regularization and adopt a fundamentally different optimization objective design.However, we think our work, although developed adhering to the standard RLHF setup, can be compatible and combined with some recent advanced preference optimization algorithms, despite their differences in optimization setups and objectives. This is because we can reformulate their alignment problem as bilevel optimization, and go through the derivation as done in the paper. Taking SimPO as an example, we can treat their reward model definition (Equation (4) in the SimPO paper) as the solution of the upper level optimization (replacing Equation (4) in our manuscript), and adopt their modified Bradley-Terry objective with reward margin (Equation (5) in the SimPO paper) to replace the standard one (Equation (10) in our manuscript). By applying these changes and rederiving the extra gradient terms, we can formulate an adaptation of our method to the SimPO objective. We will implement this combined algorithm, which adapt our methodology to the SimPO objective, and compare with the SimPO as a baseline.Recently many different alignment objectives and algorithms have emerged; it is an interesting question to discuss the compatibility and combination of our method with each objective. We will add more relevant discussions to the appendices, but due to the fact that the compatibility problem with each design is a non-trivial question, this process may incur considerably more work, and we hope the reviewer understands that this effort cannot be fully reflected by the rebuttal period. But we will continue to expand the discussion as the wide compatibility to other designs also strengthens our contribution to the community. We thank the reviewer for raising this insightful point.'}, 
#                    'time': '2024-11-26 15:27:26'
# }
# review_id = {"review_openreview_id": "DHwZxFryth"}
# review_features = research_arcade.get_node_features_by_id("openreview_reviews", review_id)
# print(review_features.to_dict(orient="records")[0])
# research_arcade.update_node("openreview_reviews", node_features=new_review_features)
# review_id = {"review_openreview_id": "DHwZxFryth"}
# review_features = research_arcade.get_node_features_by_id("openreview_reviews", review_id)
# print(review_features.to_dict(orient="records")[0])

########## openreview_revisions ##########
# construct_from_api
# venue = "ICLR.cc/2025/Conference"
# filter_list = ["Under review as a conference paper at ICLR 2025", "Published as a conference paper at ICLR 2025"]
# pdf_dir = "/data/jingjunx/openreview_pdfs_2025/"
# log_file = "./log/failed_ids_revisions_2025.txt"
# venue = "ICLR.cc/2023/Conference"
# filter_list = ["Under review as a conference paper at ICLR 2023", "Published as a conference paper at ICLR 2023"]
# pdf_dir = "/data/jingjunx/openreview_pdfs_2023/"
# log_file = "./log/failed_ids_revisions_2023.txt"
# config = {"venue": venue, "filter_list": filter_list, "pdf_dir": pdf_dir, "log_file": log_file}
# venue = "ICLR.cc/2017/conference"
# filter_list = ["Under review as a conference paper at ICLR 2017", "Published as a conference paper at ICLR 2017"]
# pdf_dir = "/data/jingjunx/openreview_pdfs_2017/"
# log_file = "./log/failed_ids_revisions_2017.txt"
# config = {"venue": venue, "filter_list": filter_list, "pdf_dir": pdf_dir, "log_file": log_file}
# research_arcade.construct_table_from_api("openreview_revisions", config)

# construct_from_csv
# config = {"csv_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/csv_data/csv_openreview_revision_example.csv"}
# research_arcade.construct_table_from_csv("openreview_revisions", config)

# construct_from_json
# config = {"json_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/json_data/json_openreview_revision_example.json"}
# research_arcade.construct_table_from_json("openreview_revisions", config)

# get_all_node_features
# openreview_revisions_df = research_arcade.get_all_node_features("openreview_revisions")
# print(len(openreview_revisions_df))

# get_node_features_by_id
# revision_id = {"revision_openreview_id": "yfHQOp5zWc"}
# revision_feature = research_arcade.get_node_features_by_id("openreview_revisions", revision_id)
# print(revision_feature.to_dict(orient="records")[0])

# delete_node
# revision_id = {"revision_openreview_id": "yfHQOp5zWc"}
# revision_feature = research_arcade.delete_node_by_id("openreview_revisions", revision_id)
# print(revision_feature.to_dict(orient="records")[0])

# insert_node
# revision_feature = {'venue': 'ICLR.cc/2025/Conference', 
#                     'original_openreview_id': 'pbTVNlX8Ig', 
#                     'revision_openreview_id': 'yfHQOp5zWc', 
#                     'content': [{'section': '1 INTRODUCTION', 
#                                  'after_section': None, 
#                                  'context_after': '2 RELATED WORK ', 
#                                  'paragraph_idx': 9, 
#                                  'before_section': None, 
#                                  'context_before': 'Published as a conference paper at ICLR 2025 tograd system in PyTorch, specifically tailored for our experimental setup, which is available at ', 
#                                  'modified_lines': 'https://github.com/stephane-rivaud/PETRA. ', 
#                                  'original_lines': 'https://github.com/streethagore/PETRA. ', 
#                                  'after_paragraph_idx': None, 
#                                  'before_paragraph_idx': None}], 
#                     'time': '2025-03-14 15:35:37'}
# research_arcade.insert_node("openreview_revisions", node_features=revision_feature)

# update_node
# new_revision_features = {'venue': 'ICLR.cc/2025/Conference', 
#                     'original_openreview_id': 'pbTVNlX8Ig', 
#                     'revision_openreview_id': 'yfHQOp5zWc', 
#                     'content': [{'section': '1 INTRODUCTION', 
#                                  'after_section': None, 
#                                  'context_after': '2 RELATED WORK ', 
#                                  'paragraph_idx': 9, 
#                                  'before_section': None, 
#                                  'context_before': 'Published as a conference paper at ICLR 2025 tograd system in PyTorch, specifically tailored for our experimental setup, which is available at ', 
#                                  'modified_lines': 'https://github.com/stephane-rivaud/PETRA. ', 
#                                  'original_lines': 'https://github.com/streethagore/PETRA. ', 
#                                  'after_paragraph_idx': None, 
#                                  'before_paragraph_idx': None}], 
#                     'time': '2025-03-14 15:35:37'}
# revision_id = {"revision_openreview_id": "yfHQOp5zWc"}
# revision_feature = research_arcade.get_node_features_by_id("openreview_revisions", revision_id)
# print(revision_feature.to_dict(orient="records")[0])
# research_arcade.update_node("openreview_revisions", node_features=new_revision_features)
# revision_id = {"revision_openreview_id": "yfHQOp5zWc"}
# revision_feature = research_arcade.get_node_features_by_id("openreview_revisions", revision_id)
# print(revision_feature.to_dict(orient="records")[0])

########## openreview_arxiv ##########
# construct_from_api
# config = {"venue": "ICLR.cc/2017/conference"}
# research_arcade.construct_table_from_api("openreview_arxiv", config)

# construct_from_csv
# config = {"csv_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/csv_data/csv_openreview_arxiv_example.csv"}
# research_arcade.construct_table_from_csv("openreview_arxiv", config)

# construct_from_json
# config = {"json_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/json_data/json_openreview_arxiv_example.json"}
# research_arcade.construct_table_from_json("openreview_arxiv", config)

# get_all_edge_features
# openreview_arxiv_df = research_arcade.get_all_edge_features("openreview_arxiv")
# print(len(openreview_arxiv_df))

# get_neighborhood
# openreview_id = {"paper_openreview_id": "zkNCWtw2fd"}
# openreview_arxiv_df = research_arcade.get_neighborhood("openreview_arxiv", openreview_id)
# print(openreview_arxiv_df.to_dict(orient="records")[0])
# arxiv_id = {"arxiv_id": "http://arxiv.org/abs/2408.10536v1"}
# openreview_arxiv_df = research_arcade.get_neighborhood("openreview_arxiv", arxiv_id)
# print(openreview_arxiv_df.to_dict(orient="records")[0])

# delete_edge
# openreview_id = {"paper_openreview_id": "zkNCWtw2fd"}
# openreview_arxiv_df = research_arcade.delete_edge_by_id("openreview_arxiv", openreview_id)
# print(openreview_arxiv_df.to_dict(orient="records")[0])
# arxiv_id = {"arxiv_id": "http://arxiv.org/abs/2408.10536v1"}
# openreview_arxiv_df = research_arcade.delete_edge_by_id("openreview_arxiv", arxiv_id)
# print(openreview_arxiv_df.to_dict(orient="records")[0])
# openreview_arxiv_id = {"paper_openreview_id": "zkNCWtw2fd", "arxiv_id": "http://arxiv.org/abs/2408.10536v1"}
# openreview_arxiv_df = research_arcade.delete_edge_by_id("openreview_arxiv", openreview_arxiv_id)
# print(openreview_arxiv_df.to_dict(orient="records")[0])

# insert_edge
# openreview_arxiv = {'venue': 'ICLR.cc/2025/Conference', 
#                     'paper_openreview_id': 'zkNCWtw2fd', 
#                     'arxiv_id': 'http://arxiv.org/abs/2408.10536v1', 
#                     'title': 'Synergistic Approach for Simultaneous Optimization of Monolingual, Cross-lingual, and Multilingual Information Retrieval'
# }
# research_arcade.insert_edge("openreview_arxiv", openreview_arxiv)

########## openreview_papers_authors ##########
# construct_from_api
# config = {"venue": "ICLR.cc/2025/Conference"}
# research_arcade.construct_table_from_api("openreview_papers_authors", config)

# construct_from_csv
# config = {"csv_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/csv_data/csv_openreview_papers_authors_example.csv"}
# research_arcade.construct_table_from_csv("openreview_papers_authors", config)

# construct_from_json
# config = {"json_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/json_data/json_openreview_papers_authors_example.json"}
# research_arcade.construct_table_from_json("openreview_papers_authors", config)

# get_all_edge_features
# openreview_papers_authors = research_arcade.get_all_edge_features("openreview_papers_authors")
# print(len(openreview_papers_authors))

# get_neighborhood
# paper_id = {"paper_openreview_id": "00SnKBGTsz"}
# openreview_papers_authors = research_arcade.get_neighborhood("openreview_papers_authors", paper_id)
# print(openreview_papers_authors.to_dict(orient="records"))
# author_id = {'author_openreview_id': '~Elias_Stengel-Eskin1'}
# openreview_papers_authors = research_arcade.get_neighborhood("openreview_papers_authors", author_id)
# print(openreview_papers_authors.to_dict(orient="records"))

# delete_edge
# paper_id = {"paper_openreview_id": "00SnKBGTsz"}
# openreview_papers_authors = research_arcade.delete_edge_by_id("openreview_papers_authors", paper_id)
# print(openreview_papers_authors.to_dict(orient="records"))
# author_id = {'author_openreview_id': '~Elias_Stengel-Eskin1'}
# openreview_papers_authors = research_arcade.delete_edge_by_id("openreview_papers_authors", author_id)
# print(openreview_papers_authors.to_dict(orient="records"))

# insert_edge
# paper_authors = [{'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'author_openreview_id': '~Elias_Stengel-Eskin1'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'author_openreview_id': '~Zaid_Khan1'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'author_openreview_id': '~Jaemin_Cho1'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'author_openreview_id': '~Mohit_Bansal2'}]
# for item in paper_authors:
#     research_arcade.insert_edge("openreview_papers_authors", item)
# author_papers = [{'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': 'Xbl6t6zxZs', 'author_openreview_id': '~Elias_Stengel-Eskin1'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': 'fDcn3S8oAt', 'author_openreview_id': '~Elias_Stengel-Eskin1'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': 'j9wBgcxa7N', 'author_openreview_id': '~Elias_Stengel-Eskin1'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': 'zd0iX5xBhA', 'author_openreview_id': '~Elias_Stengel-Eskin1'}, 
#                  {'venue': 'ICLR.cc/2024/Conference', 'paper_openreview_id': 'L4nOxziGf9', 'author_openreview_id': '~Elias_Stengel-Eskin1'}, 
#                  {'venue': 'ICLR.cc/2024/Conference', 'paper_openreview_id': 'qL9gogRepu', 'author_openreview_id': '~Elias_Stengel-Eskin1'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'author_openreview_id': '~Elias_Stengel-Eskin1'}]
# for item in author_papers:
#     research_arcade.insert_edge("openreview_papers_authors", item)

########## openreview_papers_reviews ##########
# construct_from_api
# config = {"venue": "ICLR.cc/2017/conference"}
# research_arcade.construct_table_from_api("openreview_papers_reviews", config)

# construct_from_csv
# config = {"csv_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/csv_data/csv_openreview_papers_reviews_example.csv"}
# research_arcade.construct_table_from_csv("openreview_papers_reviews", config)

# construct_from_json
# config = {"json_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/json_data/json_openreview_papers_reviews_example.json"}
# research_arcade.construct_table_from_json("openreview_papers_reviews", config)

# get_all_edge_features
# openreview_papers_reviews = research_arcade.get_all_edge_features("openreview_papers_reviews")
# print(len(openreview_papers_reviews))

# get_neighborhood
# paper_id = {"paper_openreview_id": "00SnKBGTsz"}
# openreview_papers_reviews = research_arcade.get_neighborhood("openreview_papers_reviews", paper_id)
# print(openreview_papers_reviews.to_dict(orient="records"))
# review_id = {"review_openreview_id": "13mj0Rtn5W"}
# openreview_papers_reviews = research_arcade.get_neighborhood("openreview_papers_reviews", review_id)
# print(openreview_papers_reviews.to_dict(orient="records"))

# delete_edge
# paper_review_id = {"paper_openreview_id": "00SnKBGTsz", "review_openreview_id": "13mj0Rtn5W"}
# openreview_papers_reviews = research_arcade.delete_edge_by_id("openreview_papers_reviews", paper_review_id)
# print(openreview_papers_reviews.to_dict(orient="records"))
# review_id = {"review_openreview_id": "13mj0Rtn5W"}
# openreview_papers_reviews = research_arcade.delete_edge_by_id("openreview_papers_reviews", review_id)
# print(openreview_papers_reviews.to_dict(orient="records"))
# paper_id = {"paper_openreview_id": "00SnKBGTsz"}
# openreview_papers_reviews = research_arcade.delete_edge_by_id("openreview_papers_reviews", paper_id)
# print(openreview_papers_reviews.to_dict(orient="records"))

# insert_edge
# paper_review = {'venue': 'ICLR.cc/2025/Conference', 
#                 'paper_openreview_id': '00SnKBGTsz', 
#                 'review_openreview_id': '13mj0Rtn5W', 
#                 'title': 'Response by Authors', 
#                 'time': '2024-11-27 17:27:45'}
# research_arcade.insert_edge("openreview_papers_reviews", paper_review)
# paper_reviews = [{'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': '7XT4kLWV2f', 'title': 'Official Review by Reviewer_wuGW', 'time': '2024-11-01 14:52:22'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'i3QgWgrJff', 'title': 'Official Review by Reviewer_rVo8', 'time': '2024-11-04 02:37:10'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'GMsjHLXdOx', 'title': 'Official Review by Reviewer_c5nB', 'time': '2024-11-04 09:59:14'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'r8ZflFk3T7', 'title': 'Official Review by Reviewer_VQ9Y', 'time': '2024-11-06 00:15:47'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': '4CnQpVCYkF', 'title': 'Response by Authors', 'time': '2024-11-20 22:48:42'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'h1qvpjhRP3', 'title': 'Response by Authors', 'time': '2024-11-20 22:51:07'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'pOR42YNLtU', 'title': 'Response by Authors', 'time': '2024-11-20 22:55:04'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'Aq2tBtB0lt', 'title': 'Response by Authors', 'time': '2024-11-20 22:57:18'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'm1iUqPHpwk', 'title': 'Response by Authors', 'time': '2024-11-20 22:58:29'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': '66buacQmRe', 'title': 'Response by Authors', 'time': '2024-11-20 23:02:21'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'Bgr7Ol90m7', 'title': 'Response by Authors', 'time': '2024-11-22 23:11:06'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'H2h2K6a8x5', 'title': 'Response by Reviewer', 'time': '2024-11-23 10:04:58'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'la5jPwJU4g', 'title': 'Response by Authors', 'time': '2024-11-24 19:17:22'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'DjVKsUoFN2', 'title': 'Response by Reviewer', 'time': '2024-11-25 04:00:18'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'C3MhCuKhTf', 'title': 'Response by Authors', 'time': '2024-11-25 19:44:38'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'ZqwAYtcmhv', 'title': 'Response by Authors', 'time': '2024-11-25 19:45:43'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': '9OQJoesINr', 'title': 'Response by Reviewer', 'time': '2024-11-25 20:07:51'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'wqTNtVDwef', 'title': 'Response by Authors', 'time': '2024-11-26 03:32:30'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'NEsxOTkkIV', 'title': 'Response by Reviewer', 'time': '2024-11-26 20:00:00'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': '13mj0Rtn5W', 'title': 'Response by Authors', 'time': '2024-11-27 17:27:45'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'hWat8aFBRw', 'title': 'Response by Reviewer', 'time': '2024-11-27 11:34:03'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'wnsiUkDh00', 'title': 'Response by Authors', 'time': '2024-11-27 17:28:35'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'zpboemkkjR', 'title': 'Meta Review of Submission11063 by Area_Chair_eoLd', 'time': '2024-12-20 15:14:25'}, 
#                  {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'review_openreview_id': 'kokKFEn2fw', 'title': 'Paper Decision', 'time': '2025-01-22 05:35:00'}
# ]
# for item in paper_reviews:
#     research_arcade.insert_edge("openreview_papers_reviews", item)

########## openreview_papers_revisions ##########
# construct_from_api
# config = {"venue": "ICLR.cc/2025/Conference"}
# research_arcade.construct_table_from_api("openreview_papers_revisions", config)

# construct_from_csv
# config = {"csv_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/csv_data/csv_openreview_papers_revisions_example.csv"}
# research_arcade.construct_table_from_csv("openreview_papers_revisions", config)

# construct_from_json
# config = {"json_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/json_data/json_openreview_papers_revisions_example.json"}
# research_arcade.construct_table_from_json("openreview_papers_revisions", config)

# get_all_edge_features
# openreview_papers_revisions = research_arcade.get_all_edge_features("openreview_papers_revisions")
# print(len(openreview_papers_revisions))

# get_neighborhood
# paper_id = {"paper_openreview_id": "00SnKBGTsz"}
# paper_revision = research_arcade.get_neighborhood("openreview_papers_revisions", paper_id)
# print(paper_revision.to_dict(orient="records"))
# revision_id = {"revision_openreview_id": "dzL3IRBnE4"}
# paper_revision = research_arcade.get_neighborhood("openreview_papers_revisions", revision_id)
# print(paper_revision.to_dict(orient="records"))

# delete_edge
# paper_revision_id = {"paper_openreview_id": "00SnKBGTsz", "revision_openreview_id": "dzL3IRBnE4"}
# paper_revision = research_arcade.delete_edge_by_id("openreview_papers_revisions", paper_revision_id)
# print(paper_revision.to_dict(orient="records"))
# revision_id = {"revision_openreview_id": "dzL3IRBnE4"}
# paper_revision = research_arcade.delete_edge_by_id("openreview_papers_revisions", revision_id)
# print(paper_revision.to_dict(orient="records"))
# paper_id = {"paper_openreview_id": "00SnKBGTsz"}
# paper_revision = research_arcade.delete_edge_by_id("openreview_papers_revisions", paper_id)
# print(paper_revision.to_dict(orient="records"))

# insert_edge
# paper_revision = {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'revision_openreview_id': 'dzL3IRBnE4', 'title': 'Camera_Ready_Revision', 'time': '2025-03-01 03:36:55'}
# research_arcade.insert_edge("openreview_papers_revisions", paper_revision)
# paper_revisions = [{'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'revision_openreview_id': 'oT4N28siLO', 'title': 'Camera_Ready_Revision', 'time': '2025-03-02 01:35:16'}, 
#                    {'venue': 'ICLR.cc/2025/Conference', 'paper_openreview_id': '00SnKBGTsz', 'revision_openreview_id': 'dzL3IRBnE4', 'title': 'Camera_Ready_Revision', 'time': '2025-03-01 03:36:55'}]
# for item in paper_revisions:
#     research_arcade.insert_edge("openreview_papers_revisions", item)

########## openreview_revisions_reviews ##########
# construct_based_on_existing_tables
# papers_reviews_df = research_arcade.get_all_edge_features("openreview_papers_reviews")
# print(len(papers_reviews_df))
# papers_revisions_df = research_arcade.get_all_edge_features("openreview_papers_revisions")
# print(len(papers_revisions_df))
# config = {"papers_reviews_df": papers_reviews_df, "papers_revisions_df": papers_revisions_df}
# research_arcade.construct_table_from_api("openreview_revisions_reviews", config)

# construct_from_csv
# config = {"csv_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/csv_data/csv_openreview_revisions_reviews_example.csv"}
# research_arcade.construct_table_from_csv("openreview_revisions_reviews", config)

# construct_from_json
# config = {"json_file": "/home/jingjunx/openreview_benchmark/Code/paper-crawler/examples/json_data/json_openreview_revisions_reviews_example.json"}
# research_arcade.construct_table_from_json("openreview_revisions_reviews", config)

# get_all_edge_features
# openreview_revisions_reviews = research_arcade.get_all_edge_features("openreview_revisions_reviews")
# print(len(openreview_revisions_reviews))

# get_neighborhood
# revision_id = {'revision_openreview_id': 'cX02yuzwWI'}
# revision_review = research_arcade.get_neighborhood("openreview_revisions_reviews", revision_id)
# print(revision_review.to_dict(orient="records"))
# review_id = {'review_openreview_id': 'wumckDPIQ3'}
# revision_review = research_arcade.get_neighborhood("openreview_revisions_reviews", review_id)
# print(revision_review.to_dict(orient="records"))

# delete_edge
# revision_review_id = {'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'wumckDPIQ3'}
# revision_review = research_arcade.delete_edge_by_id("openreview_revisions_reviews", revision_review_id)
# print(revision_review.to_dict(orient="records"))
# review_id = {'review_openreview_id': 'wumckDPIQ3'}
# revision_review = research_arcade.delete_edge_by_id("openreview_revisions_reviews", review_id)
# print(revision_review.to_dict(orient="records"))
# paper_id = {'revision_openreview_id': 'cX02yuzwWI'}
# revision_review = research_arcade.delete_edge_by_id("openreview_revisions_reviews", paper_id)
# print(revision_review.to_dict(orient="records"))

# insert_edge
# revision_review = {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'wumckDPIQ3'}
# research_arcade.insert_edge("openreview_revisions_reviews", revision_review)
# revision_reviews = [{'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'wumckDPIQ3'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': '138cOdBpgA'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'yKh1fQYnUZ'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'Pvt0OjNSp2'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'MUhlEYyBD9'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': '2mqiS3J8wC'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'Er8QTorcyr'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'AvtD9uxRtX'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': '2tgxTGynNm'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': '5MKJE3sFsd'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'wViZ0H4ErF'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': '0c1It75dTb'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'PFwia9lcjP'}, 
#                     {'venue': 'ICLR.cc/2025/Conference', 'revision_openreview_id': 'cX02yuzwWI', 'review_openreview_id': 'ygCqaGNPee'}]
# for item in revision_reviews:
#     research_arcade.insert_edge("openreview_revisions_reviews", item)