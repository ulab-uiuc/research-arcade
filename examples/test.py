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

import json
content = {'Title': 'Response to Reviewer 7i95 (1/2)', 'Comment': '> The method does not improve much in the AlpacaEval 2.0 Score. The author should give a detailed explanation. And why not use metrics like length-controlled win rate?**Response:** Thank you for your careful observation and question. We would like to clarify that we are already using the length-controlled (LC) AlpacaEval 2.0 win-rate metric in our evaluations. We will make this clearer in the table header of Table 3.Regarding the fact that the AlpacaEval 2.0 scores on LLama-3 (8B) do not improve compared to the baselines, we believe this is because our base model, the instruction-finetuned LLama-3 (8B), is already trained to perform exceptionally well in terms of helpfulness, which is the focus of the AlpacaEval benchmark. Additionally, the preference dataset we used, UltraFeedback, may not provide significant further enhancement in the helpfulness aspect. This is supported by the slight decrease observed in the AlpacaEval score for the standard DPO baseline as well (see Table 3, results on LLama-3). Therefore, we think these AlpacaEval 2.0 results on LLama-3 (8B) may not indicate that SAIL is ineffective; it may be simply caused by an ill-suited combination of base model, finetuning dataset, and evaluation benchmark.We also further conducted experiments on the Zephyr (7B) model as the backbone, whose AlpacaEval 2.0 win-rate is lower. We still train on the UltraFeedback preference dataset and the other experiment setups are unchanged. In this experiment, we see a larger improvement of the SAIL method compared to the standard DPO baseline (Zephyr-7B-Beta).|             | AlpacaEval 2.0 (LC) Win-Rate ||--------------------|------------------------------|| Base (Zephyr-7B-SFT-Full) | 6.4 %                        || DPO (Zephyr-7B-Beta)   | 13.2 %                       || SAIL-PP  | 15.9 %                       |> Authors should compare more advanced preference optimization algorithms like ORPO and SimPO. And current results are not impressive for the alignment community.**Response:** Thank you for raising this insightful point. We see ORPO and SimPO are two recent work which propose a different objective than the standard RLHF, and achieve remarkable improvements in terms of alignment performance and efficiency.Our work focus more on bringing standard RLHF to a bilevel optimization framework and propose an effective and efficient approximate algorithm on top of it. We can see some new preference optimization methods including ORPO and SimPO have one fundamental difference from our approach: they do not explicitly incorporate the KL regularization term. The absence of the KL regularization term allows these methods to optimize more aggressively for the reward function by deviating significantly from the reference model. In contrast, our approach is specifically grounded in the standard RLHF, where the KL regularization term ensures that the model remains aligned with the reference distribution while optimizing for the reward function. This distinction makes direct comparisons with ORPO or SimPO less meaningful theoretically, as those methods omit the KL regularization and adopt a fundamentally different optimization objective design.However, we think our work, although developed adhering to the standard RLHF setup, can be compatible and combined with some recent advanced preference optimization algorithms, despite their differences in optimization setups and objectives. This is because we can reformulate their alignment problem as bilevel optimization, and go through the derivation as done in the paper. Taking SimPO as an example, we can treat their reward model definition (Equation (4) in the SimPO paper) as the solution of the upper level optimization (replacing Equation (4) in our manuscript), and adopt their modified Bradley-Terry objective with reward margin (Equation (5) in the SimPO paper) to replace the standard one (Equation (10) in our manuscript). By applying these changes and rederiving the extra gradient terms, we can formulate an adaptation of our method to the SimPO objective. We will implement this combined algorithm, which adapt our methodology to the SimPO objective, and compare with the SimPO as a baseline.Recently many different alignment objectives and algorithms have emerged; it is an interesting question to discuss the compatibility and combination of our method with each objective. We will add more relevant discussions to the appendices, but due to the fact that the compatibility problem with each design is a non-trivial question, this process may incur considerably more work, and we hope the reviewer understands that this effort cannot be fully reflected by the rebuttal period. But we will continue to expand the discussion as the wide compatibility to other designs also strengthens our contribution to the community. We thank the reviewer for raising this insightful point.'}
print(json.dumps(content) if isinstance(content, dict) else content)
########## openreview_authors ##########
# get_all_node_features
# openreview_authors_df = research_arcade.get_all_node_features("openreview_authors")
# openreview_authors_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_authors.csv", index=False)
# print(len(openreview_authors_df))

########## openreview_papers ##########
# get_all_node_features
# openreview_papers_df = research_arcade.get_all_node_features("openreview_papers")
# openreview_papers_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_papers.csv", index=False)
# print(len(openreview_papers_df))

########## openreview_reviews ##########
# get_all_node_features
# openreview_reviews_df = research_arcade.get_all_node_features("openreview_reviews")
# openreview_reviews_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_reviews.csv", index=False)
# print(len(openreview_reviews_df))

########## openreview_revisions ##########
# get_all_node_features
# openreview_revisions_df = research_arcade.get_all_node_features("openreview_revisions")
# openreview_revisions_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_revisions.csv", index=False)
# print(len(openreview_revisions_df))

########## openreview_arxiv ##########
# get_all_edge_features
# openreview_arxiv_df = research_arcade.get_all_edge_features("openreview_arxiv")
# openreview_arxiv_df.to_csv("/data/jingjunx/my_research_arcade_data/openreview_arxiv.csv", index=False)
# print(len(openreview_arxiv_df))

########## openreview_papers_authors ##########
# get_all_edge_features
# openreview_papers_authors = research_arcade.get_all_edge_features("openreview_papers_authors")
# openreview_papers_authors.to_csv("/data/jingjunx/my_research_arcade_data/openreview_papers_authors.csv", index=False)
# print(len(openreview_papers_authors))

########## openreview_papers_reviews ##########
# get_all_edge_features
# openreview_papers_reviews = research_arcade.get_all_edge_features("openreview_papers_reviews")
# openreview_papers_reviews.to_csv("/data/jingjunx/my_research_arcade_data/openreview_papers_reviews.csv", index=False)
# print(len(openreview_papers_reviews))

########## openreview_papers_revisions ##########
# get_all_edge_features
# openreview_papers_revisions = research_arcade.get_all_edge_features("openreview_papers_revisions")
# openreview_papers_revisions.to_csv("/data/jingjunx/my_research_arcade_data/openreview_papers_revisions.csv", index=False)
# print(len(openreview_papers_revisions))

########## openreview_revisions_reviews ##########
# get_all_edge_features
# openreview_revisions_reviews = research_arcade.get_all_edge_features("openreview_revisions_reviews")
# openreview_revisions_reviews.to_csv("/data/jingjunx/my_research_arcade_data/openreview_revisions_reviews.csv", index=False)
# print(len(openreview_revisions_reviews))

# import pandas as pd
# from tqdm import tqdm
# df25 = pd.read_csv('/home/jingjunx/my-data/2025_papers.csv')
# df24 = pd.read_csv('/home/jingjunx/my-data/2024_papers.csv')
# df23 = pd.read_csv('/home/jingjunx/my-data/2023_papers.csv')
# df22 = pd.read_csv('/home/jingjunx/my-data/2022_papers.csv')
# df21 = pd.read_csv('/home/jingjunx/my-data/2021_papers.csv')
# df20 = pd.read_csv('/home/jingjunx/my-data/2020_papers.csv')
# df19 = pd.read_csv('/home/jingjunx/my-data/2019_papers.csv')
# df18 = pd.read_csv('/home/jingjunx/my-data/2018_papers.csv')
# df17 = pd.read_csv('/home/jingjunx/my-data/2017_papers.csv')
# df14 = pd.read_csv('/home/jingjunx/my-data/2014_papers.csv')
# df13 = pd.read_csv('/home/jingjunx/my-data/2013_papers.csv')
# df_all = pd.concat([df25, df24, df23, df22, df21, df20, df19, df18, df17, df14, df13], 
#                    ignore_index=True)
# all_dict = df_all.to_dict(orient="records")
# for item in tqdm(all_dict):
#     research_arcade.update_node("openreview_papers", node_features=item)
