from paper_collector.graph_construction import build_citation_graph_thread
from paper_collector.utils import None_constraint

# 2302.07842 Augmented Language Models: a Survey
# 2306.13549 MLLM survey
# 2308.11432 A Survey on Large Language Model based Autonomous Agents
# 1901.00596 A Comprehensive Survey on Graph Neural Networks
# 2310.10315v1, 2308.11269 quantum ML
# 2304.03442v1 Stanford Town
# 2310.04959v1 CoT survey
# 2401.00288 Code DL survey
# 2304.01565 graph diffusion
# 2307.00067 transformers in healthcare
# 2309.02427v1 cognitive agent
# 2306.04542v1 diffusion model
# 2312.01232v2 vision transformer
# 2312.07843v1 robot
# 2202.02703v2 auto driving
# 2110.03063v1 Gene
# 2306.11768v5 drug design
# 2211.16742v1 protein
# 2307.10500v1 cell
# build_citation_graph_thread(['2411.05902','2411.06837','2411.06606','2411.07658', '2411.07690'], '../../test-1000/source_code', '../../test-1000/working_folder', '../../test-1000/output',None, year_constraint(2024,2025), 1000, 8, True, 2000)
# arxiv_list = ['2302.07842v1', '2306.13549v1', '1901.00596', '2310.10315v1', '2308.11269','2304.03442v1', '2310.04959v1','2401.00288v1','2304.01565v1', '2307.00067v1','2309.02427v','2306.04542v1','2312.01232v2','2312.07843v1','2202.02703v2', '2110.03063v1','2306.11768v5','2211.16742v1','2307.10500v1']

arxiv_list = ["2311.01149v2"]
build_citation_graph_thread(
    arxiv_list,
    "../data/hf_/source_code",
    "../data/hf_/working_folder",
    "../data/hf_/output",
    None,
    None_constraint,
    len(arxiv_list),
    1,
    True,
    len(arxiv_list),
)
