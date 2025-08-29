import pandas as pd
from datetime import datetime
from sqlDatabaseConstructor import sqlDatabaseConstructor
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
import json
import copy
import re
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# nltk.download('punkt_tab') # predict
accelerator = Accelerator()

sql_database = sqlDatabaseConstructor()
venue = 'ICLR.cc/2025/Conference'

def construct_review_graphs(total_review_id_list, paper_id):
    total_reviews_df = reviews_df[reviews_df['review_openreview_id'].isin(total_review_id_list)].sort_values(by='time', ascending=True)
    
    review_graph_id_list = total_reviews_df[total_reviews_df['replyto_openreview_id'] == paper_id]['review_openreview_id'].tolist()
    
    review_graphs = {}
    for review_graph_id in review_graph_id_list:
        review_graphs[review_graph_id] = []
    
    # init review graphs
    for review in total_reviews_df.itertuples():
        if review.review_openreview_id in review_graphs:
            if review.title.startswith("Official Review"):
                text = review.title + "\nSummary: " + str(review.content["Summary"]) + "\nStrengths: " + str(review.content["Strengths"]) + "\nWeaknesses: " + str(review.content["Weaknesses"]) + "\nQuestions: " + str(review.content["Questions"])
            elif review.title.startswith("Response"):
                text = review.title + "\n" + str(review.content["Comment"])
            elif review.title.startswith("Meta"):
                text = review.title + "\n" + str(review.content["Meta Review"])
            elif review.title.startswith("Paper Decision"):
                text = str(review.content["Decision"]) + str(review.content["Comment"])
            content = {
                "review_id": review.review_openreview_id,
                "replyto_id": review.replyto_openreview_id,
                "text": text,
                "time": review.time
            }
            review_graphs[review.review_openreview_id].append(content)
            # review_graphs[review.review_openreview_id].append(review.review_openreview_id)
    
    # construct review graphs
    for review_graph_id in review_graph_id_list:
        current_reply_ids = [review_graph_id]
        relevant_review_df = reviews_df[reviews_df['replyto_openreview_id'].isin(current_reply_ids)].sort_values(by='time', ascending=True)
        while len(relevant_review_df) >= 1:
            for review in relevant_review_df.itertuples():
                if review.title.startswith("Official Review"):
                    text = review.title + "\nSummary: " + str(review.content["Summary"]) + "\nStrengths: " + str(review.content["Strengths"]) + "\nWeaknesses: " + str(review.content["Weaknesses"]) + "\nQuestions: " + str(review.content["Questions"])
                elif review.title.startswith("Response"):
                    text = review.title + "\n" + str(review.content["Comment"])
                elif review.title.startswith("Meta"):
                    text = review.title + "\n" + str(review.content["Meta Review"])
                elif review.title.startswith("Paper Decision"):
                    text = str(review.content["Decision"]) + str(review.content["Comment"])
                content = {
                    "review_id": review.review_openreview_id,
                    "replyto_id": review.replyto_openreview_id,
                    "text": text,
                    "time": review.time
                }
                review_graphs[review_graph_id].append(content)
                # review_graphs[review_graph_id].append(review.review_openreview_id)
            
            current_reply_ids = relevant_review_df['review_openreview_id'].unique()
            relevant_review_df = reviews_df[reviews_df['replyto_openreview_id'].isin(current_reply_ids)].sort_values(by='time', ascending=True)
    
    return review_graphs

def construct_revision_graph(total_revision_id_list, max_paragraph_num=40):
    revision_graph = []
    
    total_revisions_df = revisions_df[revisions_df['revision_openreview_id'].isin(total_revision_id_list)].sort_values(by='time', ascending=True)
    for revision in total_revisions_df.itertuples():
        # get paragraph
        modified_paragraphs = set()
        for diff in revision.content:
            if diff['paragraph_idx'] <= max_paragraph_num:
                modified_paragraphs.update([diff['paragraph_idx']])
        modified_paragraphs_list = list(modified_paragraphs)
        
        # get paper original paragraph and revised paragraph
        original_id = revision.original_openreview_id
        modified_id = revision.revision_openreview_id
        
        revision_subgraph = {
            "original_id": original_id,
            "modified_id": modified_id,
            "time": revision.time,
            "modified_content": []
        }
        
        original_paper_paragraphs_df = paragraphs_df[paragraphs_df['paper_openreview_id'] == original_id].sort_values(by='paragraph_idx', ascending=True)[:40]
        modified_paper_paragraphs_df = paragraphs_df[paragraphs_df['paper_openreview_id'] == modified_id].sort_values(by='paragraph_idx', ascending=True)[:40]
        
        # modified_paragraphs = set()
        for para_idx in modified_paragraphs_list:
            original_paragraph = original_paper_paragraphs_df[original_paper_paragraphs_df['paragraph_idx'] == para_idx]["content"].tolist()
            modified_paragraph = modified_paper_paragraphs_df[modified_paper_paragraphs_df['paragraph_idx'] == para_idx]["content"].tolist()
            
            if len(original_paragraph) == 0 or len(modified_paragraph) == 0:
                continue
            else:
                content = {
                    "para_idx": para_idx,
                    "original_paragraph": original_paragraph[0],
                    "modified_paragraph": modified_paragraph[0]
                }
                
                revision_subgraph["modified_content"].append(content)
        
        if len(revision_subgraph["modified_content"]) > 0:
            revision_graph.append(revision_subgraph)
        # revision_graph.append(revision.original_openreview_id)
    
    return revision_graph

def format_input(reviews_list, original_paragraph):
    reviews = reviews_list[0]
    for review in reviews_list[1:]:
        reviews += "\n\n" + review
# def format_input(reviews, original_paragraph):
    PROMPT_FORMAT = f'''
        REVIEWS: 
        {reviews}

        ORIGINAL PARAGRAPH: 
        {original_paragraph}

        INSTRUCTIONS:
        - Please revise the paragraph according to the provided reviews.
        - Output only the revised paragraph, enclosed between [START] and [END], without any extra explanation or analysis.

        REVISED PARAGRAPH: [START]{{your revised paragraph here}}[END]
    '''
    
    return PROMPT_FORMAT

def prompting(model, tokenizer, prompt, device):
    # Prepare the message for the model
    message = [{"role": "user", "content": prompt}]
    
    # Tokenize the message with the appropriate template, ensuring we don't tokenize the message itself
    prompt_input = tokenizer.apply_chat_template(message, tokenize=False)
    inputs = tokenizer(prompt_input, return_tensors="pt")  # Get input IDs in the proper format
    inputs = inputs.to(device)
    
    with torch.no_grad():
        # Generate the response from the model
        outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=4096, num_return_sequences=1, temperature=0.6, top_p=0.95)
    
    # Decode the output to get the generated response (excluding the user input part)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # skip special tokens like <s>, </s>, etc.
    
    # Optional: if you are still seeing the user input in the response, you can slice the response to exclude it
    # Assuming the model's output always starts after a special token or in the first few tokens.
    # You might want to cut the response based on the length of the user input:
    return response[len(prompt):].strip()  # Remove the user prompt from the beginning of the response


def extract_revised_paragraph(response_text):
    # 使用正则表达式匹配[START]和[END]之间的内容
    match = re.search(r'\[START\](.*?)\[END\]', response_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()  # 返回匹配的内容并去掉多余的空白
    else:
        return response_text
    

def compute_rouge(target, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target, prediction)
    return scores

def compute_bleu(target, prediction):
    reference_tokens = word_tokenize(target.lower())
    hypothesis_tokens = word_tokenize(prediction.lower())

    # 计算 BLEU 分数
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
    return bleu_score

# node
# papers_df = sql_database.get_node_features_by_venue("papers", venue)
reviews_df = sql_database.get_node_features_by_venue("reviews", venue)
revisions_df = sql_database.get_node_features_by_venue("revisions", venue)
paragraphs_df = sql_database.get_node_features_by_venue("paragraphs", venue)

# edge
papers_reviews_df = sql_database.get_edge_features_by_venue("papers_reviews", venue)
papers_revisions_df = sql_database.get_edge_features_by_venue("papers_revisions", venue)

# combine dataset
# extract papers that have revisions
unique_paper_ids = papers_revisions_df['paper_openreview_id'].unique()

# model
# predict
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto").to(device)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B") #Load the associated tokenizer:

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# model, tokenizer = accelerator.prepare(model, tokenizer)

result = {}

dataset = []

# 总数是4170，可以取前1000作为测试先
for paper_id in tqdm(unique_paper_ids[:100]):
    # get the review ids
    paper_review_edges = papers_reviews_df[papers_reviews_df['paper_openreview_id'] == paper_id]
    total_review_id_list = paper_review_edges['review_openreview_id'].tolist()

    # construct review graphs
    review_graphs = construct_review_graphs(total_review_id_list, paper_id)
    num_review_graphs = len(review_graphs)

    ### revisions ###
    # get revision ids
    paper_revision_edges = papers_revisions_df[papers_revisions_df['paper_openreview_id'] == paper_id]
    total_revision_id_list = paper_revision_edges['revision_openreview_id'].tolist()

    # get revision graph
    revision_graph = construct_revision_graph(total_revision_id_list)
    
    # data = {
    #     "review_graphs": review_graphs,
    #     "revision_graph": revision_graph
    # }
    
    # dataset.append(data)
    # with open("generate_modified_paragraph_dataset.json", 'w') as f:
    #     json.dump(dataset, f, indent=4, ensure_ascii=False)

    # start doing predict the modified paragraph
    former_index = [0] * num_review_graphs
    current_index = [0] * num_review_graphs
    for revision_subgraph in revision_graph:  
        # get revision time
        revision_time = datetime.strptime(revision_subgraph['time'], "%Y-%m-%d %H:%M:%S")
        
        # get corresponding review list
        reviews_list = []
        review_ids_list = []
        for idx, (graph_id, review_graph) in enumerate(zip(review_graphs.keys(), review_graphs.values())):
            reviews = ""
            review_ids_sublist = []
            for sub_idx, review_content in enumerate(review_graph[former_index[idx]:]):
                review_time = datetime.strptime(review_content['time'], "%Y-%m-%d %H:%M:%S")
                if review_time > revision_time:
                    former_index[idx] = current_index[idx]
                    break
                else:
                    reviews = reviews + "\n\n" + review_content['text']
                    review_ids_sublist.append(review_content['review_id'])
                    current_index[idx] = current_index[idx] + 1
                if sub_idx + 1 == len(review_graph[former_index[idx]:]):
                    former_index[idx] = current_index[idx]
            
            if len(reviews) >= 100:
                reviews_list.append(reviews)
                review_ids_list.append(review_ids_sublist)
        if len(review_ids_list) < 1:
            continue
        
        result[revision_subgraph["original_id"]] = {} # dataset
        result[revision_subgraph["original_id"]]["review"] = reviews_list # dataset
        result[revision_subgraph["original_id"]]["modified_content"] = [] # dataset
        
        # result[revision_subgraph["original_id"]] = [] # predict
        
        for revision_content in revision_subgraph["modified_content"]:
            bleu_scores = []
            rouge_scores = []
            
            original_paragraph = revision_content["original_paragraph"]
            modified_paragraph = revision_content["modified_paragraph"]
            
            # dataset
            data = {
                "original_paragraph": original_paragraph,
                "modified_paragraph": modified_paragraph
            }
            
            result[revision_subgraph["original_id"]]["modified_content"].append(data)
            
            # predict
            # data = {
            #         "modified_paragraph": modified_paragraph,
            #         "predicted_paragraphs": []
            #     }
            # # for review in reviews_list:
            #     # prompt = format_input(review, original_paragraph)
            # prompt = format_input(reviews_list, original_paragraph)
                
            # predicted_paragraph = extract_revised_paragraph(prompting(model, tokenizer, prompt, accelerator.device))
            
            # data["predicted_paragraphs"].append(predicted_paragraph)
                
            # result[revision_subgraph["original_id"]].append(data)
                # print(predicted_paragraph)
                # bleu_scores.append(compute_bleu(modified_paragraph, predicted_paragraph))
                # rouge_scores.append(compute_rouge(modified_paragraph, predicted_paragraph))
            
            # print("bleu")
            # print(max(bleu_scores))
            # print("rouge")
            # print(max(rouge_scores))

# output_dir = "/data/jingjunx/evaluation_tasks/generate_modified_paragraph/llm/Qwen3-8B" # predict
output_dir = "/data/jingjunx/evaluation_tasks/generate_modified_paragraph/llm" # dataset
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# output_file = output_dir+"/generate_modified_paragraph_completion.json" # predict
output_file = output_dir+"/generate_modified_paragraph_test_dataset_100.json" # dataset

with open(output_file, 'w') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
    
print("File save to "+output_file)
