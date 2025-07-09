import json
import os
import queue
import threading
import time
from datetime import datetime

from beartype.typing import Any, Callable, Dict, List, Optional
from tqdm import tqdm

from .latex_parser import (
    LatexEnvironmentNode,
    build_ast,
    extract_citations_from_ast,
    extract_section_info_from_ast,
    get_base_info,
    get_bib_names,
    load_bbl_info,
    load_bib_info,
)
from .utils import download_latex_source, search_arxiv_id


def build_citation_graph_node_info(
    working_path: str, tex_name: str, current_paper: str, get_citation_info: bool = True
) -> Dict[str, Any]:
    structured_data = None
    with open(os.path.join(working_path, tex_name), "r", encoding="utf-8") as tex_file:
        latex_code = tex_file.read()
    ast = build_ast(latex_code)
    res = {"title": None, "author": None, "doc_node": None}
    get_base_info(ast, res)
    title, author, doc_node = res["title"], res["author"], res["doc_node"]
    result = search_arxiv_id(current_paper)
    categories = []
    primary_category = None
    summary = None
    published = None
    if len(result):
        result = result[0]
        title = result.title
        author = result.authors[0].name
        categories = result.categories
        primary_category = result.primary_category
        summary = result.summary
        published = str(result.published)
    bib_names, bbl_nodes = get_bib_names(ast)
    # print(f"bib_names: {bib_names}")
    # print(f"bbl_nodes: {bbl_nodes}")
    structured_data = {
        "title": title,
        "author": author,
        "abstract": None,
        "citations": {},
        "refs": [],
        "table": [],
        "figure": [],
        "equations": [],
        "algorithm": [],
        "sections": {},
        "categories": categories,
        "published": published,
        "primary_category": primary_category,
        "summary": summary,
    }
    key2title = {}
    key2author = {}
    if doc_node:
        extract_section_info_from_ast(
            structured_data,
            [],
            [],
            [doc_node],
            [],
            (None, None, None),
            key2title,
            key2author,
            tex_name,
            False,
            [],
            "",
            working_path,
        )
        print(f"numbmer of citations in node info method: {len(structured_data['citations'])}")
    if get_citation_info:
        if len(bib_names) > 0:
            for bib in bib_names:
                bib = bib.replace(".bib", "").strip()
                if os.path.exists(os.path.join(working_path, bib + ".bib")):
                    bib_path = os.path.join(working_path, bib + ".bib")
                    load_bib_info(bib_path, key2title, key2author)
                else:
                    print(f"Cannot find the bib file {bib}.bib")
        if len(key2title) == 0:
            bbl_files = [f for f in os.listdir(working_path) if f.endswith(".bbl")]
            for file in bbl_files:
                try:
                    with open(
                        os.path.join(working_path, file), "r", encoding="utf-8"
                    ) as bbl_file:
                        bbl_text = bbl_file.read()
                    bbl_ast = build_ast(bbl_text)
                    bbl_nodes.extend(
                        [
                            node
                            for node in bbl_ast
                            if isinstance(node, LatexEnvironmentNode)
                            and node.environmentname == "thebibliography"
                        ]
                    )
                except Exception as e:
                    print(e)
                    continue
            for bbl_node in bbl_nodes:
                load_bbl_info(bbl_node, key2title, key2author)

        if len(key2title) == 0:
            print("Cannot find the bib file")
            return structured_data
        
        # It seems that even though we can find the path to bib/bbl file, we still might not extract the citation information?
        # print(f"key2title: {key2title}")
        # print(f"key2author: {key2author}")

    appendix = False
    prev_citation_contexts = []
    prev_ref_contexts = []
    next_context = ""
    next_ref_context = ""
    flattened_data = []
    recent_nodes = []
    env_stack = []
    current_section_name = None
    current_subsection_name = None
    current_subsubsection_name = None
    # The problem must lie in here!

    # print(f"doc_node: {doc_node}")

    if doc_node:
        # TODO: remove the print statement below
        print("Extracting citation information")
        extract_citations_from_ast(
            structured_data,
            flattened_data,
            recent_nodes,
            [doc_node],
            env_stack,
            (current_section_name, current_subsection_name, current_subsubsection_name),
            key2title,
            key2author,
            tex_name,
            appendix,
            prev_citation_contexts,
            prev_ref_contexts,
            next_context,
            next_ref_context,
            working_path,
        )
    return structured_data


def build_citation_graph(
    seed: List[str],
    source_path: str,
    working_path: str,
    output_path: str,
    debug_path: Optional[str],
    constraint: Callable[[str], bool],
    scale: int = 100,
    clear_source: bool = False,
    max_figure: int = 200000,
):
    # seed consists of a list of arxiv short_id
    # using BFS search to build the citation graph
    # download the latex source code into source_path
    # unzip the source code into working directory
    # save the structured data into output_path
    # constrain the paper published time to construct the train/test set
    # scale the number of papers to download
    BFS_que = queue.Queue()
    for seed_ in seed:
        BFS_que.put(seed_)
    visited = set()
    history = []
    cnt = 0
    endpoints_path = os.path.join(output_path, "endpoints")
    # create folders
    if not os.path.exists(working_path):
        os.makedirs(working_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(endpoints_path):
        os.makedirs(endpoints_path)
    if not os.path.exists(source_path):
        os.makedirs(source_path)
    if debug_path and not os.path.exists(debug_path):
        os.makedirs(debug_path)
    # print(os.path.join(output_path, 'history.json'))
    # print(os.path.exists(os.path.join(output_path, 'history.json')))
    if os.path.exists(os.path.join(output_path, "history.json")):
        with open(os.path.join(output_path, "history.json"), "r") as f:
            history = json.load(f)
        # print(history)
        for item in history:
            for item_ in item["extended"]:
                BFS_que.put(item_)
            cnt += 1
        visited = set([item["paper"] for item in history])
        print(cnt)
    # Create a tqdm progress bar
    with tqdm(total=scale, desc="Processing") as pbar:
        for _ in range(cnt):
            pbar.update(1)
        while BFS_que.qsize() > 0:
            current_paper = BFS_que.get()
            if current_paper in visited:
                continue
            get_citation_info = cnt < scale
            visited.add(current_paper)
            download_latex_source(current_paper, source_path)

            # clear the working directory
            os.system(f"rm -rf {working_path}/*")
            os.system(
                f"tar -xvf {os.path.join(source_path, current_paper+'.tar.gz')} -C {working_path} > /dev/null"
            )
            if clear_source:
                os.system(f"rm {os.path.join(source_path, current_paper+'.tar.gz')}")
            # get all files end with figure extensions in working directory
            fig_path_mapping = {}
            for root, dirs, files in os.walk(working_path):
                for file in files:
                    if (
                        file.endswith(".pdf")
                        or file.endswith(".png")
                        or file.endswith(".jpg")
                        or file.endswith(".jpeg")
                    ):
                        fig_path_mapping[file.split("/")[-1]] = os.path.join(root, file)
                        fig_path_mapping[file.split("/")[-1].lower()] = os.path.join(
                            root, file
                        )
            # os.system(f"tar -xvf {os.path.join(source_path, current_paper+'.tar.gz')} -C {working_path} > /dev/null")
            # find the tex file with document environment
            tex_files = [f for f in os.listdir(working_path) if f.endswith(".tex")]
            doc_files = []
            if len(tex_files) == 0:
                # os.system(f"cp {working_path} {os.path.join(debug_path, current_paper)}")
                if debug_path:
                    if not os.path.exists(os.path.join(debug_path, current_paper)):
                        os.makedirs(os.path.join(debug_path, current_paper))
                    os.system(
                        f"tar -xvf {os.path.join(source_path, current_paper+'.tar.gz')} -C {os.path.join(debug_path, current_paper)} > /dev/null"
                    )
                print(f"Failed to process {current_paper}")
                continue
            for tex_file in tex_files:
                try:
                    with open(
                        os.path.join(working_path, tex_file), "r", encoding="utf-8"
                    ) as tex_file_:
                        latex_code = tex_file_.read()
                    ast = build_ast(latex_code)
                except Exception as e:
                    print(e)
                    continue
                res = {"title": None, "author": None, "doc_node": None}
                get_base_info(ast, res)
                if res["doc_node"]:
                    doc_files.append(tex_file)
            structured_data = None
            print(f"doc_files: {doc_files}")
            for tex_file in doc_files:
                print(f"Processing file {tex_file}!")
                try:
                    structured_data = build_citation_graph_node_info(
                        working_path,
                        tex_file,
                        current_paper,
                        get_citation_info=get_citation_info,
                    )
                except Exception as e:
                    print(e)
            # move the figures to output_path/figures/current_paper/...
            if structured_data:
                if not os.path.exists(
                    os.path.join(output_path, "figures", current_paper)
                ):
                    os.makedirs(os.path.join(output_path, "figures", current_paper))
                if cnt < max_figure:
                    for figure in structured_data["figure"]:
                        for figure_path in figure["figure_paths"]:
                            if not os.path.exists(
                                os.path.join(working_path, figure_path)
                            ):
                                flag = False
                                file_key = figure_path.split("/")[-1].lower()
                                if fig_path_mapping.get(file_key):
                                    flag = True
                                    figure_path = fig_path_mapping[file_key]
                                for ending in [".pdf", ".png", ".jpg", ".jpeg"]:
                                    if os.path.exists(
                                        os.path.join(working_path, figure_path + ending)
                                    ):
                                        flag = True
                                        figure_path = figure_path + ending
                                        break
                                    if fig_path_mapping.get(file_key + ending):
                                        flag = True
                                        figure_path = fig_path_mapping[
                                            file_key + ending
                                        ]
                                        break
                                if not flag:
                                    print(f"Cannot find the figure {figure_path}")
                            os.system(
                                f"cp {os.path.join(working_path, figure_path)} {os.path.join(output_path, 'figures', current_paper, '_'.join(figure_path.split('/')))}"
                            )

                # os.system(f"cp {working_path} {os.path.join(debug_path, current_paper)}")
                if debug_path:
                    if not os.path.exists(os.path.join(debug_path, current_paper)):
                        os.makedirs(os.path.join(debug_path, current_paper))
                    os.system(
                        f"tar -xvf {os.path.join(source_path, current_paper+'.tar.gz')} -C {os.path.join(debug_path, current_paper)} > /dev/null"
                    )
                    print(f"Failed to process {current_paper}!")
                    continue
            if not structured_data or len(structured_data["citations"]) == 0:
                if structured_data:
                    with open(
                        os.path.join(endpoints_path, current_paper + ".json"), "w"
                    ) as f:
                        json.dump(structured_data, f)
                # os.system(f"cp {working_path} {os.path.join(debug_path, current_paper)}")
                if debug_path:
                    if not os.path.exists(os.path.join(debug_path, current_paper)):
                        os.makedirs(os.path.join(debug_path, current_paper))
                    os.system(
                        f"tar -xvf {os.path.join(source_path, current_paper+'.tar.gz')} -C {os.path.join(debug_path, current_paper)} > /dev/null"
                    )
                    print(f"Failed to process {current_paper}!")
                    continue
            if structured_data:
                if get_citation_info:
                    with open(
                        os.path.join(output_path, current_paper + ".json"), "w"
                    ) as f:
                        json.dump(structured_data, f)
                else:
                    with open(
                        os.path.join(endpoints_path, current_paper + ".json"), "w"
                    ) as f:
                        json.dump(structured_data, f)
            cnt += 1
            pbar.update(1)
            extended_list = []
            # apply constraint here.
            if (structured_data["citations"]) and (cnt <= scale):
                for citation in structured_data["citations"].values():
                    if citation["short_id"]:
                        id = citation["short_id"]
                        published = datetime.fromisoformat(citation["published"])
                        if (id not in visited) and constraint(published):
                            BFS_que.put(citation["short_id"])
                            extended_list.append(citation["short_id"])
            print(f"Extended list: {extended_list}")
            history.append(
                {
                    "paper": current_paper,
                    "extended": extended_list,
                    "endpoint": cnt >= scale,
                }
            )
            with open(os.path.join(output_path, "history.json"), "w") as f:
                json.dump(history, f)

    print(f"Finished processing {cnt} papers")


# Multithreaded version of the BFS function
def build_citation_graph_thread(
    seed: List[str],
    source_path: str,
    working_path: str,
    output_path: str,
    debug_path: Optional[str],
    constraint: Callable[[str], bool],
    scale: int = 100,
    num_threads: int = 4,
    clear_source: bool = False,
    max_figure: int = 200000,
): 
    # Shared BFS queue and visited set
    BFS_que = queue.Queue()
    print(f"seed: {seed}")
    for seed_ in seed:
        BFS_que.put((seed_, "True"))
    visited_lock = threading.Lock()
    history_lock = threading.Lock()
    cnt_lock = threading.Lock()
    visited = set()
    history = []
    cnt = 0
    start_time = time.time()
    endpoints_path = os.path.join(output_path, "endpoints")
    # create folders
    if not os.path.exists(working_path):
        os.makedirs(working_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(endpoints_path):
        os.makedirs(endpoints_path)
    if not os.path.exists(source_path):
        os.makedirs(source_path)
    if debug_path and not os.path.exists(debug_path):
        os.makedirs(debug_path)
    os.system(f"rm -rf {working_path}/*")
    # Load history if exists
    if os.path.exists(os.path.join(output_path, "history.json")):
        with open(os.path.join(output_path, "history.json"), "r") as f:
            history = json.load(f)
        # print(history)
        for item in history:
            for item_ in item["extended"]:
                BFS_que.put(item_)
            cnt += 1
        visited = set([item["paper"] for item in history])
        print(f"cnt: {cnt}")
    print(f"BFS_que.qsize(): {BFS_que.qsize()}")

    def process_papers():
        nonlocal cnt
        nonlocal BFS_que
        nonlocal visited
        nonlocal history

        # TODO: uncomment the print statement below
        # print(f"Thread {str(threading.get_ident())} Started processing")

        while True:
            try:
                try:
                    # print(f"BFS_que: {BFS_que}")
                    # print(f"Is BFS_que empty? {BFS_que.empty()}")
                    current_paper, published = BFS_que.get(
                        timeout=10
                    )  # Timeout to avoid infinite waiting
                    print(f"current paper: {current_paper}")
                    if published != "True":
                        published = datetime.fromisoformat(published)
                except queue.Empty:
                    # print("The queue is empty")
                    break
                print(f"Thread {str(threading.get_ident())} Processing {current_paper}")
                with cnt_lock:
                    cnt_ = cnt
                    # print(f"constraint(published): {constraint(published)}")
                    get_citation_info = (cnt_ < scale) and constraint(published)
                with visited_lock:
                    if current_paper in visited:
                        continue
                    visited.add(current_paper)

                # TODO: see what would happen if we comment out this line
                # Here, we separate the paper downloading and paper processing stages
                # download_latex_source(current_paper, source_path)

                # Clear the working directory
                thread_working_path = os.path.join(
                    working_path, str(threading.get_ident())
                )
                if not os.path.exists(thread_working_path):
                    os.makedirs(thread_working_path)
                os.system(f"rm -rf {thread_working_path}/*")
                os.system(
                    f"tar -xvf {os.path.join(source_path, current_paper + '.tar.gz')} -C {thread_working_path} > /dev/null"
                )
                if clear_source:
                    os.system(
                        f"rm {os.path.join(source_path, current_paper+'.tar.gz')}"
                    )
                # get all files end with figure extensions in working directory
                fig_path_mapping = {}
                for root, dirs, files in os.walk(thread_working_path):
                    for file in files:
                        if (
                            file.endswith(".pdf")
                            or file.endswith(".png")
                            or file.endswith(".jpg")
                            or file.endswith(".jpeg")
                        ):
                            fig_path_mapping[file.split("/")[-1]] = os.path.join(
                                root, file
                            )
                            fig_path_mapping[
                                file.split("/")[-1].lower()
                            ] = os.path.join(root, file)
                # Identify the main .tex file
                # print("Here!")
                tex_files = [
                    f for f in os.listdir(thread_working_path) if f.endswith(".tex")
                ]
                doc_files = []
                if len(tex_files) == 0:
                    # os.system(f"cp {working_path} {os.path.join(debug_path, current_paper)}")
                    if debug_path:
                        if not os.path.exists(os.path.join(debug_path, current_paper)):
                            os.makedirs(os.path.join(debug_path, current_paper))
                        os.system(
                            f"tar -xvf {os.path.join(source_path, current_paper+'.tar.gz')} -C {os.path.join(debug_path, current_paper)} > /dev/null"
                        )
                    print(
                        f"Thread {str(threading.get_ident())} Failed to process {current_paper}"
                    )
                    continue
                for tex_file in tex_files:
                    try:
                        with open(
                            os.path.join(thread_working_path, tex_file),
                            "r",
                            encoding="utf-8",
                        ) as tex_file_:
                            latex_code = tex_file_.read()
                    except Exception as e:
                        print(e)
                        continue
                    ast = build_ast(latex_code)
                    res = {"title": None, "author": None, "doc_node": None}
                    get_base_info(ast, res)
                    if res["doc_node"]:
                        doc_files.append(tex_file)

                structured_data = None
                for tex_file in doc_files:
                    print(
                        f"Thread {str(threading.get_ident())} Processing file {tex_file}"
                    )
                    try:
                        structured_data = build_citation_graph_node_info(
                            thread_working_path,
                            tex_file,
                            current_paper,
                            get_citation_info=get_citation_info,
                        )
                        
                        if structured_data.get("sections"):
                            break
                    except Exception as e:
                        print(e)

                ncitations = len(structured_data["citations"])
                print(f"number of citations collected: {ncitations}")

                if structured_data:
                    if not os.path.exists(
                        os.path.join(output_path, "figures", current_paper)
                    ):
                        os.makedirs(os.path.join(output_path, "figures", current_paper))
                    if cnt_ < max_figure:
                        for figure in structured_data["figure"]:
                            for figure_path in figure["figure_paths"]:
                                if not os.path.exists(
                                    os.path.join(thread_working_path, figure_path)
                                ):
                                    flag = False
                                    file_key = figure_path.split("/")[-1].lower()
                                    if fig_path_mapping.get(file_key):
                                        flag = True
                                        figure_path = fig_path_mapping[file_key]
                                    for ending in [".pdf", ".png", ".jpg", ".jpeg"]:
                                        if os.path.exists(
                                            os.path.join(
                                                thread_working_path,
                                                figure_path + ending,
                                            )
                                        ):
                                            flag = True
                                            figure_path = figure_path + ending
                                            break
                                        if fig_path_mapping.get(file_key + ending):
                                            flag = True
                                            figure_path = fig_path_mapping[
                                                file_key + ending
                                            ]
                                            break
                                    if not flag:
                                        print(f"Cannot find the figure {figure_path}")
                                os.system(
                                    f"cp {os.path.join(thread_working_path, figure_path)} {os.path.join(output_path, 'figures', current_paper, '_'.join(figure_path.split('/')))}"
                                )
                if not structured_data or len(structured_data["citations"]) == 0:
                    if structured_data:
                        with open(
                            os.path.join(endpoints_path, current_paper + ".json"), "w"
                        ) as f:
                            json.dump(structured_data, f)
                    if debug_path:
                        if not os.path.exists(os.path.join(debug_path, current_paper)):
                            os.makedirs(os.path.join(debug_path, current_paper))
                        os.system(
                            f"tar -xvf {os.path.join(source_path, current_paper+'.tar.gz')} -C {os.path.join(debug_path, current_paper)} > /dev/null"
                        )
                        print(
                            f"Thread {str(threading.get_ident())} Failed to process {current_paper}!"
                        )
                        continue

                if structured_data:
                    if get_citation_info:
                        with open(
                            os.path.join(output_path, current_paper + ".json"), "w"
                        ) as f:
                            json.dump(structured_data, f)
                    else:
                        with open(
                            os.path.join(endpoints_path, current_paper + ".json"), "w"
                        ) as f:
                            json.dump(structured_data, f)
                with cnt_lock:
                    cnt += 1
                    cnt__ = cnt
                    print(
                        f"Thread {str(threading.get_ident())} Finished processing {current_paper} ({cnt}/{scale}) Time elapsed: {time.time()-start_time:.2f}s"
                    )
                # Extend BFS queue based on citations and constraint
                extended_list = []
                if (structured_data["citations"]) and (cnt_ < scale):
                    for citation in structured_data["citations"].values():
                        if citation["short_id"]:
                            id = citation["short_id"]
                            published = datetime.fromisoformat(citation["published"])
                            with visited_lock:
                                if id not in visited:
                                    BFS_que.put((id, str(published)))
                                    extended_list.append((id, str(published)))
                published = str(published)
                with history_lock:
                    history.append(
                        {
                            "paper": current_paper,
                            "cnt": cnt__,
                            "extended": extended_list,
                            "time_used": time.time() - start_time,
                            "endpoint": cnt_ >= scale,
                            "que_len": BFS_que.qsize(),
                            "published": published,
                        }
                    )
                    with open(os.path.join(output_path, "history.json"), "w") as f:
                        json.dump(history, f)
            except Exception as e:
                print(e)
                print(
                    f"Thread {str(threading.get_ident())} Failed to process {current_paper}"
                )
                with history_lock:
                    history.append(
                        {
                            "paper": current_paper,
                            "cnt": cnt_,
                            "extended": [],
                            "time_used": time.time() - start_time,
                            "endpoint": cnt_ >= scale,
                            "que_len": BFS_que.qsize(),
                            "published": str(published),
                        }
                    )
                    with open(os.path.join(output_path, "history.json"), "w") as f:
                        json.dump(history, f)
                continue

    # Create and start threads
    num_threads = num_threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=process_papers)
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print(f"Thread {str(threading.get_ident())} Finished processing {cnt} papers")
