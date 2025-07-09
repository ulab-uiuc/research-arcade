import glob
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

global discarded
discarded = 0

ENV_RE = re.compile(
    r"""
    hline
  | \\begin\{(?:figure\*?|subfigure|table\*?|subtable|
               tabular|tabularx|tabulary|longtable|array)\}
  | \\(?:multicolumn|multirow)
  | \\(?:cline)
  | \\(?:toprule|midrule|bottomrule|cmidrule)
""",
    re.IGNORECASE | re.VERBOSE,
)


def clean_latex_code(latex_str: str) -> str:
    # Remove LaTeX commands
    cleaned_str = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", "", latex_str)
    return cleaned_str.strip()


class PaperGraphProcessor:
    """Handles the processing of academic papers and their figures."""

    def clear_latex_format(self, text: str) -> str:
        """Clear LaTeX formatting from text."""
        # Remove LaTeX commands, e.g., \em, \textbf, \textit, etc.
        plain_text = re.sub(r"\\[a-zA-Z]+\s?", "", text)

        # Remove curly braces but keep the text inside
        plain_text = re.sub(r"[{}]", "", plain_text)

        # Strip any leading/trailing whitespace
        plain_text = plain_text.strip()

        return plain_text

    def __init__(
        self, data_dir: str, figures_dir: str, output_dir: str, threshold: float = 0.8
    ):
        """Initialize with directory paths for data processing.

        Args:
            data_dir: Directory containing paper JSON files
            figures_dir: Directory containing figure files
            output_dir: Directory for processed output
        """
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        self.output_dir = output_dir
        self.output_figure_dir = os.path.join(output_dir, "figures")
        self.output_metadata = os.path.join(output_dir, "metadata.jsonl")
        self.output_text_nodes = os.path.join(output_dir, "text_nodes.jsonl")
        self.output_table_nodes = os.path.join(output_dir, "table_nodes.jsonl")
        self.output_figure_nodes = os.path.join(output_dir, "figure_nodes.jsonl")
        self.output_paper_nodes = os.path.join(output_dir, "paper_nodes.jsonl")
        self.edge_metadata = []
        self.figure_nodes = []
        self.table_nodes = []
        self.text_nodes = []
        self.paper_nodes = []
        self.convert_codes = []
        self.paper_id2node = {}
        self.threshold = threshold
        self.node_id_counter = 0
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.output_figure_dir, exist_ok=True)


    @staticmethod
    def list_json_files(directory: str) -> List[str]:
        """Get all JSON files in directory."""
        return glob.glob(os.path.join(directory, "*.json"))

    @staticmethod
    def clean_filename(filename: str) -> str:
        """Remove file extensions and standardize filename."""
        return filename.replace(".pdf", "").replace(".png", "").replace(".jpg", "")

    def load_paper(self, file_path: str) -> Tuple[Optional[dict], str]:
        """Load paper JSON and get its figure paths."""
        try:
            arxiv_id = os.path.basename(file_path).split(".json")[0]
            with open(file_path, "r") as f:
                paper_data = json.load(f)
            return paper_data, arxiv_id
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, ""

    def get_node_id(self):
        """Get a unique node ID."""
        self.node_id_counter += 1
        return self.node_id_counter

    @staticmethod
    def _create_label_mapping(objects: List[Any]) -> Dict[str, List[Any]]:
        """Create mapping from labels to objects."""
        label_to_objects = {}
        for object in objects:
            if object["label"] not in label_to_objects:
                label_to_objects[object["label"]] = []
            label_to_objects[object["label"]].append(object)
        return label_to_objects

    def extract_label(self, text: str) -> str:
        """Extract the label from a LaTeX figure reference."""
        pattern = r"\\label\{([^}]+)\}"
        match = re.search(pattern, text)
        return match.group(1) if match else ""

    def find_references(self, text: str) -> List[str]:
        """Find all figure references in text."""
        pattern = r"\\ref\{([^}]+)\}"
        return re.findall(pattern, text)

    def find_cites(self, text: str) -> List[str]:
        """Find all figure references in text."""
        pattern = r"\\cite\w*\{([^}]+)\}"
        return re.findall(pattern, text)

    def create_figure_node(self, figure: dict, paper_id: str) -> Dict:
        """Create a figure node for the graph."""
        have_paths = False
        if figure["figure_paths"]:
            content = [
                "_".join(fig_path.split("_")) for fig_path in figure["figure_paths"]
            ]
            have_paths = True
        else:
            content = figure["original"]
        return {
            "id": "figures_" + str(self.get_node_id()),
            "content": content,
            "fig_files": have_paths,
            "label": figure["label"],
            "caption": self.clear_latex_format(figure["caption"]),
            "parent_paper_id": paper_id,
            "type": "figureNode",
        }

    def create_table_node(self, table: dict, paper_id: str) -> Dict:
        """Create a table node for the graph."""
        return {
            "id": "table_" + str(self.get_node_id()),
            "label": table["label"],
            "caption": table["caption"],
            "content": table["tabular"],
            "parent_paper_id": paper_id,
            "type": "tableNode",
        }

    def create_section_node(self, section: str) -> Dict:
        """Create a section node for the graph."""
        return {
            "id": self.get_node_id(),
            "content": section,
            "type": "sectionNode",
        }

    def create_paper_node(self, paper_id, abstract, title):
        return {
            "id": "paper_" + str(self.get_node_id()),
            "paper_id": paper_id,
            "abstract": abstract,
            "title": title,
            "type": "paperNode",
        }

    def create_text_node(
        self, text: str, key2citation: Dict[str, Any], label2id: Dict[str, int]
    ) -> Dict:
        """Create a text node for the graph."""
        cites_ = self.find_cites(text)
        cites = []
        for cite in cites_:
            cites.extend(cite.split(","))
        cites = [cite.strip() for cite in cites]
        cites = [
            key2citation[cite]["short_id"] for cite in cites if cite in key2citation
        ]
        refs = self.find_references(text)
        refs = [label2id[ref] for ref in refs if ref in label2id]
        isolation = (len(refs) == 0) and (len(cites) == 0)
        return {
            "id": "text_" + str(self.get_node_id()),
            "cites": cites,
            "refs": refs,
            "content": text,
            "type": "textNode",
            "isolation": isolation,
        }

    def process_paper(self, data: dict, paper_id: str) -> Optional[Dict]:
        citations = {}
        figures = data["figure"]
        table = data["table"]
        label2id = {}
        neighbors = []
        nodes = {}
        global discarded
        temp_figure_nodes = []
        figure_nodes_set = {}
        temp_table_nodes = []
        table_nodes_set = {}
        touched_refs = set()
        for figure in figures:
            if figure["label"]:
                figure["label"] = self.extract_label(figure["label"])
                figure_node = self.create_figure_node(figure, paper_id)
                nodes[figure_node["id"]] = figure_node
                figure_nodes_set[figure_node["id"]] = figure_node
                label2id[figure["label"]] = figure_node["id"]
                temp_figure_nodes.append(figure_node)
        for table in table:
            if table["label"]:
                table["label"] = self.extract_label(table["label"])
                table_node = self.create_table_node(table, paper_id)
                table_nodes_set[table_node["id"]] = table_node
                nodes[table_node["id"]] = table_node
                label2id[table["label"]] = table_node["id"]
                temp_table_nodes.append(table_node)

        for key, citation in data["citations"].items():
            if citation["similar_score"] and (
                citation["similar_score"] > self.threshold
            ):
                citations[key] = citation
                neighbors.append(citation["short_id"])

        for section, content in data["sections"].items():
            # section_node = self.create_section_node(section)
            # nodes[section_node['id']] = section_node
            chunks = [
                chunk.strip()
                for chunk in content["content"].split("\n\n")
                if chunk.strip()
            ]
            pre_node = None
            for chunk in chunks:
                ########################################
                if ENV_RE.search(chunk):
                    continue
                ########################################
                text_node = self.create_text_node(chunk, citations, label2id)
                text_node["paper_id"] = paper_id
                text_node["section"] = section
                if True:  # not text_node['isolation']:
                    nodes[text_node["id"]] = text_node
                    self.text_nodes.append(text_node)
                    for cite in text_node["cites"]:
                        if cite in self.paper_id2node:
                            target = self.paper_id2node[cite]
                            source = self.paper_id2node[paper_id]
                            self.edge_metadata.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "type": "citing_edge",
                                }
                            )
                            self.edge_metadata.append(
                                {
                                    "source": text_node["id"],
                                    "target": source,
                                    "type": "current_paper_edge",
                                }
                            )
                        else:
                            print("No paper found for ", cite)
                    for ref in text_node["refs"]:
                        if ref in figure_nodes_set:
                            figure_node = figure_nodes_set[ref]
                            if not figure_node["fig_files"]:
                                continue
                            if len(figure_node["content"]) != 1:
                                discarded += 1
                                continue
                            fig_path = figure_node["content"][0]
                            fig_id = "_".join(fig_path.split("/"))
                            from_path = os.path.join(self.figures_dir, paper_id, fig_id)
                            if not os.path.isfile(from_path):
                                continue
                            # print(figure_node['content'])
                            target = figure_node["content"][0]
                            target = figure_node["id"] + "." + target.split(".")[-1]
                            if target.endswith("pdf"):
                                target = target.replace("pdf", "png")
                            if target.endswith("eps"):
                                target = target.replace("eps", "png")
                            type_ = "ref_figure"
                        elif ref in table_nodes_set:
                            target = ref
                            print(table_nodes_set[ref]["label"])
                            if table_nodes_set[ref]["content"]:
                                type_ = "ref_table"
                            else:
                                continue
                        touched_refs.add(ref)
                        self.edge_metadata.append(
                            {
                                "source": text_node["id"],
                                "target": target,
                                "type": type_,
                            }
                        )
                    # for cite in text_node['cites']:
                    #    self.edge_metadata.append({
                    #        'source': 'text_'+str(text_node['id']),
                    #        'target': cite,
                    #        'type': 'cites',
                    #    })
                    if pre_node:
                        self.edge_metadata.append(
                            {
                                "source": pre_node["id"],
                                "target": text_node["id"],
                                "type": "adjacent_chunk_edge",
                            }
                        )
                        self.edge_metadata.append(
                            {
                                "source": text_node["id"],
                                "target": pre_node["id"],
                                "type": "adjacent_chunk_edge",
                            }
                        )
                    pre_node = text_node

        for figure_node in temp_figure_nodes:
            if figure_node["id"] in touched_refs:
                fig_ids = []
                if figure_node["fig_files"]:
                    for fig_path in figure_node["content"]:
                        fig_id = "_".join(fig_path.split("/"))
                        new_fig_id = figure_node["id"] + "." + fig_id.split(".")[-1]
                        from_path = os.path.join(self.figures_dir, paper_id, fig_id)
                        to_path = os.path.join(self.output_figure_dir, new_fig_id)
                        # check if figure file exists
                        if os.path.isfile(from_path):
                            shutil.copy(from_path, to_path)
                            if from_path.endswith("png"):
                                shutil.copy(from_path, to_path)
                                fig_ids.append(new_fig_id)
                            elif from_path.endswith("pdf"):
                                to_path = to_path.replace("pdf", "png")
                                new_fig_id = new_fig_id.replace("pdf", "png")
                                fig_ids.append(new_fig_id)
                                self.convert_codes.append(
                                    f"convert -density 300 {from_path} {to_path}"
                                )
                            elif from_path.endswith("eps"):
                                to_path = to_path.replace("eps", "png")
                                new_fig_id = new_fig_id.replace("eps", "png")
                                fig_ids.append(new_fig_id)
                                self.convert_codes.append(
                                    f"convert -density 300 {from_path} {to_path}"
                                )
                    figure_node["content"] = fig_ids
                    if len(fig_ids) == 1:
                        self.figure_nodes.append(figure_node)
        for table_node in temp_table_nodes:
            if table_node["id"] in touched_refs:
                self.table_nodes.append(table_node)

    def save_jsonl(self, data: List[Dict], output_path: str, save_mode: str = "w"):
        """Save data to JSONL file."""
        with open(output_path, save_mode) as f:
            for line in data:
                f.write(json.dumps(line) + "\n")

    def save_processed_data(self, save_mode: str = "w"):
        """Save processed paper data to JSON file."""
        self.save_jsonl(self.edge_metadata, self.output_metadata, save_mode)
        print(
            "Total nodes: ",
            len(self.paper_nodes)
            + len(self.figure_nodes)
            + len(self.table_nodes)
            + len(self.text_nodes),
        )
        print("Total edges: ", len(self.edge_metadata))
        print("Paper nodes: ", len(self.paper_nodes))
        print("Figure nodes: ", len(self.figure_nodes))
        print("Table nodes: ", len(self.table_nodes))
        print("Text nodes: ", len(self.text_nodes))
        self.save_jsonl(self.paper_nodes, self.output_paper_nodes, save_mode)
        self.save_jsonl(self.figure_nodes, self.output_figure_nodes, save_mode)
        self.save_jsonl(self.table_nodes, self.output_table_nodes, save_mode)
        self.save_jsonl(self.text_nodes, self.output_text_nodes, save_mode)
        with open("./convert.sh", "w") as f:
            for code in self.convert_codes:
                f.write(code + "\n")

    def process_all_papers(self):
        """Process all papers in the data directory."""
        cnt = 0
        for paper_path in tqdm(self.list_json_files(self.data_dir)):
            if "history" in paper_path:
                continue
            paper_data, paper_id = self.load_paper(paper_path)
            if paper_data:
                abstract = paper_data["abstract"]
                if not abstract:
                    abstract = ""
                paper_node = self.create_paper_node(
                    paper_id, clean_latex_code(abstract), paper_data["title"]
                )
                self.paper_nodes.append(paper_node)
                self.paper_id2node[paper_id] = paper_node["id"]
                # for key, cite in paper_data['citations'].items():
                #    if cite['similar_score'] and (cite['similar_score'] > self.threshold):
                #        id = cite['short_id']
                #        abstract = cite['abstract']
                #        title = cite['title']
                #        paper_node = self.create_paper_node(id, clean_latex_code(abstract), title)
                #        self.paper_nodes.append(paper_node)
                #        self.paper_id2node[id] = paper_node['id']

        # self.save_processed_data('w')
        for paper_path in tqdm(self.list_json_files(self.data_dir)):
            if "history" in paper_path:
                continue
            paper_data, paper_id = self.load_paper(paper_path)
            if paper_data:
                cnt += 1
                print(paper_id)
                self.process_paper(paper_data, paper_id)
                citations = {}
                for key, cite in paper_data["citations"].items():
                    if cite["similar_score"] and (
                        cite["similar_score"] > self.threshold
                    ):
                        citations[key] = cite
                paper_data["citations"] = citations
                os.makedirs(os.path.join(self.output_dir, "papers"), exist_ok=True)
                # shutil.copy(paper_path, os.path.join(self.output_dir, 'papers', os.path.basename(paper_path)))
                with open(
                    os.path.join(
                        self.output_dir, "papers", os.path.basename(paper_path)
                    ),
                    "w",
                ) as f:
                    json.dump(paper_data, f)
                # if cnt % 100 == 0:
                #    self.save_processed_data('w')
        print("Paper count: ", cnt)
        self.save_processed_data("w")
        print(discarded)


    def process_papers(self, paper_paths):
        cnt  = 0
        for paper_path in tqdm(paper_paths):
            if "history" in paper_path:
                continue
            paper_data, paper_id = self.load_paper(paper_path)
            if paper_data:
                abstract = paper_data["abstract"]
                if not abstract:
                    abstract = ""
                paper_node = self.create_paper_node(
                    paper_id, clean_latex_code(abstract), paper_data["title"]
                )
                self.paper_nodes.append(paper_node)
                self.paper_id2node[paper_id] = paper_node["id"]
                # for key, cite in paper_data['citations'].items():
                #    if cite['similar_score'] and (cite['similar_score'] > self.threshold):
                #        id = cite['short_id']
                #        abstract = cite['abstract']
                #        title = cite['title']
                #        paper_node = self.create_paper_node(id, clean_latex_code(abstract), title)
                #        self.paper_nodes.append(paper_node)
                #        self.paper_id2node[id] = paper_node['id']

        # self.save_processed_data('w')
        for paper_path in tqdm(paper_paths):
            if "history" in paper_path:
                continue
            paper_data, paper_id = self.load_paper(paper_path)
            if paper_data:
                cnt += 1
                print(paper_id)
                self.process_paper(paper_data, paper_id)
                citations = {}
                for key, cite in paper_data["citations"].items():
                    if cite["similar_score"] and (
                        cite["similar_score"] > self.threshold
                    ):
                        citations[key] = cite
                paper_data["citations"] = citations
                os.makedirs(os.path.join(self.output_dir, "papers"), exist_ok=True)
                # shutil.copy(paper_path, os.path.join(self.output_dir, 'papers', os.path.basename(paper_path)))
                with open(
                    os.path.join(
                        self.output_dir, "papers", os.path.basename(paper_path)
                    ),
                    "w",
                ) as f:
                    json.dump(paper_data, f)

        print("Paper count: ", cnt)
        self.save_processed_data("w")
        print(discarded)