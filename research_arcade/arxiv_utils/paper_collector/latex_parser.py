import os
import re

import bibtexparser
from beartype.typing import Any, Dict, List, Tuple, Union
from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
    get_default_latex_context_db,
)
from pylatexenc.macrospec import MacroSpec
from .utils import query_and_match

_SUBFIGURE_ENVS = {
    "figure",
    "figure*",
    "minipage",
    "subfigure",          # from subcaption / subfigure packages
    "wrapfigure",
    "floatrow",
    "sidewaysfigure",
}

_SUBFIGURE_MACROS = {"subfloat"}


def clean_latex_code(latex_str: str) -> str:
    # Remove LaTeX commands
    cleaned_str = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", "", latex_str)
    return cleaned_str


def clean_latex_format(latex_text: str) -> str:
    # Remove LaTeX commands, e.g., \em, \textbf, \textit, etc.
    plain_text = re.sub(r"\\[a-zA-Z]+\s?", "", latex_text)

    # Remove curly braces but keep the text inside
    plain_text = re.sub(r"[{}]", "", plain_text)

    # Strip any leading/trailing whitespace
    plain_text = plain_text.strip()

    return plain_text


def is_stop_node(node: str) -> bool:
    if (
        "appendix" in node
        or "label" in node
        or "caption" in node
        or "paragraph" in node
    ):
        return True
    if "input" in node or "include" in node:
        return True
    if "section" in node:
        return True
    return False


def get_last_paragraph(text: str) -> Tuple[str, bool]:
    # Split text into paragraph and handle common punctuation
    paragraph = re.split(r"\n\n|\\\\|\}\n", text.strip())
    if not paragraph:
        return None, False

    last_paragraph = paragraph[-1]
    is_complete = len(paragraph) > 1
    return last_paragraph, is_complete


def get_first_paragraph(text: str) -> Tuple[str, bool]:
    # Split text into paragraph and handle common punctuation
    paragraph = re.split(r"\n\n|\\\\|\}\n", text.strip())
    if not paragraph:
        return None, False

    last_paragraph = paragraph[0]
    is_complete = len(paragraph) > 1
    return last_paragraph, is_complete


def get_last_sentence(text: str) -> Tuple[str, bool]:
    # Split text into sentences and handle common punctuation
    sentences = re.split(r"(?<=[.!?])\s+|\n+|\\\\|\&", text.strip())
    if not sentences:
        return None, False

    last_sentence = sentences[-1]
    is_complete = len(sentences) > 1
    if len(last_sentence) > 0 and last_sentence[-1] in ".?!":
        last_sentence = ""
        is_complete = True
    return last_sentence, is_complete


def get_first_sentence(text: str) -> Tuple[str, bool]:
    # Split text into sentences and handle common punctuation
    sentences = re.split(r"(?<=[.!?])\s+|\n+|\\\\|\&", text.strip())
    if not sentences:
        return None, False

    last_sentence = sentences[0]
    is_complete = len(sentences) > 1
    if len(last_sentence) > 0 and last_sentence[-1] in ".?!\n":
        is_complete = True
    return last_sentence, is_complete


# flat_data: for storing the text content of the current section, ignoring the node if it is a subnode of a environment
# recent_nodes: storing recent node texts for citation context
def extract_section_info_from_ast(
    structured_data: Dict[str, Any],
    flat_data: List[str],
    recent_nodes: List[str],
    nodes: List[Union[LatexCharsNode, LatexMacroNode, LatexEnvironmentNode]],
    env_stack: List[str],
    section_names: Dict[str, str],
    key2title: Dict[str, str],
    key2author: Dict[str, str],
    current_file: str,
    appendix: bool,
    prev_citation_contexts: List[Dict[str, Any]] = [],
    next_context: str = "",
    working_path: str = "",
    flag: bool = False,
):
    try:
        in_environment = len(env_stack) > 1
        (
            current_section_name,
            current_subsection_name,
            current_subsubsection_name,
        ) = section_names
        for node in nodes:
            if isinstance(node, LatexCommentNode):
                continue
            scope_name = ""
            node_text = node.latex_verbatim()
            if (not in_environment) or (not isinstance(node, LatexEnvironmentNode)):
                flat_data.append(node_text)
            if isinstance(node, LatexEnvironmentNode):
                scope_name = node.environmentname
                if "table" in node.environmentname:
                    table_info = parse_tableEnv(node)
                    if table_info:
                        structured_data["table"].append(table_info)

                    # flat_data.append(table_info)
                    # structured_data['table'].append(table_info)
                elif "figure" in node.environmentname:
                    figure_info = parse_figureEnv(node)
                    if figure_info:
                        structured_data["figure"].append(figure_info)

                    # flat_data.append(figure_info)
                    # structured_data['figure'].append(figure_info)
                elif (
                    "equation" in node.environmentname
                    or "align" in node.environmentname
                ):
                    equation_info = parse_equationEnv(node)
                    if equation_info:
                        structured_data["equations"].append(equation_info)
                    if not in_environment:
                        flat_data.append(equation_info)
                elif "algorithm" in node.environmentname:
                    algorithm_info = parse_algorithmEnv(node)
                    if algorithm_info:
                        structured_data["algorithm"].append(algorithm_info)
                    if not in_environment:
                        flat_data.append(algorithm_info)
                elif (
                    "itemize" in node.environmentname
                    or "enumerate" in node.environmentname
                ):
                    itemize_info = parse_itemizeEnv(node)
                    if not in_environment:
                        flat_data.append(itemize_info)
                elif node.environmentname == "minipage":
                    if "tabular" in node.latex_verbatim():
                        table_info = parse_tableEnv(node)
                        if table_info:
                            structured_data["table"].append(table_info)
                    elif "figure" in node.latex_verbatim():
                        figure_info = parse_figureEnv(node)
                        if figure_info:
                            structured_data["figure"].append(figure_info)
                elif node.environmentname == "document":
                    pass
                elif node.environmentname == "tabular":
                    pass
                else:
                    unknown_info = parse_unknownEnv(node)
                    if not in_environment:
                        flat_data.append(unknown_info)
                    # if unknown_info:
                    #    structured_data['unknown'].append(unknown_info)

                if node.environmentname == "abstract":
                    abstract = node.latex_verbatim()
                    if "\\input" in abstract:
                        if not in_environment:
                            flat_data.pop()
                        input = (
                            re.search(r".*?\\input{(.*)}", abstract)
                            .group(1)
                            .replace(".tex", "")
                        )
                        try:
                            with open(
                                os.path.join(working_path, f"{input}.tex"),
                                "r",
                                encoding="utf-8",
                            ) as tex_file:
                                latex_code = tex_file.read()
                            abstract = latex_code
                        except Exception as e:
                            print(e)
                    structured_data["abstract"] = abstract
            if isinstance(node, LatexMacroNode):
                if node.macroname == "appendix":
                    if not in_environment:
                        flat_data.pop()
                    appendix = True
                if node.macroname == "input" or node.macroname == "include":
                    if not in_environment:
                        flat_data.pop()
                    try:
                        input = safe_extract_chars(node.nodeargd.argnlist[0])
                        input = input.replace(".tex", "").strip()
                        with open(
                            os.path.join(working_path, f"{input}.tex"),
                            "r",
                            encoding="utf-8",
                        ) as tex_file:
                            latex_code = tex_file.read()
                        ast = build_ast(latex_code)

                        if f"{input}.tex" != current_file:
                            (
                                current_section_name,
                                current_subsection_name,
                                current_subsubsection_name,
                            ) = extract_section_info_from_ast(
                                structured_data,
                                flat_data,
                                recent_nodes,
                                ast,
                                env_stack,
                                (
                                    current_section_name,
                                    current_subsection_name,
                                    current_subsubsection_name,
                                ),
                                key2title,
                                key2author,
                                "{input}.tex",
                                appendix,
                                prev_citation_contexts,
                                next_context,
                                working_path,
                                True,
                            )

                    except Exception as e:
                        print(e)
                if node.macroname == "section":
                    section_name = ""
                    for arg_node in node.nodeargd.argnlist:
                        if (
                            isinstance(arg_node, LatexGroupNode)
                            and arg_node.delimiters[0] == "{"
                        ):
                            section_name = clean_latex_code(
                                arg_node.latex_verbatim()
                            ).strip("{}")
                    # assert section_name
                    if current_section_name:
                        structured_data["sections"][current_section_name] = {
                            "content": "".join(flat_data[:-1]),
                            "appendix": appendix,
                        }
                    flat_data.clear()
                    current_section_name = section_name
                    current_subsection_name = None
                    current_subsubsection_name = None
                if node.macroname == "subsection":
                    subsection_name = ""
                    for arg_node in node.nodeargd.argnlist:
                        if (
                            isinstance(arg_node, LatexGroupNode)
                            and arg_node.delimiters[0] == "{"
                        ):
                            subsection_name = clean_latex_code(
                                arg_node.latex_verbatim()
                            ).strip("{}")
                    # assert subsection_name
                    current_subsection_name = subsection_name
                    current_subsubsection_name = None
                if node.macroname == "subsubsection":
                    subsubsection_name = ""
                    for arg_node in node.nodeargd.argnlist:
                        if (
                            isinstance(arg_node, LatexGroupNode)
                            and arg_node.delimiters[0] == "{"
                        ):
                            subsubsection_name = clean_latex_code(
                                arg_node.latex_verbatim()
                            ).strip("{}")
                    # assert subsubsection_name
                    current_subsubsection_name = subsubsection_name
            # Recursively check child nodes if the node is an environment (e.g., document or section)
            if hasattr(node, "nodelist"):
                env_stack.append(scope_name)
                (
                    current_section_name,
                    current_subsection_name,
                    current_subsubsection_name,
                ) = extract_section_info_from_ast(
                    structured_data,
                    flat_data,
                    recent_nodes,
                    node.nodelist,
                    env_stack,
                    (
                        current_section_name,
                        current_subsection_name,
                        current_subsubsection_name,
                    ),
                    key2title,
                    key2author,
                    current_file,
                    appendix,
                    prev_citation_contexts,
                    next_context,
                    working_path,
                    flag,
                )
                env_stack.pop()

        if (
            nodes
            and isinstance(nodes[0], LatexEnvironmentNode)
            and (nodes[0].environmentname == "document")
            and (current_section_name)
        ):
            structured_data["sections"][current_section_name] = {
                "content": "".join(flat_data),
                "appendix": appendix,
            }
        return (
            current_section_name,
            current_subsection_name,
            current_subsubsection_name,
        )
    except Exception as e:
        print(e)
        print("Failed to process the node")
        return (
            current_section_name,
            current_subsection_name,
            current_subsubsection_name,
        )


def extract_citations_from_ast(
    structured_data: Dict[str, Any],
    flat_data: List[str],
    recent_nodes: List[str],
    nodes: List[Union[LatexCharsNode, LatexMacroNode, LatexEnvironmentNode]],
    env_stack: List[str],
    section_names: Tuple[str, str, str],
    key2title: Dict[str, str],
    key2author: Dict[str, str],
    current_file: str,
    appendix: bool,
    prev_citation_contexts: List[Dict[str, Any]] = [],
    prev_ref_contexts: List[Dict[str, Any]] = [],
    next_context: str = "",
    next_ref_context: str = "",
    working_path: str = "",
    flag: bool = False,
):  
    try:
        (
            current_section_name,
            current_subsection_name,
            current_subsubsection_name,
        ) = section_names
        in_environment = len(env_stack) > 1
        for node in nodes:
            if isinstance(node, LatexCommentNode):
                continue
            scope_name = ""
            node_text = node.latex_verbatim()
            if isinstance(node, LatexEnvironmentNode):
                scope_name = node.environmentname
            if not in_environment:
                flat_data.append(node_text)
            is_node_group = hasattr(node, "nodelist")
            if not is_node_group:
                recent_nodes.append(node_text)

            """if (not is_node_group) and (len(prev_ref_contexts)>0):
                next_paragraph, is_complete = get_first_paragraph(node_text)
                next_ref_context += next_paragraph
                if is_complete:
                    for ref_context in prev_ref_contexts:
                        ref_context['next_context'] = next_ref_context[ref_context['_pos']:]
                        ref_context.pop('_pos')
                    prev_ref_contexts.clear()
                    next_ref_context = ''
            """

            if (not is_node_group) and (len(prev_citation_contexts) > 0):
                next_sentence, is_complete = get_first_sentence(node_text)
                next_context += next_sentence
                if is_complete:
                    for citation_context in prev_citation_contexts:
                        citation_context["next_context"] = next_context[
                            citation_context["_pos"] :
                        ]
                        citation_context.pop("_pos")
                    prev_citation_contexts.clear()
                    next_context = ""

            if isinstance(node, LatexMacroNode):
                if node.macroname == "appendix":
                    appendix = True
                    if not in_environment:
                        flat_data.pop()
                if node.macroname == "input" or node.macroname == "include":
                    if not in_environment:
                        flat_data.pop()
                    try:
                        input = safe_extract_chars(node.nodeargd.argnlist[0])
                        input = input.replace(".tex", "").strip()
                        with open(
                            os.path.join(working_path, f"{input}.tex"),
                            "r",
                            encoding="utf-8",
                        ) as tex_file:
                            latex_code = tex_file.read()
                        ast = build_ast(latex_code)
                        if f"{input}.tex" != current_file:
                            (
                                current_section_name,
                                current_subsection_name,
                                current_subsubsection_name,
                                next_ref_context,
                                next_context,
                            ) = extract_citations_from_ast(
                                structured_data,
                                flat_data,
                                recent_nodes,
                                ast,
                                env_stack,
                                (
                                    current_section_name,
                                    current_subsection_name,
                                    current_subsubsection_name,
                                ),
                                key2title,
                                key2author,
                                f"{input}.tex",
                                appendix,
                                prev_citation_contexts,
                                prev_ref_contexts,
                                next_context,
                                next_ref_context,
                                working_path,
                                flag,
                            )
                    except Exception as e:
                        print(e)

                if node.macroname == "section":
                    section_name = ""
                    for arg_node in node.nodeargd.argnlist:
                        if (
                            isinstance(arg_node, LatexGroupNode)
                            and arg_node.delimiters[0] == "{"
                        ):
                            section_name = clean_latex_code(
                                arg_node.latex_verbatim()
                            ).strip("{}")
                    flat_data.clear()
                    current_section_name = section_name
                    current_subsection_name = None
                    current_subsubsection_name = None
                if node.macroname == "subsection":
                    subsection_name = ""
                    for arg_node in node.nodeargd.argnlist:
                        if (
                            isinstance(arg_node, LatexGroupNode)
                            and arg_node.delimiters[0] == "{"
                        ):
                            subsection_name = clean_latex_code(
                                arg_node.latex_verbatim()
                            ).strip("{}")
                    # assert subsection_name
                    current_subsection_name = subsection_name
                    current_subsubsection_name = None
                if node.macroname == "subsubsection":
                    subsubsection_name = ""
                    for arg_node in node.nodeargd.argnlist:
                        if (
                            isinstance(arg_node, LatexGroupNode)
                            and arg_node.delimiters[0] == "{"
                        ):
                            subsubsection_name = clean_latex_code(
                                arg_node.latex_verbatim()
                            ).strip("{}")
                    # assert subsubsection_name
                    current_subsubsection_name = subsubsection_name
            """
            if isinstance(node, LatexMacroNode) and 'ref' in node.macroname:
                for spec, argn in zip(node.nodeargd.argspec, node.nodeargd.argnlist):
                    if spec=='{':
                        label = ''
                        if len(argn.nodelist)>0:
                            label = argn.nodelist[0].chars
                        prev_context = ''
                        for prev in reversed(recent_nodes):
                            if is_stop_node(prev):
                                break
                            last_paragraph, is_complete = get_last_paragraph(prev)
                            prev_context = last_paragraph + prev_context
                            if is_complete:
                                break
                        ref_info = {'label': label, 'section': current_section_name, 'subsection': current_subsection_name, 'subsubsection': current_subsubsection_name, 'prev_context': prev_context, '_pos': len(next_ref_context)}
                        structured_data['refs'].append(ref_info)
                        prev_ref_contexts.append(ref_info)
            """

            # Check for citation macro "\cite"
            if (
                key2title
                and isinstance(node, LatexMacroNode)
                and "cite" in node.macroname
            ):
                for spec, citation in zip(
                    node.nodeargd.argspec, node.nodeargd.argnlist
                ):
                    if spec == "{":
                        citation_key = ""
                        if len(citation.nodelist) > 0:
                            citation_key = citation.nodelist[0].chars
                        cite_importance = 1 / len(citation_key.split(","))
                        for key in citation_key.split(","):
                            if key2title.get(key.strip()):
                                title_ = key2title[key.strip()]
                                author_ = key2author[key.strip()]
                                prev_context = ""
                                for prev in reversed(recent_nodes):
                                    if is_stop_node(prev):
                                        break
                                    last_sentence, is_complete = get_last_sentence(prev)
                                    prev_context = last_sentence + prev_context
                                    if is_complete:
                                        break


                                # TODO
                                # For short id, it corresponds to the arxiv id
                                # However, arxiv id is often stored in the journal part of paper
                                if structured_data["citations"].get(key.strip()):
                                    citation_data = structured_data["citations"][
                                        key.strip()
                                    ]
                                    citation_context = {
                                        "section": current_section_name,
                                        "subsection": current_subsection_name,
                                        "subsubsection": current_subsubsection_name,
                                        "prev_context": prev_context,
                                        "_pos": len(next_context),
                                    }
                                    prev_citation_contexts.append(citation_context)
                                    citation_data["context"].append(citation_context)
                                    citation_data["importance_score"] += cite_importance
                                else:
                                    paper, score = query_and_match(title_, author_)
                                    # Since the method query_and_match() returns None only, ignore what's inside of the if statement below.
                                    if paper:
                                        citation_context = {
                                            "section": current_section_name,
                                            "subsection": current_subsection_name,
                                            "subsubsection": current_subsubsection_name,
                                            "prev_context": prev_context,
                                            "_pos": len(next_context),
                                        }
                                        prev_citation_contexts.append(citation_context)
                                        # Also loook at the journal part and see if arxiv id is available

                                        structured_data["citations"][key.strip()] = {
                                            "bib_key": key.strip(),
                                            "bib_title": title_,
                                            "bib_author ": author_,
                                            "arxiv_id": paper.entry_id,
                                            "short_id": paper.get_short_id(),
                                            "title": paper.title,
                                            "author": paper.authors[0].name,
                                            "categories": paper.categories,
                                            "published": str(paper.published),
                                            "abstract": paper.summary,
                                            "similar_score": score,
                                            "context": [citation_context],
                                            "importance_score": cite_importance,
                                        }
                                    else:
                                        citation_context = {
                                            "section": current_section_name,
                                            "subsection": current_subsection_name,
                                            "subsubsection": current_subsubsection_name,
                                            "prev_context": prev_context,
                                            "_pos": len(next_context),
                                        }
                                        prev_citation_contexts.append(citation_context)
                                        structured_data["citations"][key.strip()] = {
                                            "bib_key": key.strip(),
                                            "bib_title": title_,
                                            "bib_author ": author_,
                                            "arxiv_id": None,
                                            "short_id": None,
                                            "title": None,
                                            "author": None,
                                            "published": None,
                                            "similar_score": None,
                                            "context": [citation_context],
                                            "importance_score": cite_importance,
                                        }
                                        # print("Result of Citation:")
                                        # print(structured_data["citations"][key.strip()])

            # Recursively check child nodes if the node is an environment (e.g., document or section)
            if hasattr(node, "nodelist"):
                env_stack.append(scope_name)
                (
                    current_section_name,
                    current_subsection_name,
                    current_subsubsection_name,
                    next_ref_context,
                    next_context,
                ) = extract_citations_from_ast(
                    structured_data,
                    flat_data,
                    recent_nodes,
                    node.nodelist,
                    env_stack,
                    (
                        current_section_name,
                        current_subsection_name,
                        current_subsubsection_name,
                    ),
                    key2title,
                    key2author,
                    current_file,
                    appendix,
                    prev_citation_contexts,
                    prev_ref_contexts,
                    next_context,
                    next_ref_context,
                    working_path,
                    flag,
                )
                env_stack.pop()
        return (
            current_section_name,
            current_subsection_name,
            current_subsubsection_name,
            next_ref_context,
            next_context,
        )
    except Exception as e:
        print(e)
        print("Failed to process the node")
        return (
            current_section_name,
            current_subsection_name,
            current_subsubsection_name,
            next_ref_context,
            next_context,
        )


def build_ast(
    latex_code: str,
) -> List[Union[LatexCharsNode, LatexMacroNode, LatexEnvironmentNode]]:
    # Create a custom LaTeX context and add the new macro
    latex_context = get_default_latex_context_db()
    latex_context.add_context_category(
        "custom",
        macros=[
            MacroSpec("addbibliograph", args_parser="{"),
            MacroSpec("addbibresource", args_parser="{"),
            MacroSpec("newcommand", args_parser="{"),
            MacroSpec("caption", args_parser="{"),
        ],
        prepend=True,
    )

    # Create a LatexWalker instance with the custom LaTeX context
    walker = LatexWalker(latex_code, latex_context=latex_context)
    nodes, pos, _ = walker.get_latex_nodes()
    return nodes


class ErrorHandlerBibTexParser(bibtexparser.bparser.BibTexParser):
    def _clean_val(self, val: str) -> str:
        try:
            return super()._clean_val(val)  # Use the original cleaning logic
        except KeyError as e:
            print(f"Warning: Undefined string '{e}'. Ignoring.")
            return ""  # Or a default value


def load_bib_info(
    path: str, key2title: Dict[str, Any], key2author: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Load the BibTeX file
    with open(path, "r") as bib_file:
        parser = ErrorHandlerBibTexParser()
        bib_database = bibtexparser.load(bib_file, parser=parser)

    # Iterate through all entries
    for entry in bib_database.entries:
        # print(f"entry: {entry}")
        if entry.get("ID"):
            id = entry.get("ID").strip()
            key2title[id] = entry.get("title")
            author = entry.get("author")
            if author:
                key2author[id] = author.split("and")[0].strip()
            else:
                key2author[id] = ""
    return key2title, key2author


# We can use this to extract arxiv id written in the journal part of citations
def load_bib_key_to_arxiv_id(
    path: str, key2id: Dict[str, Any]
) -> Dict[str, Any]:
    
    pattern = re.compile(
        r'arxiv:(?P<id>(?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z\-]+/\d{7}(?:v\d+)?))',
        re.IGNORECASE
    )
    # print(f"pattern: {pattern}")

    with open(path, "r") as bib_file:
        parser = ErrorHandlerBibTexParser()
        bib_database = bibtexparser.load(bib_file, parser=parser)
    for entry in bib_database.entries:
        # print(f"entry: {entry}")
        if entry.get("ID"):
            bib_key = entry.get("ID", "").strip()
            journal_field = entry.get("journal", "")
            # print(f"journal: {journal}")
            if not journal_field:
                continue
            journal_text = journal_field.strip().lower()
            match = pattern.search(journal_text)
            if match:
                key2id[bib_key] = match.group('id')
            else:
                # Optional: try other fields or at least log it
                # e.g. check entry.get("eprint"), entry.get("archiveprefix")...
                # print(f"[!] No arXiv ID found in journal for bib key {bib_key!r}: {journal_text!r}")
                pass
    return key2id


def parse_MathNode(node: LatexMathNode) -> str:
    try:
        return node.latex_verbatim()
    except Exception as e:
        print(e)
        return None


def parse_algorithmEnv(node: LatexEnvironmentNode) -> str:
    try:
        return node.latex_verbatim()
    except Exception as e:
        print(e)
        return None


def extract_table_info(
    nodes: Union[LatexCharsNode, LatexMacroNode, LatexEnvironmentNode],
    res: Dict[str, Any],
):
    for node in nodes:
        try:
            if isinstance(node, LatexMacroNode):
                if node.macroname == "caption":
                    res["caption"] = node.latex_verbatim()
                if node.macroname == "label":
                    res["label"] = node.latex_verbatim()
            if isinstance(node, LatexEnvironmentNode):
                if "tabular" in node.environmentname:
                    res["tabular"] = node.latex_verbatim()
                    continue
                if "subtable" in node.environmentname:
                    subtable = {
                        "caption": "",
                        "label": None,
                        "tabular": "",
                        "subtables": [],
                    }
                    extract_table_info(node.nodelist, subtable)
                    res["subtables"].append(subtable)
        except Exception as e:
            print(e)
        if hasattr(node, "nodelist"):
            extract_table_info(node.nodelist, res)


def parse_tableEnv(node: LatexEnvironmentNode) -> str:
    try:
        res = {
            "original": node.latex_verbatim(),
            "caption": "",
            "label": None,
            "tabular": "",
            "subtables": [],
        }
        extract_table_info(node.nodelist, res)
        return res
    except Exception as e:
        print(e)
        return None


def extract_figure_info(
    nodes: Union[LatexCharsNode, LatexMacroNode, LatexEnvironmentNode],
    res: Dict[str, Any],
):
    for node in nodes:
        try:
            if isinstance(node, LatexMacroNode):
                # print(f"node: {node}")

                if node.macroname == "caption":
                    res["caption"] = node.latex_verbatim()
                if node.macroname == "label":
                    # It seems that here, there may be multiple labels in nodes, but it only takes the last one.

                    res["label"] = node.latex_verbatim()
                if node.macroname == "includegraphics":
                    # We can use this part to extract label-path pairs
                    # print(f"node of figure: {node}")
                    # print(f"node of figure verbatim: {node.latex_verbatim()}")
                    for argd, arg_node in zip(
                        node.nodeargd.argspec, node.nodeargd.argnlist
                    ):
                        # print(f"argd: {argd}")
                        # print(f"arg_node: {arg_node}")
                        if argd == "{":
                            res["figure_paths"].append(
                                arg_node.latex_verbatim().strip("{}")
                            )
            if isinstance(node, LatexEnvironmentNode):
                if "figure" in node.environmentname:
                    subfigure = {
                        "caption": "",
                        "label": None,
                        "figure_paths": [],
                        "subfigures": [],
                    }
                    extract_figure_info(node.nodelist, subfigure)
                    res["subfigures"].append(subfigure)
        except Exception as e:
            print(e)
        if hasattr(node, "nodelist"):
            extract_figure_info(node.nodelist, res)

def is_implicit_subfigure(node):
    has_graphic = False
    has_caption_or_label = False

    def walk(n):
        nonlocal has_graphic, has_caption_or_label
        if isinstance(n, LatexMacroNode):
            if n.macroname == "includegraphics":
                has_graphic = True
            if n.macroname in ("caption", "label"):
                has_caption_or_label = True
        if hasattr(n, "nodelist") and n.nodelist:
            for child in n.nodelist:
                walk(child)
    
    walk(node)
    return has_graphic and has_caption_or_label


def extract_figure_info_new(
    nodes: Union[list, LatexCharsNode, LatexMacroNode, LatexEnvironmentNode],
    res: Dict[str, Any],
):
    """
    Extract subfigures inside of the figure
    """
    if not nodes:
        return

    # Normalize to iterable
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]

    for node in nodes:
        try:
            # If this node itself is a subfigure-like container, create a fresh context
            if isinstance(node, LatexEnvironmentNode) and node.environmentname in _SUBFIGURE_ENVS:
                # New subfigure dict
                subfigure = {
                    "caption": "",
                    "label": None,
                    "figure_paths": [],
                    "subfigures": [],
                }
                # Recurse into its content
                extract_figure_info(node.nodelist, subfigure)
                # Append to current result's subfigures
                res.setdefault("subfigures", []).append(subfigure)
                # Also continue into its children in case of nested structures
                continue

            # explicit subfloat-style macro
            if isinstance(node, LatexMacroNode) and node.macroname in _SUBFIGURE_MACROS:
                subfigure = {"caption": "", "label": None, "figure_paths": [], "subfigures": []}
                # recurse into its arguments (often contains the inner content)
                for arg in node.nodeargd.argnlist:
                    extract_figure_info(arg, subfigure)
                res.setdefault("subfigures", []).append(subfigure)
                continue

            # implicit grouping via heuristic
            if (isinstance(node, LatexEnvironmentNode) or isinstance(node, LatexMacroNode)) and is_implicit_subfigure(node):
                subfigure = {"caption": "", "label": None, "figure_paths": [], "subfigures": []}
                extract_figure_info(getattr(node, "nodelist", []) or [], subfigure)
                # also dive into macro args if any
                if isinstance(node, LatexMacroNode):
                    for arg in node.nodeargd.argnlist:
                        extract_figure_info(arg, subfigure)
                res.setdefault("subfigures", []).append(subfigure)
                continue

            if isinstance(node, LatexMacroNode):
                if node.macroname == "caption":
                    # If multiple captions appear, you can decide to keep first/last or concatenate
                    # Here we overwrite so last wins; change if needed.
                    res["caption"] = node.latex_verbatim()
                elif node.macroname == "label":
                    res["label"] = node.latex_verbatim()
                elif node.macroname == "includegraphics":
                    for argd, arg_node in zip(
                        node.nodeargd.argspec, node.nodeargd.argnlist
                    ):
                        if argd == "{":
                            path = arg_node.latex_verbatim().strip("{}")
                            res.setdefault("figure_paths", []).append(path)

            if hasattr(node, "nodelist"):
                # Recurse into children for everything else
                extract_figure_info(node.nodelist, res)
        except Exception as e:
            # You might want to replace prints with logging.
            print(f"error processing node {node}: {e}")


def parse_figureEnv(node: LatexEnvironmentNode) -> Dict[str, Any]:
    try:
        res = {
            "original": node.latex_verbatim(),
            "caption": "",
            "label": None,
            "subfigures": [],
            "figure_paths": [],
        }
        res2 = res
        # print(f"res before parsing: {res}")
        # extract_figure_info(node.nodelist, res)
        # print(f"parsed figure info: {res}")
        extract_figure_info_new(node.nodelist, res2)
        # print(f"parsed figure info with consideration of subfigures: {res2}")
        return res2
    except Exception as e:
        print(e)
        return None


def parse_equationEnv(node: LatexEnvironmentNode) -> str:
    try:
        return node.latex_verbatim()
    except Exception as e:
        print(e)
        return None


def parse_itemizeEnv(node: LatexEnvironmentNode) -> str:
    try:
        return node.latex_verbatim()
    except Exception as e:
        print(e)
        return None


def parse_unknownEnv(node: LatexEnvironmentNode) -> str:
    try:
        return node.latex_verbatim()
    except Exception as e:
        print(e)
        return None


def safe_extract_chars(
    node: Union[LatexCharsNode, LatexMacroNode, LatexEnvironmentNode, LatexMathNode]
) -> str:
    if isinstance(node, LatexCharsNode):
        return node.chars
    elif isinstance(node, LatexGroupNode):
        res = [safe_extract_chars(node_) for node_ in node.nodelist]
        return "".join(res)
    return ""


def get_bib_names(
    nodes: List[Union[LatexCharsNode, LatexMacroNode, LatexEnvironmentNode]]
) -> Tuple[List[str], List[LatexEnvironmentNode]]:
    bib_names = []
    bbl_node = []
    for node in nodes:
        if isinstance(node, LatexMacroNode):
            if node.macroname == "bibliography":
                bib_name = safe_extract_chars(node.nodeargd.argnlist[0])
                bib_names.extend(bib_name.split(","))
            if node.macroname == "addbibresource":
                bib_name = safe_extract_chars(node.nodeargd.argnlist[0])
                bib_names.extend(bib_name.split(","))
        if isinstance(node, LatexEnvironmentNode):
            if node.environmentname == "thebibliography":
                bbl_node.append(node)
        if hasattr(node, "nodelist"):
            names, nodes = get_bib_names(node.nodelist)
            bib_names.extend(names)
            bbl_node.extend(nodes)
    return bib_names, bbl_node


def get_base_info(
    ast: List[Union[LatexCharsNode, LatexMacroNode, LatexEnvironmentNode]],
    res: Dict[str, Any],
):
    for node in ast:
        if isinstance(node, LatexMacroNode):
            if node.macroname == "title":
                res["title"] = safe_extract_chars(node.nodeargd.argnlist[0])
            if node.macroname == "author":
                res["author"] = safe_extract_chars(node.nodeargd.argnlist[0])
        if isinstance(node, LatexEnvironmentNode):
            if node.environmentname == "document":
                res["doc_node"] = node
        if hasattr(node, "nodelist"):
            get_base_info(node.nodelist, res)


def load_bbl_info(
    bbl_node: LatexEnvironmentNode,
    key2title: Dict[str, str],
    key2author: Dict[str, str],
):
    bbl_text = bbl_node.latex_verbatim()
    bbl_list = re.sub(r"\[.*\]", "", bbl_text).split("\\bibitem")
    for bbl_item in bbl_list[1:]:
        key = re.search(r".*?{(.*?)}", bbl_item).group(1)
        bbl_item = bbl_item.replace(r"{" + key + "}", "").strip("\n")
        divider_left = ["{", "\\newblock", "'"]
        divider_right = ["}", "\\newblock", "'"]
        pos = len(bbl_item)
        divider = -1
        for i in range(len(divider_left)):
            divider_ = divider_left[i]
            pos_ = bbl_item.find(divider_)
            if pos_ >= 0 and pos_ < pos:
                pos = pos_
                divider = i
        if divider >= 0:
            authors = bbl_item[:pos]
            title = bbl_item[pos + len(divider_left[divider]) :]
            pos_ = title.find(divider_right[divider])
            title = title[:pos_]
            authors = clean_latex_format(" ".join(authors.split("\n")))
            title = clean_latex_format(" ".join(title.split("\n")))
            key2title[key] = title
            key2author[key] = authors

def extract_arxiv_id():
    """
    Extract arxiv id from latex by recognizing possible patterns
    Example: journal={arXiv preprint arXiv:(arxiv id)}
    We need to map the arxiv id to the bib_key
    Possibly store it somewhere
    """


    pass