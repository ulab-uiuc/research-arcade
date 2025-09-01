"""
Given a figure or a table, predict which paragraph it is inserted.

Three approaches:
1. LLM-based method
    1.1. oken-based graph-llm on raw document
    1.2. oken-based graph-llm on our dataset with graph structure
    1.3. embedding-based graph-llm on our dataset with graph structure
2. GNN based method: Graph-based (GNN): Paper Graph → Embedding → Aggregation
    2.1. Simple GNN
    2.2. Heterogeneous GNN, which includes 
3. RAG based method: paragraph embedding → top-k aggregations
    3.1. RAG
Prior to all, we need to fetch the data needed
The real model methods are stored in another python file
"""

def _data_extraction_non_vlm(jsonl_file_path, use_figure=True, use_table=True, model_name="openai/clip-vit-base-patch32"):
    outputs: List[Dict[str, Any]] = []
    download_path = "./download"

    with open(jsonl_file_path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[WARN] Skipping line {line_no}: invalid JSON ({e})")
                continue

            paragraph_key_id = rec.get("paragraph_key_id")
            paper_arxiv_id   = rec.get("paper_arxiv_id", "")
            paper_section    = rec.get("paper_section", "")
            paragraph_id_loc = rec.get("paragraph_id_local")
            original_content = rec.get("original_content", "") or ""
            num_char         = int(rec.get("num_char") or len(original_content))
            title            = rec.get("title", "") or ""
            abstract         = rec.get("abstract", "") or ""
            prev_paras       = rec.get("prev_paras", []) or []
            next_paras       = rec.get("next_paras", []) or []
            figures          = rec.get("figures", []) or []
            tables           = rec.get("tables", []) or []
            fig_labels       = rec.get("fig_labels", []) or []
            table_labels     = rec.get("table_labels", []) or []
            bib_keys         = rec.get("bib_keys", []) or []
            cited_triples    = rec.get("cited_triples", []) or []

            abstracts_norm = []
            for t in cited_triples:
                if isinstance(t, dict):
                    abstracts_norm.append((
                        t.get("0") or t.get("bib_key") or "",
                        t.get("1") or t.get("title")   or "",
                        t.get("2") or t.get("abstract") or "",
                    ))
                elif isinstance(t, (list, tuple)):
                    bk  = t[0] if len(t) > 0 else ""
                    ttl = t[1] if len(t) > 1 else ""
                    abs_ = t[2] if len(t) > 2 else ""
                    abstracts_norm.append((bk or "", ttl or "", abs_ or ""))
                else:
                    continue

            k_prev = len(prev_paras)
            k_next = len(next_paras)
            k = max(k_prev, k_next) if (k_prev or k_next) else 2

            prompt = _build_prompt_contract(
                k=k,
                figures=figures,
                tables=tables,
                abstracts=abstracts_norm,
                prev_paras=prev_paras,
                next_paras=next_paras,
                paper_section=paper_section,
                num_char=num_char,
                title=title,
                abstract=abstract,
                fig_labels=fig_labels,
                table_labels=table_labels,
                bib_keys=bib_keys,
                use_figure=use_figure,
                use_table=use_table,
            )

            figure_paths = []
            for f in figures:
                latex_path = (f.get("path") or "").strip()
                if not latex_path:
                    continue
                try:
                    abs_path = figure_latex_path_to_path(
                        path=download_path,
                        arxiv_id=paper_arxiv_id,
                        latex_path=latex_path,
                    )
                    figure_paths.append(abs_path)
                except Exception as e:
                    print(f"[WARN] Failed to resolve figure path '{latex_path}' "
                          f"for {paper_arxiv_id}: {e}")

            # === ORIGINAL: infer tags via CLIP; ADDED: also keep soft-token payloads ===
            image_tags_projections = visual_adaptation(image_paths=figure_paths)

            image_tag_list: List[List[str]] = []
            image_projection_payloads: List[Dict[str, Any]] = []   # ADDED

            for tag, projection in image_tags_projections:
                image_tag_list.append(tag)
                # ADDED: when a projection exists (shape [1,4096]), create [T,4096] soft tokens and payload
                if projection is not None:
                    try:
                        soft = _repeat_to_soft_tokens(projection, T=SOFT_TOKENS)   # [T,4096] float32
                        image_projection_payloads.append(_to_prompt_embeds_payload(soft))
                    except Exception as e:
                        print(f"[WARN] Failed to build soft tokens: {e}")

            print(f"Image tag list: {image_tag_list}")

            outputs.append({
                "paragraph_key_id": paragraph_key_id,
                "paper_arxiv_id": paper_arxiv_id,
                "prompt": prompt,
                "original_content": original_content,
                "image_tag_list": image_tag_list,
                # === ADDED: provide to llm_generate when use_image_projection=True ===
                "image_projections": image_projection_payloads,
                "fig_labels": fig_labels,
                "table_labels": table_labels,
                "bib_keys": bib_keys,
                "k": k,
                "meta": {
                    "paper_section": paper_section,
                    "paragraph_id_local": paragraph_id_loc,
                    "title": title,
                    "abstract": abstract,
                    "num_char": num_char,
                    "prev_paras": prev_paras,
                    "next_paras": next_paras,
                    "figures": figures,
                    "tables": tables,
                    "cited_triples": abstracts_norm,
                    "figure_paths": figure_paths,
                    # ADDED: debugging
                    "soft_tokens_T": SOFT_TOKENS,
                }
            })

    return outputs

