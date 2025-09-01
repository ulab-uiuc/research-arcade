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

"""
For LLM task, the pipeline is:
1. Map figures into texts
2. Go through all the paragraphs. For each paragraph, provide it with the related paragraphs
3. Prompt the LLM to generate the index of the paragraph that the figure should be inserted
"""

"""
For GNN, the pipeline is:
1. First embed the nodes, including texts, images, tables and cited papers (title and abstract if provided) into embedding space.
2. Given the embeddings, we perform aggregation
3. After several iterations, we obtain the final embeddings. We use MLP to generate a similarity score, pass in a MLP? and ultimately obtain the highest value.
"""

"""
For RAG, the pipeline is:
1. Embed each paragraph along its images, texts, tables and citations into the embedding space.
2. After that, we generate the embeddings of each paragraph.
3. Perform RAG similarity score searching, take the top 1 paragraph
"""


"""
Task: Given a figure or a table, predict which paragraph it should be inserted into.

This file provides a clean, extensible SKELETON with three interchangeable approaches:
  1) LLM-based
  2) GNN-based
  3) RAG-based

Notes
-----
- Real model methods are expected to live in separate files (e.g., models/llm_backend.py,
  models/gnn_backend.py, models/rag_backend.py). Here we just define the interfaces and
  callsites. You can plug in your own implementations later.
- Prior to inference, we fetch and normalize data from your PaperGraph (DB/JSON/etc.).
- Output is either a paragraph *index* (0..N-1 within a paper) or a paragraph *ID*.

Directory Suggestion
--------------------
project_root/
  tasks/
    figure_table_insertion_skeleton.py  <-- THIS FILE
  models/
    llm_backend.py
    gnn_backend.py
    rag_backend.py
  data/
    ...      # your sources / configs

Replace all `pass` and `raise NotImplementedError` with your logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable, Literal, Protocol, Any
import enum
import math
import logging
import argparse

# Optional: if you want to use numpy for RAG similarity
try:
    import numpy as np
except Exception:  # keep skeleton import-light
    np = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core data structures
# -----------------------------------------------------------------------------

class ItemType(str, enum.Enum):
    FIGURE = "figure"
    TABLE = "table"

@dataclass
class Paragraph:
    id: str
    text: str
    section: Optional[str] = None
    index: Optional[int] = None  # position within the paper (0..N-1)

@dataclass
class Figure:
    id: str
    label: Optional[str] = None  # e.g., "Figure 2"
    caption: Optional[str] = None
    image_path: Optional[str] = None  # absolute/relative path (if available)
    # You can add precomputed embeddings or tags here later

@dataclass
class Table:
    id: str
    label: Optional[str] = None  # e.g., "Table 1"
    caption: Optional[str] = None
    data_path: Optional[str] = None  # e.g., CSV/JSON extracted table

@dataclass
class Citation:
    id: str
    title: Optional[str] = None
    abstract: Optional[str] = None

@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: Optional[str]
    paragraphs: List[Paragraph]
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)

# -----------------------------------------------------------------------------
# Fetchers / Mappers (DB → In-memory)
# -----------------------------------------------------------------------------

class DataFetcher(Protocol):
    """Protocol for fetching data for a single paper.

    Implement this using your DB schema (e.g., PostgreSQL tables `sections`, `paragraphs`,
    `figures`, `tables`, `citations`, etc.).
    """
    def fetch_paper(self, paper_arxiv_id: str) -> Paper:
        ...

class DummyDataFetcher:
    """Minimal stub that illustrates the shape; replace with your DB-backed fetcher."""
    def fetch_paper(self, paper_arxiv_id: str) -> Paper:
        # TODO: connect to your DB and populate all fields
        example_paragraphs = [
            Paragraph(id="p1", text="Intro paragraph about the problem.", section="Introduction", index=0),
            Paragraph(id="p2", text="Method paragraph describing approach.", section="Methods", index=1),
            Paragraph(id="p3", text="Results paragraph with key findings.", section="Results", index=2),
            Paragraph(id="p4", text="Discussion paragraph.", section="Discussion", index=3),
        ]
        example_figures = [Figure(id="f1", label="Figure 1", caption="Model architecture.")]
        example_tables = [Table(id="t1", label="Table 1", caption="Hyperparameters.")]
        return Paper(
            arxiv_id=paper_arxiv_id,
            title="Example Title",
            abstract="This paper explores...",
            paragraphs=example_paragraphs,
            figures=example_figures,
            tables=example_tables,
            citations=[],
        )

# -----------------------------------------------------------------------------
# Interfaces to Model Backends (kept thin; real logic in models/*.py)
# -----------------------------------------------------------------------------

class LLMBackend(Protocol):
    def choose_paragraph_index(self, prompts: List[str]) -> int:
        """Return the index (0..len(prompts)-1) chosen by the LLM.
        Typically you build one prompt per candidate paragraph and ask
        the model to select the best, OR use a single prompt with options.
        """
        ...

class GNNBackend(Protocol):
    def predict_paragraph_index(self, graph: "HeteroGraph", target_node: str) -> int:
        """Given a heterogeneous graph and a target node (figure/table),
        return the predicted paragraph index.
        """
        ...

class RAGBackend(Protocol):
    def embed_text(self, text: str) -> List[float]:
        ...
    def embed_image(self, image_path: Optional[str], caption: Optional[str]) -> List[float]:
        ...
    def similarity(self, a: List[float], b: List[float]) -> float:
        ...

# Optional lightweight default RAG backend (cosine over numpy)
class SimpleRAGBackend:
    def __init__(self):
        if np is None:
            raise ImportError("numpy is required for SimpleRAGBackend")

    def _norm(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-12
        return v / n

    def embed_text(self, text: str) -> List[float]:
        # TODO: replace with real encoder (e.g., sentence-transformers)
        rng = abs(hash(text)) % (10**6)
        vec = np.random.default_rng(rng).random(768)
        return self._norm(vec).tolist()

    def embed_image(self, image_path: Optional[str], caption: Optional[str]) -> List[float]:
        seed_src = (image_path or "") + (caption or "")
        rng = abs(hash(seed_src)) % (10**6)
        vec = np.random.default_rng(rng).random(768)
        return self._norm(vec).tolist()

    def similarity(self, a: List[float], b: List[float]) -> float:
        va = np.asarray(a)
        vb = np.asarray(b)
        return float((va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))

# -----------------------------------------------------------------------------
# Utilities: context selection, prompt building, simple graph container
# -----------------------------------------------------------------------------

@dataclass
class RelatedContext:
    window_before: int = 1
    window_after: int = 1

    def collect(self, paragraphs: List[Paragraph], idx: int) -> List[Paragraph]:
        left = max(0, idx - self.window_before)
        right = min(len(paragraphs), idx + self.window_after + 1)
        return paragraphs[left:right]

class FigTableTextMapper:
    """Maps figures/tables to textual surrogates (captions, tags, OCR, etc.)."""
    def to_text(self, item: Figure | Table) -> str:
        parts = []
        if isinstance(item, Figure):
            parts.append(item.label or "Figure")
            if item.caption:
                parts.append(item.caption)
        else:
            parts.append(item.label or "Table")
            if item.caption:
                parts.append(item.caption)
        return ". ".join(p for p in parts if p)

class PromptBuilder:
    """LLM prompt builder per candidate paragraph."""
    def build(self, paper: Paper, item_text: str, candidate_idx: int, ctx: List[Paragraph]) -> str:
        context_block = "\n\n".join(f"[{p.index}] {p.text}" for p in ctx)
        candidate = paper.paragraphs[candidate_idx]
        return (
            f"You are given a figure/table description and must decide whether it belongs in the given candidate paragraph.\n"
            f"Paper title: {paper.title}\n"
            f"Abstract: {paper.abstract}\n\n"
            f"Item description: {item_text}\n\n"
            f"Candidate paragraph index: {candidate.index}\n"
            f"Candidate paragraph text: {candidate.text}\n\n"
            f"Related context (nearby paragraphs):\n{context_block}\n\n"
            f"Answer YES if the figure/table should be inserted here, otherwise NO."
        )

# Minimal hetero-graph container (replace with DGL/PyG/etc. as needed)
@dataclass
class HeteroGraph:
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # node_id -> {type, feat, ...}
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  # (src, rel, dst)

    def add_node(self, node_id: str, ntype: str, **attrs: Any) -> None:
        self.nodes[node_id] = {"type": ntype, **attrs}

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        self.edges.append((src, rel, dst))

# -----------------------------------------------------------------------------
# Pipelines (orchestrators) for each approach
# -----------------------------------------------------------------------------

@dataclass
class LLMPipeline:
    llm: LLMBackend
    mapper: FigTableTextMapper = field(default_factory=FigTableTextMapper)
    prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)
    related: RelatedContext = field(default_factory=lambda: RelatedContext(1, 1))

    def predict(self, paper: Paper, item: Figure | Table) -> int:
        item_text = self.mapper.to_text(item)
        prompts: List[str] = []
        for i, _ in enumerate(paper.paragraphs):
            ctx = self.related.collect(paper.paragraphs, i)
            prompts.append(self.prompt_builder.build(paper, item_text, i, ctx))
        logger.debug("Built %d prompts for LLM selection", len(prompts))
        chosen_idx = self.llm.choose_paragraph_index(prompts)
        return chosen_idx

@dataclass
class GNNPipeline:
    gnn: GNNBackend

    def build_graph(self, paper: Paper) -> HeteroGraph:
        g = HeteroGraph()
        # Add paragraph nodes
        for p in paper.paragraphs:
            g.add_node(node_id=f"para::{p.id}", ntype="paragraph", text=p.text, index=p.index)
        # Add figure nodes
        for f in paper.figures:
            g.add_node(node_id=f"fig::{f.id}", ntype="figure", caption=f.caption)
        # Add table nodes
        for t in paper.tables:
            g.add_node(node_id=f"tab::{t.id}", ntype="table", caption=t.caption)
        # Add citation nodes (optional)
        for c in paper.citations:
            g.add_node(node_id=f"cit::{c.id}", ntype="citation", title=c.title, abstract=c.abstract)

        # Example edges (customize using your PaperGraph):
        # - adjacency between neighboring paragraphs
        for i in range(len(paper.paragraphs) - 1):
            a = paper.paragraphs[i]
            b = paper.paragraphs[i + 1]
            g.add_edge(f"para::{a.id}", "NEXT", f"para::{b.id}")
            g.add_edge(f"para::{b.id}", "PREV", f"para::{a.id}")

        # - section edges (if available)
        by_section: Dict[str, List[Paragraph]] = {}
        for p in paper.paragraphs:
            if p.section:
                by_section.setdefault(p.section, []).append(p)
        for sec, plist in by_section.items():
            for p in plist:
                g.add_edge(f"para::{p.id}", f"IN_SECTION::{sec}", f"para::{p.id}")

        # - (optional) cite edges from paragraphs to citations
        # - (optional) textual/visual linkage edges (e.g., mentions of Fig. X in text)
        # Fill with your heuristics or DB relations
        return g

    def predict(self, paper: Paper, item: Figure | Table) -> int:
        graph = self.build_graph(paper)
        target_node = ("fig::" + item.id) if isinstance(item, Figure) else ("tab::" + item.id)
        pred_idx = self.gnn.predict_paragraph_index(graph, target_node)
        return pred_idx

@dataclass
class RAGPipeline:
    rag: RAGBackend

    def predict(self, paper: Paper, item: Figure | Table) -> int:
        # 1) Embed each paragraph (text + optional extras if you have them)
        para_vecs: List[List[float]] = []
        for p in paper.paragraphs:
            vec = self.rag.embed_text(p.text)
            para_vecs.append(vec)

        # 2) Embed the figure/table (image+caption or table+caption)
        if isinstance(item, Figure):
            item_vec = self.rag.embed_image(item.image_path, item.caption)
        else:
            # For tables, many systems treat caption as the primary textual surrogate
            item_vec = self.rag.embed_text(item.caption or "")

        # 3) Score similarity and pick top-1
        best_idx = 0
        best_score = -math.inf
        for i, v in enumerate(para_vecs):
            score = self.rag.similarity(item_vec, v)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

@dataclass
class LabeledItem:
    paper_arxiv_id: str
    item_type: ItemType
    item_id: str
    gold_paragraph_index: int

@dataclass
class Metrics:
    top1_acc: float
    mrr: float

class Evaluator:
    def __init__(self, pipeline: "UnionPipeline"):
        self.pipeline = pipeline

    def evaluate(self, labeled_items: List[LabeledItem], fetcher: DataFetcher) -> Metrics:
        ranks: List[int] = []
        correct = 0
        for li in labeled_items:
            paper = fetcher.fetch_paper(li.paper_arxiv_id)
            item = _find_item(paper, li.item_type, li.item_id)
            pred_idx = self.pipeline.predict(paper, item)
            if pred_idx == li.gold_paragraph_index:
                correct += 1
            # For the skeleton, we only produce a top-1; rank is 1 if correct else large
            ranks.append(1 if pred_idx == li.gold_paragraph_index else 1000)
        top1 = correct / max(1, len(labeled_items))
        mrr = sum(1.0 / r for r in ranks) / max(1, len(ranks))
        return Metrics(top1_acc=top1, mrr=mrr)

# -----------------------------------------------------------------------------
# Simple Union wrapper to swap approaches from CLI
# -----------------------------------------------------------------------------

class UnionPipeline:
    def __init__(self, approach: Literal["llm", "gnn", "rag"], **kwargs: Any):
        self.approach = approach
        if approach == "llm":
            llm_backend: LLMBackend = kwargs.get("llm_backend") or _load_llm_backend()
            self.inner = LLMPipeline(llm=llm_backend)
        elif approach == "gnn":
            gnn_backend: GNNBackend = kwargs.get("gnn_backend") or _load_gnn_backend()
            self.inner = GNNPipeline(gnn=gnn_backend)
        elif approach == "rag":
            rag_backend: RAGBackend = kwargs.get("rag_backend") or SimpleRAGBackend()
            self.inner = RAGPipeline(rag=rag_backend)
        else:
            raise ValueError(f"Unknown approach: {approach}")

    def predict(self, paper: Paper, item: Figure | Table) -> int:
        return self.inner.predict(paper, item)

# -----------------------------------------------------------------------------
# Backend loaders (stubs)
# -----------------------------------------------------------------------------

def _load_llm_backend() -> LLMBackend:
    """Dynamically import your LLM backend. Replace with your path/module."""
    try:
        from models.llm_backend import MyLLMBackend  # type: ignore
        return MyLLMBackend()
    except Exception as e:
        raise NotImplementedError("Provide models/llm_backend.py with MyLLMBackend") from e

def _load_gnn_backend() -> GNNBackend:
    try:
        from models.gnn_backend import MyGNNBackend  # type: ignore
        return MyGNNBackend()
    except Exception as e:
        raise NotImplementedError("Provide models/gnn_backend.py with MyGNNBackend") from e

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _find_item(paper: Paper, item_type: ItemType, item_id: str) -> Figure | Table:
    if item_type == ItemType.FIGURE:
        for f in paper.figures:
            if f.id == item_id:
                return f
        raise KeyError(f"Figure not found: {item_id}")
    else:
        for t in paper.tables:
            if t.id == item_id:
                return t
        raise KeyError(f"Table not found: {item_id}")

# -----------------------------------------------------------------------------
# CLI Demo (runs a single prediction with dummy fetcher)
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Figure/Table → Paragraph insertion predictor (skeleton)")
    parser.add_argument("paper", type=str, help="arXiv id of the paper")
    parser.add_argument("item_type", choices=["figure", "table"], help="Item type")
    parser.add_argument("item_id", type=str, help="Item id in that paper")
    parser.add_argument("--approach", choices=["llm", "gnn", "rag"], default="rag", help="Which pipeline to use")

    args = parser.parse_args()

    fetcher: DataFetcher = DummyDataFetcher()  # replace with your DB-backed fetcher
    paper = fetcher.fetch_paper(args.paper)
    item = _find_item(paper, ItemType(args.item_type), args.item_id)

    pipeline = UnionPipeline(approach=args.approach)
    pred_idx = pipeline.predict(paper, item)

    logger.info("Predicted insertion paragraph index: %d", pred_idx)

# -----------------------------------------------------------------------------
# LLM-specific prompt orchestration pattern (reference)
# -----------------------------------------------------------------------------

class YesNoVotingLLMBackend:
    """Example LLM backend API: you would implement the actual LLM calls elsewhere.

    Strategy (one of many):
      - Build one prompt per candidate paragraph asking YES/NO.
      - Query the model; convert response to {YES:1, NO:0} score.
      - Pick the index with highest YES score (ties broken by confidence or heuristics).
    """
    def choose_paragraph_index(self, prompts: List[str]) -> int:
        # TODO: Replace with batched LLM calls
        # For now: naive random pick for skeleton
        import random
        return random.randrange(len(prompts))

# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()


