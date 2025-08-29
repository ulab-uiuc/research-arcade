import os, json, csv, time, hashlib, mmap
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import torch
import numpy as np
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

try:
    import faiss  # optional but recommended for large tag banks
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# -----------------------
# Utilities
# -----------------------

def _read_tags(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    tags = []
    if ext in [".txt"]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    tags.append(t)
    elif ext in [".csv"]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # assume first column is the tag
                t = row[0].strip()
                if t:
                    tags.append(t)
    elif ext in [".jsonl", ".ndjson"]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # try common keys
                t = obj.get("tag") or obj.get("label") or obj.get("text")
                if t and isinstance(t, str) and t.strip():
                    tags.append(t.strip())
    else:
        raise ValueError(f"Unsupported tag file extension: {ext}")
    # de-dup, preserve order
    seen = set()
    uniq = []
    for t in tags:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def _hash_tags(tags: List[str]) -> str:
    h = hashlib.sha256()
    for t in tags:
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    # x: [N, D]
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


# -----------------------
# Builder
# -----------------------

def build_tagbank(
    tag_source: str,
    model_id: str = "openai/clip-vit-base-patch32",
    out_dir: str = "./tagbank",
    device: str = "cuda",
    batch_size: int = 512,
    write_faiss: bool = True,
    fp16: bool = True,
) -> None:
    """
    Read tags, compute CLIP text embeddings (normalized), and save to disk.
    Artifacts:
      - tags.json
      - meta.json
      - text_embeds.npy (float16 or float32, L2-normalized)
      - faiss.index (optional)
    """
    _ensure_dir(out_dir)
    tags = _read_tags(tag_source)
    if not tags:
        raise ValueError("No tags found.")

    print(f"[TagBank] Loaded {len(tags)} tags from {tag_source}")

    # Load CLIP
    print(f"[TagBank] Loading CLIP model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    # Compute text embeddings in batches
    embs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(tags), batch_size):
            batch = tags[i : i + batch_size]
            inputs = processor.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            # get_text_features does the proper pooling for CLIP text tower
            feats = model.get_text_features(**inputs)  # [B, D]
            feats = torch.nn.functional.normalize(feats, dim=-1)
            if fp16:
                feats = feats.half()
            embs.append(feats.detach().cpu())
            if (i // batch_size) % 10 == 0:
                print(f"[TagBank] {i}/{len(tags)} done...")
    embs = torch.cat(embs, dim=0).numpy()  # [N, D], normalized
    print(f"[TagBank] Encoded {len(tags)} tags in {time.time()-t0:.1f}s")

    # Save files
    tags_path = os.path.join(out_dir, "tags.json")
    meta_path = os.path.join(out_dir, "meta.json")
    vec_path  = os.path.join(out_dir, "text_embeds.npy")
    with open(tags_path, "w", encoding="utf-8") as f:
        json.dump({"tags": tags}, f, ensure_ascii=False, indent=2)

    # (They’re already normalized; store as float16 for compactness if chosen)
    np.save(vec_path, embs)

    meta = {
        "model_id": model_id,
        "dtype": "float16" if fp16 else "float32",
        "num_tags": len(tags),
        "dim": int(embs.shape[1]),
        "tags_sha": _hash_tags(tags),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "faiss": bool(write_faiss and _HAS_FAISS),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if write_faiss:
        if not _HAS_FAISS:
            print("[TagBank] faiss not installed; skipping index build.")
        else:
            # cosine sim == inner product if vectors are L2-normalized
            # Use IndexFlatIP for simplicity; switch to IVF/PQ for huge banks.
            print("[TagBank] Building FAISS index (IP)...")
            xb = embs.astype(np.float32)  # faiss prefers float32
            index = faiss.IndexFlatIP(xb.shape[1])
            index.add(xb)
            faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
            print("[TagBank] Saved FAISS index.")

    print(f"[TagBank] Done. Saved to {out_dir}")


# -----------------------
# Loader / Query
# -----------------------

@dataclass
class _TagBankData:
    tags: List[str]
    vecs: np.ndarray  # normalized, [N, D], float16/float32
    dim: int
    meta: Dict
    faiss_index: Optional["faiss.Index"] = None


class TagBank:
    def __init__(self, data: _TagBankData):
        self._data = data

    @staticmethod
    def load(path: str, use_faiss: bool = True) -> "TagBank":
        tags_path = os.path.join(path, "tags.json")
        meta_path = os.path.join(path, "meta.json")
        vec_path  = os.path.join(path, "text_embeds.npy")

        if not (os.path.exists(tags_path) and os.path.exists(meta_path) and os.path.exists(vec_path)):
            raise FileNotFoundError("TagBank files missing in directory.")

        with open(tags_path, "r", encoding="utf-8") as f:
            tags = json.load(f)["tags"]

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        vecs = np.load(vec_path, mmap_mode="r")  # memory-map for constant-time load
        dim = int(meta["dim"])

        faiss_index = None
        if use_faiss and _HAS_FAISS and os.path.exists(os.path.join(path, "faiss.index")):
            faiss_index = faiss.read_index(os.path.join(path, "faiss.index"))

        return TagBank(_TagBankData(tags=tags, vecs=vecs, dim=dim, meta=meta, faiss_index=faiss_index))

    def info(self) -> Dict:
        return dict(self._data.meta, num_tags=len(self._data.tags))

    def rank(self, image_feats: np.ndarray, topk: int = 5) -> Tuple[List[str], np.ndarray]:
        """
        image_feats: [D] or [1, D] numpy OR torch tensor. Should be L2-normalized.
        Returns (topk_tags, topk_scores) where scores are cosine similarities.
        """
        # normalize + ensure [1, D], numpy float32
        if isinstance(image_feats, torch.Tensor):
            feats = image_feats.detach().cpu().numpy()
        else:
            feats = np.asarray(image_feats)
        if feats.ndim == 1:
            feats = feats[None, :]
        feats = feats.astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

        if self._data.faiss_index is not None:
            # inner product search on normalized vectors = cosine similarity
            D, I = self._data.faiss_index.search(feats, topk)
            idxs = I[0].tolist()
            scores = D[0]
        else:
            # fall back to dense matmul against memory-mapped matrix
            # vecs might be float16—cast to float32 for matmul
            mat = np.asarray(self._data.vecs, dtype=np.float32)  # [N, D]
            scores = feats @ mat.T
            idxs = np.argpartition(scores[0], -topk)[-topk:]
            idxs = idxs[np.argsort(scores[0, idxs])[::-1]]
            scores = scores[0, idxs]

        tags = [self._data.tags[i] for i in idxs]
        return tags, scores

    # Optional: batch scoring if you have many images at once
    def rank_batch(self, image_feats_batch: np.ndarray, topk: int = 5) -> List[Tuple[List[str], np.ndarray]]:
        results = []
        for i in range(image_feats_batch.shape[0]):
            t, s = self.rank(image_feats_batch[i], topk=topk)
            results.append((t, s))
        return results
