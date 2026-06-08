"""
Fast corpus cache: float32 matrix (mmap) + compact manifest.
Avoids unpickling multi‑GB pickle blobs on panel startup.
"""
from __future__ import annotations

import gc
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from clipseek_video import ClipseekVideo, load_pickle_compat

META_NAME = "cached_embeddings.meta"
MATRIX_NAME = "cached_embeddings.matrix.npy"
# Legacy v2 cache names kept for read-only fallback in Premiere.
MATRIX_TMP_NAME = "cached_embeddings.matrix.tmp.npy"
MANIFEST_V3_NAME = "cached_embeddings.manifest"
DEFAULT_MATRIX_BIN_NAME = "cached_embeddings.matrix.bin"


def manifest_path_key(path: str) -> str:
    """Stable filesystem-free key for manifest de-duping."""
    working = str(path or "").strip().strip('"')
    working = working.replace("\\\\", "\\")
    working = working.replace("\\ ", " ")
    working = working.replace("/ ", " ")
    working = working.replace("\\", "/")
    while "//" in working:
        working = working.replace("//", "/")
    working = working.rstrip("/")
    if os.name == "nt":
        working = working.lower()
    return working


def chunks_to_numpy_2d(chunks: Any) -> np.ndarray:
    """Return (n_chunks, dim) float32 contiguous array."""
    if chunks is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(chunks, torch.Tensor):
        t = chunks.detach().cpu().float()
        if t.numel() == 0:
            return np.zeros((0, 0), dtype=np.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return np.ascontiguousarray(t.numpy().astype(np.float32, copy=False))
    if isinstance(chunks, np.ndarray):
        if chunks.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        a = np.ascontiguousarray(chunks.astype(np.float32, copy=False))
        if a.ndim == 1:
            a = a.reshape(1, -1)
        return a
    if isinstance(chunks, list):
        if len(chunks) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        parts = [np.asarray(c, dtype=np.float32) for c in chunks]
        parts = [p.reshape(1, -1) if p.ndim == 1 else p for p in parts]
        arr = np.vstack(parts)
        return np.ascontiguousarray(arr.astype(np.float32, copy=False))
    arr = np.asarray(chunks, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.ascontiguousarray(arr)


def _video_mtime_ts(obj: Any) -> float:
    p = getattr(obj, "path", None)
    if not p:
        return 0.0
    try:
        return float(os.stat(p).st_mtime)
    except OSError:
        dt = getattr(obj, "datetime", None)
        if isinstance(dt, datetime):
            return dt.timestamp()
        return 0.0


def _manifest_v3_path(embeddings_path: str) -> str:
    return os.path.join(embeddings_path, MANIFEST_V3_NAME)


def _load_v3_manifest(embeddings_path: str) -> Dict[str, Any]:
    with open(_manifest_v3_path(embeddings_path), "rb") as f:
        manifest = load_pickle_compat(f)
    if manifest.get("v") != 3:
        raise ValueError("Unsupported append cache manifest version")
    return manifest


def _list_field(manifest: Dict[str, Any], name: str) -> List[Any]:
    value = manifest.get(name, [])
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _array_field(manifest: Dict[str, Any], name: str, dtype) -> np.ndarray:
    return np.asarray(manifest.get(name, []), dtype=dtype)


def _matrix_file_path(embeddings_path: str, manifest: Dict[str, Any]) -> str:
    matrix_file = manifest.get("matrix_file") or DEFAULT_MATRIX_BIN_NAME
    return os.path.join(embeddings_path, os.path.basename(str(matrix_file)))


def v3_bundle_exists(embeddings_path: str) -> bool:
    if not os.path.isfile(_manifest_v3_path(embeddings_path)):
        return False
    try:
        manifest = _load_v3_manifest(embeddings_path)
    except Exception:
        return False
    return os.path.isfile(_matrix_file_path(embeddings_path, manifest))


def read_v3_manifest_summary(embeddings_path: str) -> Optional[Dict[str, Any]]:
    try:
        manifest = _load_v3_manifest(embeddings_path)
        return {
            "generation": int(manifest.get("generation", 0)),
            "video_count": len(manifest.get("paths", [])),
            "total_rows": int(manifest.get("total_rows", 0)),
            "embed_dim": int(manifest.get("embed_dim", 0)),
            "matrix_file": str(manifest.get("matrix_file") or DEFAULT_MATRIX_BIN_NAME),
        }
    except Exception:
        return None


def load_v3_objects(embeddings_path: str) -> List[Any]:
    """Memory-map append-only raw matrix cache and build ClipseekVideo shells."""
    manifest = _load_v3_manifest(embeddings_path)
    embed_dim = int(manifest.get("embed_dim", 0))
    total_rows = int(manifest.get("total_rows", 0))
    if embed_dim <= 0:
        raise ValueError("Invalid append cache embed_dim")
    if total_rows < 0:
        raise ValueError("Invalid append cache total_rows")

    matrix_path = _matrix_file_path(embeddings_path, manifest)
    expected_bytes = total_rows * embed_dim * np.dtype(np.float32).itemsize
    try:
        actual_bytes = os.path.getsize(matrix_path)
    except OSError as e:
        raise ValueError(f"Append cache matrix is missing: {e}") from e
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"Append cache matrix is truncated: expected at least {expected_bytes} bytes, got {actual_bytes}"
        )

    matrix = np.memmap(
        matrix_path,
        dtype=np.float32,
        mode="r",
        shape=(total_rows, embed_dim),
    )
    paths = _list_field(manifest, "paths")
    titles = _list_field(manifest, "titles")
    row_start = _array_field(manifest, "row_start", np.int64)
    n_chunks = _array_field(manifest, "n_chunks", np.int32)
    strides = _array_field(manifest, "chunk_stride_sec", np.float32)
    mtimes = _array_field(manifest, "mtime_ts", np.float64)
    generation = int(manifest.get("generation", 0))

    out: List[Any] = []
    for i, p in enumerate(paths):
        rs = int(row_start[i])
        nc = int(n_chunks[i])
        if nc <= 0:
            continue
        if rs < 0 or rs + nc > total_rows:
            raise ValueError(f"Append cache row range is out of bounds for {p}")

        obj = ClipseekVideo.__new__(ClipseekVideo)
        obj._corpus_video_index = len(out)
        obj.path = p
        obj.title = titles[i] if i < len(titles) else os.path.basename(p)
        obj.chunk_stride_sec = float(strides[i]) if i < len(strides) else 10.0
        ts = float(mtimes[i]) if i < len(mtimes) else 0.0
        obj.datetime = datetime.fromtimestamp(ts) if ts > 0 else datetime.fromtimestamp(0)
        obj.chunks = matrix[rs : rs + nc]
        obj.audio = None
        obj.shot_type = []
        obj.annotations = []
        obj.transcript = ""
        obj.sims = []
        obj.proxy = ""
        obj._corpus_matrix = matrix
        obj._corpus_row_start = rs
        obj._corpus_n_chunks = nc
        obj._corpus_cache_generation = generation
        out.append(obj)

    return out


def _bundle_mtime(embeddings_path: str) -> float:
    mf = os.path.join(embeddings_path, META_NAME)
    mx = os.path.join(embeddings_path, MATRIX_NAME)
    t = 0.0
    for p in (mf, mx):
        try:
            t = max(t, os.path.getmtime(p))
        except OSError:
            return -1.0
    return t


def v2_bundle_exists(embeddings_path: str) -> bool:
    return os.path.isfile(os.path.join(embeddings_path, META_NAME)) and os.path.isfile(
        os.path.join(embeddings_path, MATRIX_NAME)
    )


def v2_is_current(embeddings_path: str, per_video_pkl_max_mtime: float) -> bool:
    if not v2_bundle_exists(embeddings_path):
        return False
    b = _bundle_mtime(embeddings_path)
    if b < 0:
        return False
    return per_video_pkl_max_mtime <= b


def load_v2_objects(embeddings_path: str) -> List[Any]:
    """Memory-map the matrix; build ClipseekVideo shells with chunk views (no huge unpickle)."""
    meta_path = os.path.join(embeddings_path, META_NAME)
    matrix_path = os.path.join(embeddings_path, MATRIX_NAME)

    with open(meta_path, "rb") as f:
        meta = load_pickle_compat(f)

    if meta.get("v") != 2:
        raise ValueError("Unsupported embed cache meta version")

    matrix = np.load(matrix_path, mmap_mode="r")
    paths = meta["paths"]
    n = len(paths)
    out: List[Any] = []

    for i in range(n):
        rs = int(meta["row_start"][i])
        nc = int(meta["n_chunks"][i])
        sl = matrix[rs : rs + nc]

        obj = ClipseekVideo.__new__(ClipseekVideo)
        obj._corpus_video_index = i
        obj.path = paths[i]
        obj.title = meta["titles"][i] if i < len(meta["titles"]) else os.path.basename(paths[i])
        obj.chunk_stride_sec = float(meta["chunk_stride_sec"][i])
        ts = float(meta["mtime_ts"][i])
        obj.datetime = datetime.fromtimestamp(ts) if ts > 0 else datetime.fromtimestamp(0)
        obj.chunks = sl
        obj.audio = None
        obj.shot_type = []
        obj.annotations = []
        obj.transcript = ""
        obj.sims = []
        obj.proxy = ""
        # Fast search: shared mmap + row range (avoid per-video stack/copies).
        obj._corpus_matrix = matrix
        obj._corpus_row_start = rs
        obj._corpus_n_chunks = nc
        out.append(obj)

    return out


def max_per_video_embedding_mtime(embeddings_path: str) -> float:
    """Newest mtime among per-video ``*.pkl`` files (excludes ``cached_embeddings.pkl``)."""
    m = 0.0
    try:
        for f in os.scandir(embeddings_path):
            if not f.name.endswith(".pkl"):
                continue
            if f.name == "cached_embeddings.pkl":
                continue
            try:
                m = max(m, f.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return m


def load_corpus_row_owner(embeddings_path: str) -> np.ndarray:
    """
    For each global chunk row index, the corpus video index (meta row) that owns it.
    Shape (total_chunks,). Used for FAISS candidate filtering.
    """
    meta_path = os.path.join(embeddings_path, META_NAME)
    with open(meta_path, "rb") as f:
        meta = load_pickle_compat(f)
    n_v = len(meta["paths"])
    if n_v == 0:
        return np.zeros(0, dtype=np.int32)
    rs = meta["row_start"]
    nc = meta["n_chunks"]
    total = int(rs[-1] + nc[-1])
    out = np.empty(total, dtype=np.int32)
    for v in range(n_v):
        a = int(rs[v])
        b = int(nc[v])
        out[a : a + b] = v
    return out


def read_manifest_video_count(embeddings_path: str) -> Optional[int]:
    """Number of videos in the current cache manifest, or None if missing/invalid."""
    v3_summary = read_v3_manifest_summary(embeddings_path)
    if v3_summary is not None:
        return int(v3_summary["video_count"])

    p = os.path.join(embeddings_path, META_NAME)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "rb") as f:
            meta = load_pickle_compat(f)
        if meta.get("v") != 2:
            return None
        return len(meta["paths"])
    except Exception:
        return None
