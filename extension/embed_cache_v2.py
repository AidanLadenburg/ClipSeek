"""
Fast corpus cache: float32 matrix (mmap) + compact manifest.
Avoids unpickling multi‑GB pickle blobs on panel startup.
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import torch

from clipseek_video import ClipseekVideo

META_NAME = "cached_embeddings.meta"
MATRIX_NAME = "cached_embeddings.matrix.npy"
# Temp file for atomic matrix write. Must end in ".npy" so np.save(path, ...) does not
# append another ".npy" (e.g. "...matrix.npy.tmp" wrongly became "...matrix.npy.tmp.npy").
MATRIX_TMP_NAME = "cached_embeddings.matrix.tmp.npy"


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


def save_v2_from_objects(objects: List[Any], embeddings_path: str) -> None:
    """Pack merged ClipseekVideo-like objects into matrix + meta (atomic)."""
    os.makedirs(embeddings_path, exist_ok=True)
    parts: List[np.ndarray] = []
    paths: List[str] = []
    titles: List[str] = []
    strides: List[float] = []
    mtimes: List[float] = []

    embed_dim = 0
    for obj in objects:
        arr = chunks_to_numpy_2d(getattr(obj, "chunks", None))
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            continue
        if embed_dim == 0:
            embed_dim = int(arr.shape[1])
        elif int(arr.shape[1]) != embed_dim:
            raise ValueError(
                f"Inconsistent embed dim: expected {embed_dim}, got {arr.shape[1]} for {getattr(obj, 'path', '?')}"
            )
        parts.append(arr)
        paths.append(str(getattr(obj, "path", "")))
        titles.append(str(getattr(obj, "title", "") or os.path.basename(paths[-1])))
        strides.append(float(getattr(obj, "chunk_stride_sec", 10.0)))
        mtimes.append(_video_mtime_ts(obj))

    if not parts:
        big = np.zeros((0, embed_dim), dtype=np.float32)
    else:
        big = np.vstack(parts)

    row_start = np.zeros(len(paths), dtype=np.int64)
    n_chunks = np.zeros(len(paths), dtype=np.int32)
    off = 0
    for i, p in enumerate(parts):
        r, _ = p.shape
        row_start[i] = off
        n_chunks[i] = r
        off += r

    meta = {
        "v": 2,
        "embed_dim": int(big.shape[1]) if big.size else embed_dim,
        "paths": paths,
        "titles": titles,
        "row_start": row_start,
        "n_chunks": n_chunks,
        "chunk_stride_sec": np.asarray(strides, dtype=np.float32),
        "mtime_ts": np.asarray(mtimes, dtype=np.float64),
    }

    matrix_path = os.path.join(embeddings_path, MATRIX_NAME)
    meta_path = os.path.join(embeddings_path, META_NAME)
    tmp_m = os.path.join(embeddings_path, MATRIX_TMP_NAME)
    tmp_meta = meta_path + ".tmp"

    # Remove stale/intermediate names from older saves or failed runs.
    for stale in (matrix_path + ".tmp", matrix_path + ".tmp.npy"):
        try:
            if os.path.isfile(stale):
                os.remove(stale)
        except OSError:
            pass

    np.save(tmp_m, big)
    os.replace(tmp_m, matrix_path)

    with open(tmp_meta, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_meta, meta_path)


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
        meta = pickle.load(f)

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
        meta = pickle.load(f)
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
    """Number of videos in the mmap cache manifest, or None if missing/invalid."""
    p = os.path.join(embeddings_path, META_NAME)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "rb") as f:
            meta = pickle.load(f)
        if meta.get("v") != 2:
            return None
        return len(meta["paths"])
    except Exception:
        return None
