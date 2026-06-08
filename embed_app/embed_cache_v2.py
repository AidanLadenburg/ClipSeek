"""
Fast corpus cache: float32 matrix (mmap) + compact manifest.
Avoids unpickling multi‑GB pickle blobs on panel startup.
"""
from __future__ import annotations

import gc
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from clipseek_video import ClipseekVideo, load_pickle_compat

META_NAME = "cached_embeddings.meta"
MATRIX_NAME = "cached_embeddings.matrix.npy"
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


def release_v2_shared_mmap_before_matrix_replace(objects: List[Any]) -> None:
    """
    Close the shared mmap backing corpus rows before replacing cached_embeddings.matrix.npy.

    On Windows, os.replace fails with 'Access is denied' if the destination file is still
    open (e.g. np.load(..., mmap_mode='r')). Call after all chunk data has been copied out.
    """
    matrix = None
    for obj in objects:
        m = getattr(obj, "_corpus_matrix", None)
        if m is not None:
            matrix = m
            break
    if matrix is None:
        return
    try:
        mm = getattr(matrix, "_mmap", None)
        if mm is not None:
            mm.close()
        elif isinstance(matrix, np.memmap) and hasattr(matrix, "close"):
            matrix.close()
    except OSError:
        pass
    for obj in objects:
        if getattr(obj, "_corpus_matrix", None) is matrix:
            obj._corpus_matrix = None
            obj.chunks = None
    del matrix
    gc.collect()


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


def _write_v3_manifest_atomic(embeddings_path: str, manifest: Dict[str, Any]) -> None:
    manifest_path = _manifest_v3_path(embeddings_path)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, manifest_path)


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


def _next_matrix_bin_name(embeddings_path: str) -> str:
    default_path = os.path.join(embeddings_path, DEFAULT_MATRIX_BIN_NAME)
    if not os.path.exists(default_path):
        return DEFAULT_MATRIX_BIN_NAME
    return f"cached_embeddings.matrix.{time.time_ns()}.bin"


def write_v3_from_objects(objects: List[Any], embeddings_path: str) -> None:
    """
    Write a complete append-cache generation from objects.

    This is used by the embedding app for explicit regeneration/conversion. It writes
    a new matrix filename when a previous raw matrix exists, so open panel mmaps are
    not replaced.
    """
    os.makedirs(embeddings_path, exist_ok=True)
    matrix_file = _next_matrix_bin_name(embeddings_path)
    matrix_path = os.path.join(embeddings_path, matrix_file)
    tmp_matrix = matrix_path + ".tmp"

    paths: List[str] = []
    path_keys: List[str] = []
    titles: List[str] = []
    strides: List[float] = []
    mtimes: List[float] = []
    row_start: List[int] = []
    n_chunks: List[int] = []
    embed_dim = 0
    row_cursor = 0

    with open(tmp_matrix, "wb") as f:
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
            f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C"))
            p = str(getattr(obj, "path", ""))
            paths.append(p)
            path_keys.append(manifest_path_key(p))
            titles.append(str(getattr(obj, "title", "") or os.path.basename(p)))
            strides.append(float(getattr(obj, "chunk_stride_sec", 10.0)))
            mtimes.append(_video_mtime_ts(obj))
            row_start.append(row_cursor)
            n_chunks.append(int(arr.shape[0]))
            row_cursor += int(arr.shape[0])
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_matrix, matrix_path)

    old_summary = read_v3_manifest_summary(embeddings_path)
    old_generation = int(old_summary["generation"]) if old_summary else 0
    manifest = {
        "v": 3,
        "generation": old_generation + 1,
        "matrix_file": matrix_file,
        "embed_dim": int(embed_dim),
        "total_rows": int(row_cursor),
        "paths": paths,
        "path_keys": path_keys,
        "titles": titles,
        "row_start": np.asarray(row_start, dtype=np.int64),
        "n_chunks": np.asarray(n_chunks, dtype=np.int32),
        "chunk_stride_sec": np.asarray(strides, dtype=np.float32),
        "mtime_ts": np.asarray(mtimes, dtype=np.float64),
    }
    _write_v3_manifest_atomic(embeddings_path, manifest)


def append_v3_objects(objects: List[Any], embeddings_path: str) -> None:
    """Append new/updated object rows to the raw matrix and publish a new manifest generation."""
    if not objects:
        return

    os.makedirs(embeddings_path, exist_ok=True)
    if v3_bundle_exists(embeddings_path):
        manifest = _load_v3_manifest(embeddings_path)
    else:
        manifest = {
            "v": 3,
            "generation": 0,
            "matrix_file": DEFAULT_MATRIX_BIN_NAME,
            "embed_dim": 0,
            "total_rows": 0,
            "paths": [],
            "path_keys": [],
            "titles": [],
            "row_start": np.asarray([], dtype=np.int64),
            "n_chunks": np.asarray([], dtype=np.int32),
            "chunk_stride_sec": np.asarray([], dtype=np.float32),
            "mtime_ts": np.asarray([], dtype=np.float64),
        }

    embed_dim = int(manifest.get("embed_dim", 0))
    parts: List[Tuple[Any, np.ndarray]] = []
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
        parts.append((obj, np.ascontiguousarray(arr, dtype=np.float32)))

    if not parts:
        return

    matrix_file = str(manifest.get("matrix_file") or DEFAULT_MATRIX_BIN_NAME)
    matrix_path = os.path.join(embeddings_path, os.path.basename(matrix_file))
    row_bytes = embed_dim * np.dtype(np.float32).itemsize
    manifest_rows = int(manifest.get("total_rows", 0))
    try:
        file_bytes = os.path.getsize(matrix_path)
    except OSError:
        if manifest_rows > 0:
            raise ValueError("Append cache manifest points to a missing matrix file")
        file_bytes = 0

    if file_bytes < manifest_rows * row_bytes:
        raise ValueError("Append cache matrix is shorter than the committed manifest")

    row_cursor = (file_bytes + row_bytes - 1) // row_bytes if row_bytes else 0
    pad_bytes = row_cursor * row_bytes - file_bytes

    paths = _list_field(manifest, "paths")
    path_keys = _list_field(manifest, "path_keys")
    if len(path_keys) != len(paths):
        path_keys = [manifest_path_key(p) for p in paths]
    titles = _list_field(manifest, "titles")
    row_start = _array_field(manifest, "row_start", np.int64).tolist()
    n_chunks = _array_field(manifest, "n_chunks", np.int32).tolist()
    strides = _array_field(manifest, "chunk_stride_sec", np.float32).tolist()
    mtimes = _array_field(manifest, "mtime_ts", np.float64).tolist()
    key_to_index = {str(k): i for i, k in enumerate(path_keys)}

    with open(matrix_path, "ab") as f:
        if pad_bytes:
            f.write(b"\0" * pad_bytes)
        for obj, arr in parts:
            p = str(getattr(obj, "path", ""))
            key = manifest_path_key(p)
            rs = int(row_cursor)
            nc = int(arr.shape[0])
            f.write(arr.tobytes(order="C"))
            row_cursor += nc

            if key in key_to_index:
                i = key_to_index[key]
                paths[i] = p
                path_keys[i] = key
                titles[i] = str(getattr(obj, "title", "") or os.path.basename(p))
                row_start[i] = rs
                n_chunks[i] = nc
                strides[i] = float(getattr(obj, "chunk_stride_sec", 10.0))
                mtimes[i] = _video_mtime_ts(obj)
            else:
                key_to_index[key] = len(paths)
                paths.append(p)
                path_keys.append(key)
                titles.append(str(getattr(obj, "title", "") or os.path.basename(p)))
                row_start.append(rs)
                n_chunks.append(nc)
                strides.append(float(getattr(obj, "chunk_stride_sec", 10.0)))
                mtimes.append(_video_mtime_ts(obj))
        f.flush()
        os.fsync(f.fileno())

    new_manifest = {
        "v": 3,
        "generation": int(manifest.get("generation", 0)) + 1,
        "matrix_file": os.path.basename(matrix_file),
        "embed_dim": int(embed_dim),
        "total_rows": int(row_cursor),
        "paths": paths,
        "path_keys": path_keys,
        "titles": titles,
        "row_start": np.asarray(row_start, dtype=np.int64),
        "n_chunks": np.asarray(n_chunks, dtype=np.int32),
        "chunk_stride_sec": np.asarray(strides, dtype=np.float32),
        "mtime_ts": np.asarray(mtimes, dtype=np.float64),
    }
    _write_v3_manifest_atomic(embeddings_path, new_manifest)


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

    release_v2_shared_mmap_before_matrix_replace(objects)

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
        obj._corpus_matrix = matrix
        obj._corpus_row_start = rs
        obj._corpus_n_chunks = nc
        out.append(obj)

    return out


def load_corpus_row_owner(embeddings_path: str) -> np.ndarray:
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
