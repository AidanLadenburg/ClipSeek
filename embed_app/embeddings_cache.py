"""
Build and maintain mmap corpus cache (cached_embeddings.matrix.npy + cached_embeddings.meta)
for extension/io.PersistentSearch.load_objs.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import torch

from clipseek_video import load_clipseek_video_pickle
from embed_cache_v2 import (
    load_v2_objects,
    max_per_video_embedding_mtime,
    read_manifest_video_count,
    save_v2_from_objects,
    v2_bundle_exists,
    v2_is_current,
)

# Old monolithic cache filename — excluded from per-video scans; removed on explicit regenerate.
LEGACY_MONOLITHIC_PKL = "cached_embeddings.pkl"

# Tune: balance freshness vs I/O when embedding huge libraries.
CACHE_SAVE_EVERY_N_VIDEOS = 50
CACHE_SAVE_MIN_INTERVAL_SEC = 45.0


# Match extension/io.py — used as dict keys when merging.
def merge_path_key(path: str) -> str:
    working = path.replace("\\\\", "\\")
    working = working.replace("//", "/")
    working = working.replace("\\ ", "\\")
    working = working.replace("/ ", "/")
    working = working.replace("\\", "/")
    if os.path.exists(working):
        return working
    if os.path.exists(working.replace(".mp4", ".mov.mp4")):
        return working.replace(".mp4", ".mov.mp4")
    if os.path.exists(working.replace(".mp4", ".mp4.mp4")):
        return working.replace(".mp4", ".mp4.mp4")
    if os.path.exists(working.replace(".mp4", ".wav.mp4")):
        return working.replace(".mp4", ".wav.mp4")
    if os.path.exists(working.replace(".mp4", ".mp3.mp4")):
        return working.replace(".mp4", ".mp3.mp4")
    return path


def _count_per_video_pkls(embeddings_path: str) -> int:
    n = 0
    try:
        for f in os.scandir(embeddings_path):
            if f.name.endswith(".pkl") and f.name != LEGACY_MONOLITHIC_PKL:
                n += 1
    except OSError:
        pass
    return n


def _load_one_pkl(p: str):
    try:
        obj = load_clipseek_video_pickle(p)
        if obj is None:
            return None
        ch = obj.chunks
        if isinstance(ch, list) and len(ch) == 0:
            return None
        if isinstance(ch, torch.Tensor) and ch.numel() == 0:
            return None
        return obj
    except Exception as e:
        logging.warning("Error loading %s: %s", p, e)
        return None


def rebuild_merged_list_from_disk(
    embeddings_path: str,
    chunk_size: int = 1000,
    *,
    on_load_progress: Optional[Callable[[int, int], None]] = None,
) -> List[Any]:
    """Load every per-video .pkl and merge by path (same as extension/io.py)."""
    disk_pkls: List[str] = []
    try:
        for f in os.scandir(embeddings_path):
            if f.name.endswith(".pkl") and f.name != LEGACY_MONOLITHIC_PKL:
                disk_pkls.append(f.path)
    except OSError as e:
        logging.error("Cannot read embedding folder: %s", e)
        return []

    total_files = len(disk_pkls)
    logging.info("Rebuilding merged cache from %d embedding file(s).", total_files)

    if total_files == 0:
        if on_load_progress:
            on_load_progress(0, 0)
        return []

    if on_load_progress:
        on_load_progress(0, total_files)

    all_objects: List[Any] = []
    for i in range(0, total_files, chunk_size):
        chunk = disk_pkls[i : i + chunk_size]
        with ThreadPoolExecutor(max_workers=8) as executor:
            loaded = list(filter(None, executor.map(_load_one_pkl, chunk)))
        all_objects.extend(loaded)
        gc.collect()
        processed = min(i + len(chunk), total_files)
        if on_load_progress:
            on_load_progress(processed, total_files)

    by_path: Dict[str, Any] = {}
    for obj in all_objects:
        key = merge_path_key(getattr(obj, "path", ""))
        by_path[key] = obj
    return list(by_path.values())


def atomic_write_cache(embeddings_path: str, merged: List[Any]) -> None:
    """Write mmap matrix + meta."""
    save_v2_from_objects(merged, embeddings_path)


def regenerate_mmap_cache(
    embeddings_path: str,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
) -> int:
    """
    Rebuild mmap cache from all per-video .pkl files in ``embeddings_path``.
    Removes legacy ``cached_embeddings.pkl`` if present. Returns number of videos indexed.

    If ``progress`` is set, it is called periodically with ``(fraction_0_to_1, message)``.
    """
    LOAD_END = 0.88

    def _report(frac: float, msg: str) -> None:
        if progress:
            progress(min(1.0, max(0.0, frac)), msg)

    os.makedirs(embeddings_path, exist_ok=True)
    _report(0.0, "Scanning embedding folder…")

    def _load_progress(processed: int, total: int) -> None:
        if not total:
            return
        frac = LOAD_END * (processed / total)
        _report(frac, f"Loading embeddings ({processed} / {total} files)…")

    merged = rebuild_merged_list_from_disk(
        embeddings_path, on_load_progress=_load_progress
    )
    if merged:
        _report(0.9, "Writing mmap cache to disk…")
        save_v2_from_objects(merged, embeddings_path)
        _report(1.0, "Finished.")
    else:
        _report(1.0, "No videos to index (folder empty or no valid .pkl files).")
    legacy = os.path.join(embeddings_path, LEGACY_MONOLITHIC_PKL)
    if os.path.isfile(legacy):
        try:
            os.remove(legacy)
            logging.info("Removed legacy %s.", LEGACY_MONOLITHIC_PKL)
        except OSError as e:
            logging.warning("Could not remove legacy %s: %s", LEGACY_MONOLITHIC_PKL, e)
    return len(merged)


class EmbeddingIndexCache:
    """
    In-memory index keyed by merge_path_key(obj.path), kept in sync with disk
    via periodic atomic writes. Avoids re-reading every per-video pkl on each save.
    """

    def __init__(self, embeddings_path: str):
        self.embeddings_path = embeddings_path
        self._lock = threading.Lock()
        self._by_path: Dict[str, Any] = {}
        self._last_save_time = 0.0
        self._videos_since_save = 0

    def _replace_index_from_mmap(self) -> None:
        """Reload in-memory index from mmap files (after a successful write)."""
        if not v2_bundle_exists(self.embeddings_path):
            return
        objs = load_v2_objects(self.embeddings_path)
        self._by_path = {merge_path_key(getattr(o, "path", "")): o for o in objs}

    def _merge_mmap_reload_with_concurrent_adds(self, saved_keys: frozenset) -> None:
        """
        Reload mmap-backed rows from disk after save, but keep paths that were not part
        of that save snapshot (e.g. a video finished while the mmap file was being written).
        """
        if not v2_bundle_exists(self.embeddings_path):
            return
        objs = load_v2_objects(self.embeddings_path)
        fresh = {merge_path_key(getattr(o, "path", "")): o for o in objs}
        for k, o in self._by_path.items():
            if k not in saved_keys:
                fresh[k] = o
        self._by_path = fresh

    def _rebuild_index_from_disk(self) -> None:
        n_disk = _count_per_video_pkls(self.embeddings_path)
        if n_disk == 0:
            self._by_path = {}
            logging.info("No per-video embeddings in folder; mmap cache empty.")
            self._last_save_time = time.monotonic()
            self._videos_since_save = 0
            return
        logging.info("Building mmap cache from %d per-video .pkl file(s).", n_disk)
        merged = rebuild_merged_list_from_disk(self.embeddings_path)
        self._by_path = {merge_path_key(getattr(o, "path", "")): o for o in merged}
        if merged:
            try:
                atomic_write_cache(self.embeddings_path, merged)
                logging.info("Wrote mmap cache (%d videos).", len(merged))
                try:
                    self._replace_index_from_mmap()
                except Exception as e:
                    logging.warning("Could not mmap-reload after rebuild save: %s", e)
            except Exception as e:
                logging.error("Could not write mmap cache: %s", e)
        self._last_save_time = time.monotonic()
        self._videos_since_save = 0

    def initialize(self) -> None:
        """Load mmap cache or rebuild from per-video .pkls."""
        n_disk = _count_per_video_pkls(self.embeddings_path)
        mt = max_per_video_embedding_mtime(self.embeddings_path)

        if v2_bundle_exists(self.embeddings_path) and v2_is_current(self.embeddings_path, mt):
            n_meta = read_manifest_video_count(self.embeddings_path)
            if n_meta is not None and n_meta == n_disk:
                try:
                    objs = load_v2_objects(self.embeddings_path)
                    self._by_path = {merge_path_key(getattr(o, "path", "")): o for o in objs}
                    logging.info("Using mmap cache (%d videos).", len(self._by_path))
                    self._last_save_time = time.monotonic()
                    self._videos_since_save = 0
                    return
                except Exception as e:
                    logging.warning("Mmap cache unreadable (%s); rebuilding from per-video .pkls.", e)
                    self._rebuild_index_from_disk()
                    return
            if n_meta is None:
                logging.info("Could not read mmap manifest; rebuilding from disk.")
            else:
                logging.info(
                    "Mmap manifest count (%d) != per-video pkls (%d); rebuilding from disk.",
                    n_meta,
                    n_disk,
                )
            self._rebuild_index_from_disk()
            return

        self._rebuild_index_from_disk()

    def record_completed_video(self, vid_obj: Any) -> None:
        with self._lock:
            self._by_path[merge_path_key(getattr(vid_obj, "path", ""))] = vid_obj
            self._videos_since_save += 1

    def maybe_periodic_save(
        self,
        every_n_videos: int,
        min_interval_sec: float,
        force: bool = False,
    ) -> bool:
        """Write cache if thresholds met or force. Returns True if wrote."""
        with self._lock:
            n = self._videos_since_save
            elapsed = time.monotonic() - self._last_save_time
            should = force or (n >= every_n_videos) or (elapsed >= min_interval_sec and n > 0)
            if not should:
                return False
            merged = list(self._by_path.values())
            saved_keys = frozenset(
                merge_path_key(getattr(o, "path", "")) for o in merged
            )
            self._videos_since_save = 0
            self._last_save_time = time.monotonic()

        try:
            atomic_write_cache(self.embeddings_path, merged)
            with self._lock:
                try:
                    self._merge_mmap_reload_with_concurrent_adds(saved_keys)
                except Exception as e:
                    logging.warning("Could not refresh mmap index after periodic save: %s", e)
            logging.info(
                "Updated mmap cache (%d videos, periodic/forced=%s).",
                len(merged),
                force,
            )
            return True
        except Exception as e:
            logging.error("Failed to update mmap cache: %s", e)
            return False

    def final_save(self) -> None:
        self.maybe_periodic_save(1, 0.0, force=True)
