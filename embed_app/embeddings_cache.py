"""
Build and maintain the append corpus cache for clipseek_panel/io.PersistentSearch.load_objs.
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
    _bundle_mtime,
    append_v3_objects,
    load_v3_objects,
    load_v2_objects,
    MANIFEST_V3_NAME,
    manifest_path_key,
    read_v3_manifest_summary,
    v3_bundle_exists,
    v2_bundle_exists,
    write_v3_from_objects,
    write_v3_from_pkls_streaming,
)

# Old monolithic cache filename — excluded from per-video scans; removed on explicit regenerate.
LEGACY_MONOLITHIC_PKL = "cached_embeddings.pkl"

# Tune: balance freshness vs I/O when embedding huge libraries. Saves append
# dirty videos and publish a new manifest generation.
CACHE_SAVE_EVERY_N_VIDEOS = 500
CACHE_SAVE_MIN_INTERVAL_SEC = 1200


# Match extension/io.py — used as dict keys when merging.
def merge_path_key(path: str) -> str:
    return manifest_path_key(path)


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
    """Write a complete v3 append-cache generation."""
    write_v3_from_objects(merged, embeddings_path)


def regenerate_mmap_cache(
    embeddings_path: str,
    *,
    progress: Optional[Callable[[float, str], None]] = None,
) -> int:
    """
    Rebuild append cache from all per-video .pkl files in ``embeddings_path``.
    Removes legacy ``cached_embeddings.pkl`` if present. Returns number of videos indexed.

    If ``progress`` is set, it is called periodically with ``(fraction_0_to_1, message)``.
    """
    def _report_stream(frac: float, msg: str) -> None:
        if progress:
            progress(min(1.0, max(0.0, frac)), msg)

    os.makedirs(embeddings_path, exist_ok=True)
    _report_stream(0.0, "Scanning embedding folder...")

    disk_pkls: List[str] = []
    try:
        for f in os.scandir(embeddings_path):
            if f.name.endswith(".pkl") and f.name != LEGACY_MONOLITHIC_PKL:
                disk_pkls.append(f.path)
    except OSError as e:
        logging.error("Cannot read embedding folder: %s", e)
        _report_stream(1.0, "Could not read embedding folder.")
        return 0

    total_files = len(disk_pkls)
    logging.info("Streaming append cache rebuild from %d embedding file(s).", total_files)
    if total_files == 0:
        _report_stream(1.0, "No videos to index (folder empty or no valid .pkl files).")
        return 0

    def _stream_progress(processed: int, total: int, indexed: int) -> None:
        if not total:
            return
        frac = 0.98 * (processed / total)
        _report_stream(
            frac,
            f"Streaming append cache ({processed} / {total} files; {indexed} videos indexed)...",
        )

    count = write_v3_from_pkls_streaming(
        disk_pkls,
        embeddings_path,
        _load_one_pkl,
        progress=_stream_progress,
    )
    if count:
        _report_stream(1.0, f"Finished ({count} videos indexed).")
    else:
        _report_stream(1.0, "No videos to index (folder empty or no valid .pkl files).")

    legacy = os.path.join(embeddings_path, LEGACY_MONOLITHIC_PKL)
    if os.path.isfile(legacy):
        try:
            os.remove(legacy)
            logging.info("Removed legacy %s.", LEGACY_MONOLITHIC_PKL)
        except OSError as e:
            logging.warning("Could not remove legacy %s: %s", LEGACY_MONOLITHIC_PKL, e)
    return count


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
        self._dirty_keys = set()

    def _replace_index_from_mmap(self) -> None:
        """Reload in-memory index from mmap files (after a successful write)."""
        if v3_bundle_exists(self.embeddings_path):
            objs = load_v3_objects(self.embeddings_path)
        elif v2_bundle_exists(self.embeddings_path):
            objs = load_v2_objects(self.embeddings_path)
        else:
            return
        self._by_path = {merge_path_key(getattr(o, "path", "")): o for o in objs}

    def _rebuild_index_from_disk(self) -> None:
        n_disk = _count_per_video_pkls(self.embeddings_path)
        if n_disk == 0:
            self._by_path = {}
            logging.info("No per-video embeddings in folder; append cache empty.")
            self._last_save_time = time.monotonic()
            self._videos_since_save = 0
            self._dirty_keys = set()
            return
        logging.info("Building append cache from %d per-video .pkl file(s).", n_disk)
        merged = rebuild_merged_list_from_disk(self.embeddings_path)
        self._by_path = {merge_path_key(getattr(o, "path", "")): o for o in merged}
        if merged:
            try:
                atomic_write_cache(self.embeddings_path, merged)
                logging.info("Wrote append cache (%d videos).", len(merged))
                try:
                    self._replace_index_from_mmap()
                except Exception as e:
                    logging.warning("Could not reload after rebuild save: %s", e)
            except Exception as e:
                logging.error("Could not write append cache: %s", e)
        self._last_save_time = time.monotonic()
        self._videos_since_save = 0
        self._dirty_keys = set()

    def initialize(self) -> None:
        """
        Load append cache when available and only fall back to a full rebuild
        when the bundle is missing or unreadable.

        Strategy:
          1. If the v3 manifest exists, mmap-load it as the base index.
          2. Incrementally load any per-video ``.pkl`` files whose mtime is
             newer than the manifest and merge them in (covers partial runs).
          3. If only the legacy v2 bundle exists, convert it once to v3.
          4. A full rebuild only happens when no usable bundle exists.
        """
        n_disk = _count_per_video_pkls(self.embeddings_path)

        if v3_bundle_exists(self.embeddings_path):
            try:
                objs = load_v3_objects(self.embeddings_path)
                self._by_path = {
                    merge_path_key(getattr(o, "path", "")): o for o in objs
                }
                manifest_ts = os.path.getmtime(
                    os.path.join(self.embeddings_path, MANIFEST_V3_NAME)
                )
                summary = read_v3_manifest_summary(self.embeddings_path) or {}
                logging.info(
                    "Using append cache generation %s (%d entries; %d per-video .pkl files on disk).",
                    summary.get("generation", "?"),
                    len(self._by_path),
                    n_disk,
                )

                added = self._merge_newer_pkls_into_index(manifest_ts)
                if added:
                    logging.info(
                        "Merged %d per-video .pkl(s) newer than the append cache.",
                        added,
                    )

                self._last_save_time = time.monotonic()
                self._videos_since_save = added
                return
            except Exception as e:
                logging.warning(
                    "Append cache unreadable (%s); trying legacy cache or per-video .pkls.",
                    e,
                )

        if v2_bundle_exists(self.embeddings_path):
            try:
                objs = load_v2_objects(self.embeddings_path)
                self._by_path = {
                    merge_path_key(getattr(o, "path", "")): o for o in objs
                }
                bundle_ts = _bundle_mtime(self.embeddings_path)
                logging.info(
                    "Converting legacy mmap cache (%d entries; %d per-video .pkl files on disk).",
                    len(self._by_path),
                    n_disk,
                )

                added = self._merge_newer_pkls_into_index(bundle_ts)
                if added:
                    logging.info(
                        "Merged %d per-video .pkl(s) newer than the bundle.",
                        added,
                    )

                write_v3_from_objects(list(self._by_path.values()), self.embeddings_path)
                self._replace_index_from_mmap()
                self._last_save_time = time.monotonic()
                self._videos_since_save = 0
                self._dirty_keys = set()
                return
            except Exception as e:
                logging.warning(
                    "Mmap cache unreadable (%s); rebuilding from per-video .pkls.",
                    e,
                )

        self._rebuild_index_from_disk()

    def _merge_newer_pkls_into_index(self, bundle_ts: float) -> int:
        """
        Load any per-video ``.pkl`` files modified after ``bundle_ts`` and
        merge them into ``self._by_path``. Returns the number merged.
        """
        if bundle_ts <= 0:
            return 0
        newer: List[str] = []
        try:
            for f in os.scandir(self.embeddings_path):
                if not f.name.endswith(".pkl") or f.name == LEGACY_MONOLITHIC_PKL:
                    continue
                try:
                    if f.stat().st_mtime > bundle_ts:
                        newer.append(f.path)
                except OSError:
                    continue
        except OSError as e:
            logging.warning("Could not scan embedding folder for newer .pkls: %s", e)
            return 0

        if not newer:
            return 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            loaded = list(filter(None, executor.map(_load_one_pkl, newer)))

        for o in loaded:
            key = merge_path_key(getattr(o, "path", ""))
            self._by_path[key] = o
            self._dirty_keys.add(key)
        return len(loaded)

    def record_completed_video(self, vid_obj: Any) -> None:
        with self._lock:
            key = merge_path_key(getattr(vid_obj, "path", ""))
            self._by_path[key] = vid_obj
            self._dirty_keys.add(key)
            self._videos_since_save += 1

    def maybe_periodic_save(
        self,
        every_n_videos: int,
        min_interval_sec: float,
        force: bool = False,
    ) -> bool:
        """Write cache if thresholds met or force. Returns True if wrote."""
        with self._lock:
            n = len(self._dirty_keys)
            elapsed = time.monotonic() - self._last_save_time
            should = force or (n >= every_n_videos) or (elapsed >= min_interval_sec and n > 0)
            if not should:
                return False
            dirty_keys = frozenset(self._dirty_keys)
            dirty_objects = [
                self._by_path[k] for k in dirty_keys if k in self._by_path
            ]

            try:
                append_v3_objects(dirty_objects, self.embeddings_path)
                self._dirty_keys.difference_update(dirty_keys)
                self._videos_since_save = len(self._dirty_keys)
                self._last_save_time = time.monotonic()
                try:
                    self._replace_index_from_mmap()
                except Exception as e:
                    logging.warning("Could not refresh cache index after save: %s", e)
                logging.info(
                    "Updated append cache (%d dirty videos, periodic/forced=%s).",
                    len(dirty_objects),
                    force,
                )
                return True
            except Exception as e:
                logging.error("Failed to update append cache: %s", e)
                return False

    def final_save(self) -> None:
        with self._lock:
            if not self._dirty_keys:
                logging.info("Append cache already current; skipping final save.")
                return
        self.maybe_periodic_save(1, 0.0, force=True)
