import sys
import json
import pickle
import os
import torch
import time
import math
import traceback
from datetime import datetime
import gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

# Rows of mmap gathered to GPU per batched similarity matmul (tune for VRAM).
SEARCH_GATHER_CHUNK_ROWS = 65536

from cosmos_embedder import CosmosEmbedder
from clipseek_video import load_clipseek_video_pickle
from embed_cache_v2 import (
    load_v2_objects,
    max_per_video_embedding_mtime,
    save_v2_from_objects,
    v2_bundle_exists,
    v2_is_current,
)

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
    _FAISS_IMPORT_ERROR = None
except ImportError as _faiss_err:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
    _FAISS_IMPORT_ERROR = str(_faiss_err)
    sys.stderr.write(
        "ClipSeek: faiss import failed; approximate (FAISS) text search is disabled. "
        "Install with: pip install faiss-cpu\n"
        f"  ({_FAISS_IMPORT_ERROR})\n"
    )

# FAISS: probe video-level candidates, then exact GPU rerank on those videos.
# The old HNSW implementation indexed every chunk at first search, which could take
# minutes or exhaust memory on large libraries before the panel saw a result.
FAISS_MAX_RERANK_VIDEOS = 1000
FAISS_MIN_RERANK_VIDEOS = 100
FAISS_RERANK_FRACTION = 0.35
FAISS_ADD_BATCH = 10000
FAISS_PROBE_REP_MULT = 8
FAISS_MIN_PROBE_REPS = 5000


def _clipseek_ui(phase: str, message: str, **extra) -> None:
    """Single-line JSON for the CEP panel (Premiere alerts + UI); not mixed with search JSON."""
    payload = {"phase": phase, "message": message, **extra}
    sys.stdout.write("CLIPSEEK_UI " + json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _clipseek_error(message: str, *, exc: BaseException = None, phase: str = "error") -> None:
    """Send an error event to the panel; full traceback in ``traceback`` for debug-mode display."""
    payload = {"phase": phase, "message": message, "level": "error"}
    if exc is not None:
        payload["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    sys.stdout.write("CLIPSEEK_UI " + json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def fix_path(path):
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
    print(f"failed to find working path {path}")
    return path


def chunks_to_tensor(chunks, device):
    if chunks is None:
        return torch.zeros(0, device=device)
    if isinstance(chunks, torch.Tensor):
        t = chunks.float().to(device)
    elif isinstance(chunks, list):
        if len(chunks) == 0:
            return torch.zeros(0, device=device)
        parts = [np.asarray(c, dtype=np.float32) for c in chunks]
        parts = [p.reshape(1, -1) if p.ndim == 1 else p for p in parts]
        arr = np.vstack(parts)
        t = torch.from_numpy(arr).to(device)
    elif isinstance(chunks, np.ndarray):
        arr = np.ascontiguousarray(chunks.astype(np.float32, copy=False))
        if not arr.flags.writeable:
            arr = arr.copy()
        t = torch.from_numpy(arr).to(device)
    else:
        t = torch.as_tensor(chunks, dtype=torch.float32, device=device)
    return t


def chunk_stride_sec(obj):
    return float(getattr(obj, "chunk_stride_sec", 10.0))


def _vid_objs_shared_corpus_matrix(vid_objs):
    """If all objects share the same mmap matrix (v2 cache), return it; else None."""
    if not vid_objs:
        return None
    m0 = getattr(vid_objs[0], "_corpus_matrix", None)
    if m0 is None:
        return None
    if not all(getattr(o, "_corpus_matrix", None) is m0 for o in vid_objs):
        return None
    return m0


def _gather_chunk_indices_and_split_sizes(vid_objs):
    """Concatenate corpus row indices in ``vid_objs`` order; per-video chunk counts."""
    parts = []
    sizes = []
    for o in vid_objs:
        rs = int(getattr(o, "_corpus_row_start"))
        nc = int(getattr(o, "_corpus_n_chunks"))
        if nc > 0:
            parts.append(np.arange(rs, rs + nc, dtype=np.int64))
        sizes.append(nc)
    gather = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return gather, sizes


def cosine_query_to_chunks(query_feat, stacked_chunks):
    """query_feat (n_q, d) or (d,), stacked (n_s, d) — returns (n_s,) similarity in [-1,1]."""
    q = query_feat.float()
    s = stacked_chunks.float()
    if q.dim() == 1:
        q = q.unsqueeze(0)
    num = q @ s.T
    qn = q.norm(dim=1, keepdim=True)
    sn = s.norm(dim=1)
    cos = num / (qn * sn).clamp(min=1e-8)
    return cos.max(dim=0).values


class annotation_obj:
    def __init__(self, key):
        self.key = key
        self.len = 0
        self.values = {"imgs": [], "text": []}
        self.mean = 0

    def add_value(self, value, t, embedder):
        if t == "text":
            self.values["text"].append(value)
        if t == "image" or t == "video":
            encoding = embedder.get_image_feat(value)
            if encoding.dim() > 1:
                encoding = encoding.squeeze(0)
            self.values["imgs"].append(encoding.cpu())

    def save(self, out):
        with open(out, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)


class PersistentSearch:
    def __init__(self, video_folder=None, embedding_folder=None):
        print("Startup...", flush=True)
        self.embedder = CosmosEmbedder()
        _clipseek_ui(
            "model_loading",
            "ClipSeek: Loading Cosmos embedding model…",
        )
        self.embedder.load()
        _clipseek_ui(
            "model_ready",
            "ClipSeek: Embedding model loaded.",
        )
        if not _FAISS_AVAILABLE:
            _clipseek_ui(
                "faiss_unavailable",
                "ClipSeek: FAISS not installed — FAISS mode falls back to exact search. pip install faiss-cpu",
            )
        self.device = torch.device(self.embedder.device)
        self.cached_objects = None
        self.current_embedding_folder = embedding_folder
        self.current_video_folder = video_folder
        self._faiss_index = None
        self._faiss_index_embedding_path = None
        self._faiss_rep_video_ids = None
        self._last_faiss_reason = ""
        self._last_faiss_candidate_count = 0
        self._last_faiss_total_count = 0
        if embedding_folder:
            print("loading startup embedding folder", flush=True)
            self.cached_objects = self.load_objs(video_folder, embedding_folder)
        else:
            _clipseek_ui(
                "embeddings_pending",
                "ClipSeek: No embedding folder at startup — choose one in Settings when ready.",
            )
        self._emit_search_ready_and_ready_line()

    def _emit_search_ready_and_ready_line(self) -> None:
        n = len(self.cached_objects) if self.cached_objects else 0
        if n:
            msg = f"ClipSeek: Cache loaded — ready to search ({n} videos)."
        else:
            msg = (
                "ClipSeek: Ready — select an embedding folder in Settings to load your library."
            )
        _clipseek_ui("search_ready", msg, videos=n)
        print("READY", flush=True)

    def update_embedding_folder(self, video_folder, new_folder):
        # Bridge re-sends the same folder right after boot; avoid duplicate reload/alerts.
        if (
            new_folder
            and new_folder == self.current_embedding_folder
            and self.cached_objects is not None
        ):
            return
        self._clear_faiss_cache()
        # Drop stale cached objects BEFORE updating current_embedding_folder, otherwise the
        # cache-hit short-circuit in load_objs (which keys off current_embedding_folder) returns
        # the previous folder's objects and the panel keeps searching the old library.
        self.cached_objects = None
        self.current_embedding_folder = new_folder
        self.current_video_folder = video_folder
        if not new_folder:
            self.cached_objects = []
            _clipseek_ui(
                "embeddings_cleared",
                "ClipSeek: Embedding folder cleared.",
                videos=0,
            )
            return
        _clipseek_ui(
            "cache_loading",
            f"ClipSeek: Loading embeddings from {new_folder}…",
        )
        try:
            self.cached_objects = self.load_objs(video_folder, new_folder)
        except Exception as e:
            self.cached_objects = []
            _clipseek_error(
                f"ClipSeek: Failed to load embedding folder: {e}",
                exc=e,
                phase="embeddings_error",
            )
            return
        n = len(self.cached_objects) if self.cached_objects else 0
        _clipseek_ui(
            "embeddings_reloaded",
            f"ClipSeek: Embedding folder reloaded — ready to search ({n} videos)."
            if n
            else "ClipSeek: Embedding folder has no indexed videos yet.",
            videos=n,
        )

    def _clear_faiss_cache(self):
        self._faiss_index = None
        self._faiss_index_embedding_path = None
        self._faiss_rep_video_ids = None

    def _reset_faiss_status(self):
        self._last_faiss_reason = ""
        self._last_faiss_candidate_count = 0
        self._last_faiss_total_count = 0

    def _faiss_skip(self, reason: str):
        self._last_faiss_reason = reason
        self._last_faiss_candidate_count = 0
        self._last_faiss_total_count = 0
        print(f"FAISS skipped: {reason}", flush=True)
        return None

    def _faiss_precheck_reason(
        self,
        matrix_shared,
        embeddings_path: str,
        vid_objs,
        has_query: bool,
        has_anno: bool,
    ) -> str:
        if not _FAISS_AVAILABLE:
            return "faiss-cpu is not installed"
        if not embeddings_path:
            return "no embedding folder is loaded"
        if matrix_shared is None:
            return "FAISS needs the mmap embedding cache; regenerate or reload the embedding cache"
        if not has_query:
            return "annotation-only queries use exact search"
        if has_anno:
            return "annotation-assisted queries use exact search"
        if not all(hasattr(o, "_corpus_video_index") for o in vid_objs):
            return "the loaded embeddings do not expose FAISS corpus metadata"
        return ""

    def create_annotation(self, folder, key, annotation_type, value):
        if not os.path.exists(folder):
            print("NO ANNOTATION FOLDER FOUND", flush=True)
            return
        if os.path.exists(os.path.join(folder, f"{key}.anno")):
            print(f"adding {value} to {key} annotation", flush=True)
            with open(os.path.join(folder, f"{key}.anno"), "rb") as f:
                current = pickle.load(f)
        else:
            print(f"creating new annotation: {key}", flush=True)
            current = annotation_obj(key)

        if isinstance(value, list):
            for img_path in value:
                current.add_value(img_path, annotation_type, self.embedder)
        else:
            current.add_value(value, annotation_type, self.embedder)
        if len(current.values["imgs"]) > 0:
            current.mean = torch.stack(current.values["imgs"]).mean(dim=0)
        current.len += 1
        current.save(f"{folder}\\{key}.anno")

        print(f"Annotation created: {json.dumps(f'{key}, {annotation_type}, {value}')}", flush=True)

    def search_file(
        self,
        file_path,
        query_type,
        video_folder,
        embedding_folder,
        annotation_folder,
        search_mode: str = "exact",
        is_mean=True,
    ):
        try:
            t_wall = time.time()
            if isinstance(is_mean, str):
                is_mean = is_mean.lower() in ("true", "1", "yes")
            if search_mode not in ("exact", "faiss"):
                search_mode = "exact"
            faiss_requested = search_mode == "faiss"

            short_name = os.path.basename(file_path) if file_path else "(file)"
            _clipseek_ui(
                "search_start",
                f"ClipSeek: Searching by {query_type} — {short_name}…",
            )

            t_load0 = time.time()
            vid_objs = self.load_objs(video_folder, embedding_folder)
            load_seconds = time.time() - t_load0

            if not vid_objs:
                _clipseek_ui(
                    "search_empty",
                    "ClipSeek: No embeddings loaded — choose an embedding folder in Settings.",
                )
                return {
                    "results": [],
                    "search_seconds": time.time() - t_wall,
                    "load_seconds": load_seconds,
                    "retrieve_seconds": 0.0,
                    "faiss_used": False,
                    "faiss_requested": faiss_requested,
                    "faiss_available": _FAISS_AVAILABLE,
                    "faiss_reason": "no embeddings are loaded" if faiss_requested else "",
                    "faiss_candidates": 0,
                    "faiss_total_candidates": 0,
                }

            _clipseek_ui("search_encode", f"ClipSeek: Encoding {query_type} query…")
            if query_type == "image":
                q = self.embedder.get_image_feat(file_path).to(self.device)
            elif query_type == "video":
                q = self.embedder.get_vid_feat_tensor(fix_path(file_path)).to(self.device)
            else:
                _clipseek_error(f"ClipSeek: Unsupported query type: {query_type}")
                return {"error": "Unsupported query type", "results": []}

            _clipseek_ui(
                "search_compute",
                f"ClipSeek: Comparing against {len(vid_objs)} videos ({search_mode})…",
            )

            emb_path = embedding_folder or self.current_embedding_folder or ""
            matrix_shared = _vid_objs_shared_corpus_matrix(vid_objs)
            faiss_used = False
            faiss_reason = ""
            t_sim0 = time.time()
            final_sims = None

            if faiss_requested:
                self._reset_faiss_status()
                faiss_reason = self._faiss_precheck_reason(
                    matrix_shared,
                    emb_path,
                    vid_objs,
                    True,
                    False,
                )
                if not faiss_reason:
                    fr = self._try_faiss_rerank_videos(
                        q,
                        vid_objs,
                        matrix_shared,
                        is_mean,
                        False,
                        True,
                        emb_path,
                    )
                    faiss_reason = self._last_faiss_reason
                    if fr is not None:
                        final_sims = fr
                        faiss_used = True

            if final_sims is None:
                with torch.no_grad():
                    if matrix_shared is not None:
                        g_idx, split_sizes = _gather_chunk_indices_and_split_sizes(vid_objs)
                        sims_1d = self._chunkwise_cosine_max_chunk_sims(q, matrix_shared, g_idx)
                    else:
                        all_chunks = [
                            chunks_to_tensor(vid_obj.chunks, self.device) for vid_obj in vid_objs
                        ]
                        stacked_chunks = torch.cat(all_chunks, dim=0)
                        sims_1d = cosine_query_to_chunks(q, stacked_chunks)
                        split_sizes = [t.shape[0] for t in all_chunks]

                    sims_split_list = torch.split(sims_1d, split_sizes, dim=0)
                for i, vo in enumerate(vid_objs):
                    vo.sims = sims_split_list[i]
                final_sims = torch.stack(
                    [
                        torch.mean(sim) if is_mean else torch.max(sim)
                        for sim in sims_split_list
                    ]
                )

            retrieve_seconds = time.time() - t_sim0

            top_values, top_indices = torch.topk(
                final_sims, k=min(len(vid_objs), 500), largest=True, sorted=True
            )
            out = []
            for idx in top_indices.tolist():
                vid = vid_objs[idx]
                si = vid.sims
                t_sec = torch.argmax(si).item() * chunk_stride_sec(vid)
                out.append([vid.path, t_sec])

            total = time.time() - t_wall
            _clipseek_ui(
                "search_done",
                f"ClipSeek: Done — {len(out)} results in {total:.2f}s.",
                results=len(out),
                seconds=total,
            )
            return {
                "results": out,
                "search_seconds": total,
                "load_seconds": load_seconds,
                "retrieve_seconds": retrieve_seconds,
                "faiss_used": faiss_used,
                "faiss_requested": faiss_requested,
                "faiss_available": _FAISS_AVAILABLE,
                "faiss_reason": faiss_reason,
                "faiss_candidates": self._last_faiss_candidate_count if faiss_used else 0,
                "faiss_total_candidates": self._last_faiss_total_count if faiss_used else 0,
            }

        except Exception as e:
            tb = traceback.format_exc()
            print(f"error with image search {e}\n{tb}", flush=True)
            _clipseek_error(f"ClipSeek: Search failed — {e}", exc=e, phase="search_error")
            return {"error": str(e), "traceback": tb, "results": []}

    def filter_metadata(self, vid_objs, date_from=None, date_to=None, embeddings_folder=None):
        if not vid_objs:
            return []

        filtered_objects = []
        date_from = datetime.strptime(date_from, "%Y-%m-%d") if date_from else None
        date_to = datetime.strptime(date_to, "%Y-%m-%d") if date_to else None

        for obj in vid_objs:
            try:
                if not hasattr(obj, "datetime"):
                    print(
                        f"Object {obj.path} missing datetime: please re-encode/fix files",
                        flush=True,
                    )
                    continue

                if (not date_from or obj.datetime >= date_from) and (
                    not date_to or obj.datetime <= date_to
                ):
                    filtered_objects.append(obj)

            except Exception as e:
                print(f"Error processing object {e}", flush=True)
                continue

        print(f"Filtered {len(vid_objs)} videos down to {len(filtered_objects)}", flush=True)
        return filtered_objects

    def _chunkwise_cosine_max_chunk_sims(self, query_feat, matrix, gather_idx):
        """
        Cosine similarity per corpus row vs multi-row ``query_feat`` (max over query rows),
        same math as ``cosine_query_to_chunks``. Batched along mmap rows to limit GPU memory.
        """
        if gather_idx.size == 0:
            return torch.zeros(0, device=self.device)
        q = query_feat.float()
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = F.normalize(q, dim=1)
        N = int(gather_idx.shape[0])
        out = torch.empty(N, device=self.device, dtype=torch.float32)
        bs = SEARCH_GATHER_CHUNK_ROWS
        for s in range(0, N, bs):
            e = min(s + bs, N)
            idx = gather_idx[s:e]
            cpu = np.ascontiguousarray(matrix[idx].astype(np.float32, copy=False))
            block = torch.from_numpy(cpu).to(self.device, non_blocking=True)
            block = F.normalize(block, dim=1)
            part = (q @ block.T).max(dim=0).values
            out[s:e] = part
            del block, cpu, part
        return out

    def load_objs(self, video_folder, embeddings_path, chunk_size=1000, save_interval=50000):
        prev_emb = getattr(self, "current_embedding_folder", None)
        if prev_emb is not None and prev_emb != embeddings_path:
            self._clear_faiss_cache()

        if (
            embeddings_path == self.current_embedding_folder
            and self.cached_objects is not None
        ):
            print("Using pre-cached objs", flush=True)
            return self.cached_objects

        pkl_mtime = max_per_video_embedding_mtime(embeddings_path)

        if v2_bundle_exists(embeddings_path) and v2_is_current(embeddings_path, pkl_mtime):
            try:
                _clipseek_ui(
                    "cache_loading",
                    "ClipSeek: Loading mmap embedding cache (cached_embeddings.matrix.npy)…",
                )
                t0 = time.time()
                cached_objects = load_v2_objects(embeddings_path)
                dt = time.time() - t0
                print(
                    f"Using mmap cache ({len(cached_objects)} videos) in {dt:.2f}s.",
                    flush=True,
                )
                self.cached_objects = cached_objects
                self.current_video_folder = video_folder
                self.current_embedding_folder = embeddings_path
                return cached_objects
            except Exception as e:
                print(f"Mmap cache invalid ({e}); rebuilding from per-video .pkls.", flush=True)
                _clipseek_ui(
                    "cache_rebuild",
                    "ClipSeek: Mmap cache unavailable; loading per-video .pkl files instead…",
                )

        disk_pkls = []
        try:
            for f in os.scandir(embeddings_path):
                if f.name.endswith(".pkl") and f.name != "cached_embeddings.pkl":
                    disk_pkls.append(f.path)
        except OSError as e:
            print(f"Cannot read embedding folder: {e}", flush=True)
            _clipseek_ui(
                "cache_error",
                f"ClipSeek: Cannot read embedding folder: {e}",
            )
            self.cached_objects = []
            return []

        n_pkls = len(disk_pkls)
        if n_pkls == 0:
            _clipseek_ui(
                "cache_empty",
                "ClipSeek: No per-video .pkl files in this folder.",
                videos=0,
            )
            self.cached_objects = []
            return []

        _clipseek_ui(
            "cache_loading",
            f"ClipSeek: Loading {n_pkls} per-video embedding file(s) — building cache may take a while…",
            total=n_pkls,
        )
        print(f"Loading {n_pkls} embedding file(s) from disk.", flush=True)

        def load_one(p):
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
                print(f"Error loading {p}: {e}", flush=True)
                return None

        all_objects = []
        report_step = max(500, n_pkls // 25)
        last_reported = 0
        for i in range(0, len(disk_pkls), chunk_size):
            chunk = disk_pkls[i : i + chunk_size]
            with ThreadPoolExecutor(max_workers=8) as executor:
                loaded = list(filter(None, executor.map(load_one, chunk)))
            all_objects.extend(loaded)
            gc.collect()
            processed = min(i + len(chunk), n_pkls)
            if processed == n_pkls or processed - last_reported >= report_step:
                _clipseek_ui(
                    "cache_progress",
                    f"ClipSeek: Loading embeddings {processed} / {n_pkls}…",
                    loaded=processed,
                    total=n_pkls,
                )
                last_reported = processed

        by_path = {}
        for obj in all_objects:
            key = fix_path(getattr(obj, "path", ""))
            by_path[key] = obj
        merged = list(by_path.values())

        try:
            save_v2_from_objects(merged, embeddings_path)
            print("Saved mmap embedding cache (cached_embeddings.matrix.npy + .meta).", flush=True)
        except Exception as e:
            print(f"Error saving mmap cache: {e}", flush=True)
            _clipseek_ui(
                "cache_save_error",
                f"ClipSeek: Could not save mmap cache: {e}",
            )

        self.cached_objects = merged
        self.current_video_folder = video_folder
        self.current_embedding_folder = embeddings_path
        return merged

    def sigmoid(self, x: list, r=5, c=0.5):
        return [0 if i == 0 else 1 / (1 + math.exp(-r * (i - c))) for i in x]

    def _faiss_source_objects(self, embeddings_path: str, vid_objs):
        source_objs = None
        if (
            self.cached_objects is not None
            and embeddings_path == self.current_embedding_folder
            and _vid_objs_shared_corpus_matrix(self.cached_objects) is not None
        ):
            source_objs = self.cached_objects
        else:
            source_objs = vid_objs

        if not source_objs:
            return [], None
        matrix = _vid_objs_shared_corpus_matrix(source_objs)
        if matrix is None:
            return [], None
        required = ("_corpus_video_index", "_corpus_row_start", "_corpus_n_chunks")
        if not all(all(hasattr(o, attr) for attr in required) for o in source_objs):
            return [], None
        return source_objs, matrix

    def _ensure_faiss_index(self, embeddings_path: str, vid_objs):
        if not _FAISS_AVAILABLE:
            return None
        if self._faiss_index is not None and self._faiss_index_embedding_path == embeddings_path:
            return self._faiss_index

        source_objs, matrix = self._faiss_source_objects(embeddings_path, vid_objs)
        if matrix is None or not source_objs:
            return None

        N = int(matrix.shape[0])
        D = int(matrix.shape[1])
        if N == 0 or D == 0:
            return None

        rep_sources = []
        for o in source_objs:
            nc = int(getattr(o, "_corpus_n_chunks"))
            if nc <= 0:
                continue
            rep_sources.append(
                (
                    int(getattr(o, "_corpus_video_index")),
                    int(getattr(o, "_corpus_row_start")),
                    nc,
                )
            )
        if not rep_sources:
            return None

        _clipseek_ui(
            "faiss_building",
            f"ClipSeek: Preparing FAISS candidate index ({len(rep_sources)} videos)...",
            videos=len(rep_sources),
        )
        print(
            f"Building FAISS video candidate index ({len(rep_sources)} videos, dim {D})...",
            flush=True,
        )
        t0 = time.time()

        try:
            index = faiss.IndexFlatIP(D)
            rep_video_ids = np.empty(len(rep_sources), dtype=np.int32)
            batch_size = min(FAISS_ADD_BATCH, len(rep_sources))
            reps_batch = np.empty((batch_size, D), dtype=np.float32)
            batch_count = 0
            report_step = max(1000, len(rep_sources) // 10)
            last_reported = 0

            for i, (corpus_video_index, row_start, n_chunks) in enumerate(rep_sources, start=1):
                arr = np.asarray(
                    matrix[row_start : row_start + n_chunks],
                    dtype=np.float32,
                )
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr_norm = arr / np.maximum(norms, 1e-8)
                reps_batch[batch_count] = arr_norm.mean(axis=0)
                rep_video_ids[i - 1] = corpus_video_index
                batch_count += 1
                del arr, norms, arr_norm
                if batch_count == batch_size:
                    index.add(np.ascontiguousarray(reps_batch[:batch_count]))
                    batch_count = 0

                if i == len(rep_sources) or i - last_reported >= report_step:
                    _clipseek_ui(
                        "faiss_progress",
                        f"ClipSeek: Preparing FAISS candidates {i} / {len(rep_sources)}...",
                        loaded=i,
                        total=len(rep_sources),
                    )
                    last_reported = i

            if batch_count:
                index.add(np.ascontiguousarray(reps_batch[:batch_count]))
            del reps_batch
            gc.collect()

            self._faiss_index = index
            self._faiss_index_embedding_path = embeddings_path
            self._faiss_rep_video_ids = rep_video_ids
            print(f"FAISS candidate index ready in {time.time() - t0:.2f}s.", flush=True)
            _clipseek_ui(
                "faiss_ready",
                f"ClipSeek: FAISS candidate index ready in {time.time() - t0:.2f}s.",
                videos=len(rep_sources),
            )
            return index
        except Exception as e:
            self._clear_faiss_cache()
            print(f"FAISS index build failed: {e}", flush=True)
            _clipseek_error(
                f"ClipSeek: FAISS index failed; using exact search: {e}",
                exc=e,
                phase="faiss_error",
            )
            return None

    def _try_faiss_rerank_videos(
        self,
        query_feat,
        vid_objs,
        matrix_shared,
        isMean,
        has_anno,
        has_query,
        embeddings_path: str,
    ):
        """Approximate ANN probe + exact batched rerank on candidate videos. Returns tensor or None."""
        self._reset_faiss_status()
        if not _FAISS_AVAILABLE:
            return self._faiss_skip("faiss-cpu is not installed")
        if has_anno:
            return self._faiss_skip("annotation-assisted queries use exact search")
        if not has_query:
            return self._faiss_skip("annotation-only queries use exact search")
        if not all(hasattr(o, "_corpus_video_index") for o in vid_objs):
            return self._faiss_skip("the loaded embeddings do not expose FAISS corpus metadata")
        allowed = {int(getattr(o, "_corpus_video_index")) for o in vid_objs}
        if not allowed:
            return self._faiss_skip("no candidate videos are available after filters")
        if len(allowed) <= FAISS_MIN_RERANK_VIDEOS:
            return self._faiss_skip(
                f"only {len(allowed)} candidate videos; exact search is faster at this size"
            )
        target_videos = max(
            FAISS_MIN_RERANK_VIDEOS,
            int(math.ceil(len(allowed) * FAISS_RERANK_FRACTION)),
        )
        target_videos = min(FAISS_MAX_RERANK_VIDEOS, target_videos, len(allowed) - 1)
        if target_videos <= 0:
            return self._faiss_skip("candidate set is too small for FAISS to narrow")

        idx = self._ensure_faiss_index(embeddings_path, vid_objs)
        rep_video_ids = self._faiss_rep_video_ids
        if idx is None or rep_video_ids is None:
            return self._faiss_skip("could not prepare the FAISS candidate index")

        N = int(idx.ntotal)
        q = query_feat.float()
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = F.normalize(q, dim=1)
        probe = F.normalize(q.mean(dim=0, keepdim=True), dim=1)
        probe_np = probe.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(probe_np)

        vid_hit = {}
        k_search = min(
            N,
            max(FAISS_MIN_PROBE_REPS, target_videos * FAISS_PROBE_REP_MULT),
        )
        while True:
            dists, idxs = idx.search(probe_np, k_search)
            for rep_i, di in zip(idxs[0], dists[0]):
                if rep_i < 0 or rep_i >= rep_video_ids.shape[0]:
                    continue
                cv = int(rep_video_ids[rep_i])
                if cv not in allowed:
                    continue
                vid_hit[cv] = max(vid_hit.get(cv, -1e9), float(di))
            if len(vid_hit) >= target_videos or k_search >= N:
                break
            k_search = min(N, k_search * 2)

        if not vid_hit:
            return self._faiss_skip("FAISS returned no candidate videos")

        ranked = sorted(vid_hit.keys(), key=lambda v: vid_hit[v], reverse=True)[:target_videos]
        top_allowed = set(ranked)
        narrow_objs = [
            o for o in vid_objs if int(getattr(o, "_corpus_video_index")) in top_allowed
        ]
        if not narrow_objs:
            return self._faiss_skip("FAISS candidates were removed by the active filters")
        gather_n, split_n = _gather_chunk_indices_and_split_sizes(narrow_objs)
        with torch.no_grad():
            qn = query_feat
            if qn.dim() == 1:
                qn = qn.unsqueeze(0)
            query_sims_1d = self._chunkwise_cosine_max_chunk_sims(qn, matrix_shared, gather_n)
            sims_split = torch.split(query_sims_1d, split_n, dim=0)
            final_narrow = torch.stack(
                [torch.mean(sim) if isMean else torch.max(sim) for sim in sims_split]
            )
        corpus_to_i = {
            int(getattr(vid_objs[i], "_corpus_video_index")): i
            for i in range(len(vid_objs))
            if hasattr(vid_objs[i], "_corpus_video_index")
        }
        full = torch.full((len(vid_objs),), float("-inf"), device=self.device)
        for j, o in enumerate(narrow_objs):
            ci = int(getattr(o, "_corpus_video_index"))
            li = corpus_to_i.get(ci)
            if li is None:
                continue
            full[li] = final_narrow[j]
            vid_objs[li].sims = sims_split[j]
        for i in range(len(vid_objs)):
            if torch.isinf(full[i]):
                vid_objs[i].sims = torch.tensor([0.0], device=self.device)
        if not torch.isfinite(full).any():
            return self._faiss_skip("FAISS rerank produced no finite scores")
        self._last_faiss_candidate_count = len(narrow_objs)
        self._last_faiss_total_count = len(allowed)
        self._last_faiss_reason = (
            f"selected {len(narrow_objs)} of {len(allowed)} videos for exact rerank"
        )
        return full

    def retrieve_vids(
        self,
        query,
        annotation_folder,
        vid_objs,
        isMean,
        query_type,
        search_mode: str = "exact",
    ):
        has_anno = False
        has_query = True
        annotations_vector = []
        query_feat = None

        if query_type == "text":
            query = query.lower()
            annotation_list = []
            if os.path.exists(annotation_folder):
                annotations = [i for i in os.listdir(annotation_folder) if i.endswith(".anno")]
                annotation_list = [i for i in annotations if i[:-5] in query]
            if len(annotation_list) > 0:
                query = [i for i in query.split(annotation_list[0][:-5]) if i != ""]
                has_anno = True
                print("adding annos", annotation_list, flush=True)

            for anno in annotation_list:
                with open(os.path.join(annotation_folder, anno), "rb") as a:
                    anno_obj = pickle.load(a)
                annotations_vector.append(anno_obj.mean.to(self.device))

            if len(query) > 0:
                text_feat = self.embedder.get_text_feat(query).to(self.device)
                query_feat = text_feat.squeeze(0) if text_feat.dim() > 1 else text_feat
            else:
                has_query = False
        elif query_type == "video":
            query_path = fix_path(query)
            match = next(
                (o for o in vid_objs if fix_path(o.path) == query_path),
                None,
            )
            if match is not None:
                query_feat = chunks_to_tensor(match.chunks, self.device)
            else:
                query_feat = self.embedder.get_vid_feat_tensor(query_path).to(self.device)

        if not vid_objs:
            return torch.tensor([], device=self.device), {
                "faiss_used": False,
                "faiss_requested": search_mode == "faiss",
                "faiss_available": _FAISS_AVAILABLE,
                "retrieve_seconds": 0.0,
                "faiss_reason": "no embeddings are loaded" if search_mode == "faiss" else "",
                "faiss_candidates": 0,
                "faiss_total_candidates": 0,
            }

        start = time.time()

        emb_path = getattr(self, "current_embedding_folder", None) or ""

        matrix_shared = _vid_objs_shared_corpus_matrix(vid_objs)
        if matrix_shared is not None:
            gather_idx, split_sizes_fs = _gather_chunk_indices_and_split_sizes(vid_objs)
        else:
            gather_idx = None
            split_sizes_fs = None

        faiss_reason = ""
        if search_mode == "faiss":
            self._reset_faiss_status()
            faiss_reason = self._faiss_precheck_reason(
                matrix_shared,
                emb_path,
                vid_objs,
                has_query,
                has_anno,
            )
            if not faiss_reason:
                fr = self._try_faiss_rerank_videos(
                    query_feat,
                    vid_objs,
                    matrix_shared,
                    isMean,
                    has_anno,
                    has_query,
                    emb_path,
                )
                faiss_reason = self._last_faiss_reason
                if fr is not None:
                    elapsed = time.time() - start
                    print("Search (cos sim) in: ", elapsed, flush=True)
                    return fr, {
                        "faiss_used": True,
                        "faiss_requested": True,
                        "faiss_available": True,
                        "retrieve_seconds": elapsed,
                        "faiss_reason": faiss_reason,
                        "faiss_candidates": self._last_faiss_candidate_count,
                        "faiss_total_candidates": self._last_faiss_total_count,
                    }

        sims_split = None
        anno_sims_split = None

        with torch.no_grad():
            if matrix_shared is not None:
                if has_query:
                    q = query_feat
                    if q.dim() == 1:
                        q = q.unsqueeze(0)
                    query_sims_1d = self._chunkwise_cosine_max_chunk_sims(
                        q, matrix_shared, gather_idx
                    )
                    sims_split = torch.split(query_sims_1d, split_sizes_fs, dim=0)
                if has_anno:
                    av = annotations_vector[0]
                    aq = av.unsqueeze(0) if av.dim() == 1 else av
                    anno_sims_1d = self._chunkwise_cosine_max_chunk_sims(
                        aq, matrix_shared, gather_idx
                    )
                    anno_sims_split = torch.split(anno_sims_1d, split_sizes_fs, dim=0)
            else:
                all_chunks = [
                    chunks_to_tensor(vid_obj.chunks, self.device) for vid_obj in vid_objs
                ]
                stacked_chunks = (
                    torch.cat(all_chunks, dim=0) if all_chunks else torch.zeros(0, device=self.device)
                )
                split_sizes = [t.shape[0] for t in all_chunks]
                if has_query:
                    q = query_feat
                    if q.dim() == 1:
                        q = q.unsqueeze(0)
                    query_sims_1d = cosine_query_to_chunks(q, stacked_chunks)
                    sims_split = torch.split(query_sims_1d, split_sizes, dim=0)
                if has_anno:
                    av = annotations_vector[0]
                    aq = av.unsqueeze(0) if av.dim() == 1 else av
                    anno_sims_1d = cosine_query_to_chunks(aq, stacked_chunks)
                    anno_sims_split = torch.split(anno_sims_1d, split_sizes, dim=0)

            if has_query and sims_split is not None:
                final_sims = torch.stack(
                    [torch.mean(sim) if isMean else torch.max(sim) for sim in sims_split]
                )
            if has_anno and anno_sims_split is not None:
                final_anno_sims = torch.stack(
                    [torch.mean(sim) if isMean else torch.max(sim) for sim in anno_sims_split]
                )

        if has_anno and has_query:
            sims_tensor = final_sims * final_anno_sims
            for i in range(len(vid_objs)):
                vid_objs[i].sims = sims_split[i] * anno_sims_split[i]
        elif not has_anno and has_query:
            sims_tensor = final_sims
            for i in range(len(vid_objs)):
                vid_objs[i].sims = sims_split[i]
        elif has_anno and not has_query:
            sims_tensor = final_anno_sims
            for i in range(len(vid_objs)):
                vid_objs[i].sims = anno_sims_split[i]

        elapsed = time.time() - start
        print("Search (cos sim) in: ", elapsed, flush=True)
        return sims_tensor, {
            "faiss_used": False,
            "faiss_requested": search_mode == "faiss",
            "faiss_available": _FAISS_AVAILABLE,
            "retrieve_seconds": elapsed,
            "faiss_reason": faiss_reason,
            "faiss_candidates": 0,
            "faiss_total_candidates": 0,
        }

    def search_videos(
        self,
        video_folder,
        embedding_folder,
        annotation_folder,
        query,
        isMean,
        query_type,
        date_from=None,
        date_to=None,
        search_mode: str = "exact",
    ):
        try:
            t_wall = time.time()
            short_q = (str(query) if query is not None else "")[:80]
            _clipseek_ui(
                "search_start",
                f"ClipSeek: Searching for “{short_q}”…" if short_q else "ClipSeek: Searching…",
                query=short_q,
            )
            t_load = time.time()
            vid_objs = self.load_objs(video_folder, embedding_folder)
            load_seconds = time.time() - t_load
            print("Objs loaded in: ", load_seconds, flush=True)
            if date_from or date_to:
                _clipseek_ui("search_filter", "ClipSeek: Filtering by date…")
                vid_objs = self.filter_metadata(vid_objs, date_from, date_to, embedding_folder)
            if isinstance(isMean, str):
                isMean = isMean.lower() in ("true", "1", "yes")
            if search_mode not in ("exact", "faiss"):
                search_mode = "exact"
            if not vid_objs:
                _clipseek_ui(
                    "search_empty",
                    "ClipSeek: No embeddings loaded — choose an embedding folder in Settings.",
                )
                return {
                    "results": [],
                    "load_seconds": load_seconds,
                    "retrieve_seconds": 0.0,
                    "search_seconds": time.time() - t_wall,
                    "faiss_used": False,
                    "faiss_requested": search_mode == "faiss",
                    "faiss_available": _FAISS_AVAILABLE,
                    "faiss_reason": "no embeddings are loaded" if search_mode == "faiss" else "",
                    "faiss_candidates": 0,
                    "faiss_total_candidates": 0,
                }
            _clipseek_ui(
                "search_compute",
                f"ClipSeek: Comparing against {len(vid_objs)} videos ({search_mode}, {'mean' if isMean else 'max'})…",
                videos=len(vid_objs),
            )
            sims_tensor, retrieve_meta = self.retrieve_vids(
                query,
                annotation_folder,
                vid_objs,
                isMean,
                query_type,
                search_mode,
            )
            if sims_tensor.numel() == 0:
                _clipseek_ui("search_empty", "ClipSeek: No matching videos.")
                return {
                    "results": [],
                    "load_seconds": load_seconds,
                    "retrieve_seconds": retrieve_meta["retrieve_seconds"],
                    "search_seconds": time.time() - t_wall,
                    "faiss_used": retrieve_meta["faiss_used"],
                    "faiss_requested": retrieve_meta["faiss_requested"],
                    "faiss_available": retrieve_meta["faiss_available"],
                    "faiss_reason": retrieve_meta.get("faiss_reason", ""),
                    "faiss_candidates": retrieve_meta.get("faiss_candidates", 0),
                    "faiss_total_candidates": retrieve_meta.get("faiss_total_candidates", 0),
                }
            top_values, top_indices = torch.topk(
                sims_tensor, k=min(len(vid_objs), 500), largest=True, sorted=True
            )
            top_vids = [vid_objs[i] for i in top_indices]
            del vid_objs

            times = [
                [i.path, (torch.argmax(i.sims) * chunk_stride_sec(i)).item()]
                for i in top_vids
            ]
            total = time.time() - t_wall
            _clipseek_ui(
                "search_done",
                f"ClipSeek: Done — {len(times)} results in {total:.2f}s.",
                results=len(times),
                seconds=total,
            )
            return {
                "results": times,
                "load_seconds": load_seconds,
                "retrieve_seconds": retrieve_meta["retrieve_seconds"],
                "search_seconds": total,
                "faiss_used": retrieve_meta["faiss_used"],
                "faiss_requested": retrieve_meta["faiss_requested"],
                "faiss_available": retrieve_meta["faiss_available"],
                "faiss_reason": retrieve_meta.get("faiss_reason", ""),
                "faiss_candidates": retrieve_meta.get("faiss_candidates", 0),
                "faiss_total_candidates": retrieve_meta.get("faiss_total_candidates", 0),
            }

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error in search: {e}\n{tb}", flush=True)
            _clipseek_error(f"ClipSeek: Search failed — {e}", exc=e, phase="search_error")
            return {"error": str(e), "traceback": tb, "results": []}

    def process_commands(self):
        while True:
            try:
                line = input()
                if line.strip().lower() == "exit":
                    break

                try:
                    command = json.loads(line)
                    if command.get("command") == "create_annotation":
                        folder = command.get("annotation_folder")
                        key = command.get("key")
                        annotation_type = command.get("type")
                        value = command.get("value")
                        self.create_annotation(folder, key, annotation_type, value)
                    elif command.get("command") == "update_embedding_folder":
                        new_folder = command.get("embedding_folder")
                        video_folder = command.get("video_folder")
                        self.update_embedding_folder(video_folder, new_folder)
                    elif command.get("command") == "search_file":
                        file_path = command.get("file_path")
                        query_type = command.get("query_type")
                        video_folder = command.get("video_folder")
                        embedding_folder = command.get("embedding_folder")
                        annotation_folder = command.get("annotation_folder")

                        result = self.search_file(
                            file_path,
                            query_type,
                            video_folder,
                            embedding_folder,
                            annotation_folder,
                            command.get("search_mode", "exact"),
                            command.get("is_mean", True),
                        )
                        print(json.dumps(result), flush=True)
                    else:
                        date_from = command.get("date_from")
                        date_to = command.get("date_to")
                        sm = command.get("search_mode", "exact")
                        out = self.search_videos(
                            command["video_folder"],
                            command["embedding_folder"],
                            command["annotation_folder"],
                            command["query"],
                            command["is_mean"],
                            command["query_type"],
                            date_from,
                            date_to,
                            sm,
                        )
                        print(json.dumps(out), flush=True)
                except json.JSONDecodeError as e:
                    _clipseek_error(f"ClipSeek: Invalid JSON command: {e}", phase="command_error")
                    print(json.dumps({"error": "Invalid JSON"}), flush=True)
                except Exception as e:
                    tb = traceback.format_exc()
                    _clipseek_error(
                        f"ClipSeek: Command failed — {e}",
                        exc=e,
                        phase="command_error",
                    )
                    print(json.dumps({"error": str(e), "traceback": tb}), flush=True)

            except EOFError:
                break
            except Exception as e:
                tb = traceback.format_exc()
                _clipseek_error(f"ClipSeek: I/O error — {e}", exc=e, phase="io_error")
                print(json.dumps({"error": str(e), "traceback": tb}), flush=True)


if __name__ == "__main__":
    embedding_folder = sys.argv[1] if len(sys.argv) > 1 else None
    search = PersistentSearch(embedding_folder=embedding_folder)
    search.process_commands()
