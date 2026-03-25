import sys
import json
import pickle
import os
import torch
import time
import math
from datetime import datetime
import gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from cosmos_embedder import CosmosEmbedder
from clipseek_video import load_clipseek_video_pickle


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
    else:
        t = torch.as_tensor(chunks, dtype=torch.float32, device=device)
    return t


def chunk_stride_sec(obj):
    return float(getattr(obj, "chunk_stride_sec", 10.0))


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
        self.embedder.load()
        self.device = torch.device(self.embedder.device)
        self.cached_objects = None
        self.current_embedding_folder = embedding_folder
        self.current_video_folder = video_folder
        if embedding_folder:
            print("loading startup embedding folder", flush=True)
            self.cached_objects = self.load_objs(video_folder, embedding_folder)
        print("READY", flush=True)

    def update_embedding_folder(self, video_folder, new_folder):
        self.current_embedding_folder = new_folder
        self.current_video_folder = video_folder
        self.cached_objects = self.load_objs(video_folder, new_folder)
        print("Embedding folder updated and embeddings reloaded.", flush=True)

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

    def search_file(self, file_path, query_type, video_folder, embedding_folder, annotation_folder):
        try:
            vid_objs = self.load_objs(video_folder, embedding_folder)

            if query_type == "image":
                q = self.embedder.get_image_feat(file_path).to(self.device)
            elif query_type == "video":
                q = self.embedder.get_vid_feat_tensor(fix_path(file_path)).to(self.device)
            else:
                return {"error": "Unsupported query type"}

            all_chunks = [chunks_to_tensor(vid_obj.chunks, self.device) for vid_obj in vid_objs]
            stacked_chunks = torch.cat(all_chunks, dim=0)

            with torch.no_grad():
                sims_1d = cosine_query_to_chunks(q, stacked_chunks)

            split_sizes = [chunks_to_tensor(vid_obj.chunks, self.device).shape[0] for vid_obj in vid_objs]
            sims_split_list = torch.split(sims_1d, split_sizes, dim=0)
            final_sims = torch.stack([torch.mean(sim) for sim in sims_split_list])

            top_values, top_indices = torch.topk(
                final_sims, k=min(len(vid_objs), 500), largest=True, sorted=True
            )
            top_vids = [vid_objs[i] for i in top_indices]

            out = []
            for orig_idx, i in zip(top_indices.tolist(), top_vids):
                si = sims_split_list[orig_idx]
                t_sec = torch.argmax(si).item() * chunk_stride_sec(i)
                out.append((i.path, t_sec))

            return out

        except Exception as e:
            print(f"error with image search {e}", flush=True)
            return []

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

    def _max_pkl_mtime(self, embeddings_path):
        m = 0.0
        try:
            for f in os.scandir(embeddings_path):
                if not f.name.endswith(".pkl") or f.name == "cached_embeddings.pkl":
                    continue
                try:
                    m = max(m, f.stat().st_mtime)
                except OSError:
                    continue
        except OSError:
            pass
        return m

    def load_objs(self, video_folder, embeddings_path, chunk_size=1000, save_interval=50000):
        if (
            embeddings_path == self.current_embedding_folder
            and self.cached_objects is not None
        ):
            print("Using pre-cached objs", flush=True)
            return self.cached_objects

        cache_file = os.path.join(embeddings_path, "cached_embeddings.pkl")

        if os.path.exists(cache_file):
            try:
                pkl_mtime = self._max_pkl_mtime(embeddings_path)
                cache_mtime = os.path.getmtime(cache_file)
                if pkl_mtime <= cache_mtime:
                    with open(cache_file, "rb") as f:
                        cached_objects = pickle.load(f)
                    print(f"Using cache file ({len(cached_objects)} videos).", flush=True)
                    self.cached_objects = cached_objects
                    self.current_video_folder = video_folder
                    self.current_embedding_folder = embeddings_path
                    return cached_objects
            except Exception as e:
                print(f"Cache stale or invalid ({e}); rebuilding from .pkl files.", flush=True)

        disk_pkls = []
        try:
            for f in os.scandir(embeddings_path):
                if f.name.endswith(".pkl") and f.name != "cached_embeddings.pkl":
                    disk_pkls.append(f.path)
        except OSError as e:
            print(f"Cannot read embedding folder: {e}", flush=True)
            self.cached_objects = []
            return []

        print(f"Loading {len(disk_pkls)} embedding file(s) from disk.", flush=True)

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
        for i in range(0, len(disk_pkls), chunk_size):
            chunk = disk_pkls[i : i + chunk_size]
            with ThreadPoolExecutor(max_workers=8) as executor:
                loaded = list(filter(None, executor.map(load_one, chunk)))
            all_objects.extend(loaded)
            gc.collect()

        by_path = {}
        for obj in all_objects:
            key = fix_path(getattr(obj, "path", ""))
            by_path[key] = obj
        merged = list(by_path.values())

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(merged, f, pickle.HIGHEST_PROTOCOL)
            print("Saved embedding cache (cached_embeddings.pkl).", flush=True)
        except Exception as e:
            print(f"Error saving cache file: {e}", flush=True)

        self.cached_objects = merged
        self.current_video_folder = video_folder
        self.current_embedding_folder = embeddings_path
        return merged

    def sigmoid(self, x: list, r=5, c=0.5):
        return [0 if i == 0 else 1 / (1 + math.exp(-r * (i - c))) for i in x]

    def retrieve_vids(self, query, annotation_folder, vid_objs, isMean, query_type):
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

        start = time.time()

        all_chunks = [chunks_to_tensor(vid_obj.chunks, self.device) for vid_obj in vid_objs]
        stacked_chunks = torch.cat(all_chunks, dim=0)
        split_sizes = [chunks_to_tensor(vid_obj.chunks, self.device).shape[0] for vid_obj in vid_objs]

        sims_split = None
        anno_sims_split = None

        with torch.no_grad():
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

            if has_query:
                final_sims = torch.stack(
                    [torch.mean(sim) if isMean else torch.max(sim) for sim in sims_split]
                )
            if has_anno:
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

        print("Search (cos sim) in: ", time.time() - start, flush=True)
        return sims_tensor

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
    ):
        try:
            start = time.time()
            vid_objs = self.load_objs(video_folder, embedding_folder)
            print("Objs loaded in: ", time.time() - start, flush=True)
            if date_from or date_to:
                vid_objs = self.filter_metadata(vid_objs, date_from, date_to, embedding_folder)
            if isinstance(isMean, str):
                isMean = isMean.lower() in ("true", "1", "yes")
            sims_tensor = self.retrieve_vids(query, annotation_folder, vid_objs, isMean, query_type)
            top_values, top_indices = torch.topk(
                sims_tensor, k=min(len(vid_objs), 500), largest=True, sorted=True
            )
            top_vids = [vid_objs[i] for i in top_indices]
            del vid_objs

            return top_vids

        except Exception as e:
            print(f"Error in search: {e}", flush=True)
            return []

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
                        )
                        print(json.dumps(result), flush=True)
                    else:
                        date_from = command.get("date_from")
                        date_to = command.get("date_to")
                        vids = self.search_videos(
                            command["video_folder"],
                            command["embedding_folder"],
                            command["annotation_folder"],
                            command["query"],
                            command["is_mean"],
                            command["query_type"],
                            date_from,
                            date_to,
                        )
                        times = [
                            (i.path, (torch.argmax(i.sims) * chunk_stride_sec(i)).item())
                            for i in vids
                        ]
                        print(json.dumps(times), flush=True)
                except json.JSONDecodeError:
                    print(json.dumps({"error": "Invalid JSON"}), flush=True)
                except Exception as e:
                    print(json.dumps({"error": str(e)}), flush=True)

            except EOFError:
                break
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    embedding_folder = sys.argv[1] if len(sys.argv) > 1 else None
    search = PersistentSearch(embedding_folder=embedding_folder)
    search.process_commands()
