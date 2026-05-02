"""
Cosmos-Embed1 inference for ClipSeek search (Premiere panel / io.exe).
Model resolution: `extension/cosmos_model`, then sibling `embed_app/cosmos_model`, else Hugging Face hub.
"""
import os
import numpy as np
import torch
import decord
from PIL import Image
from transformers import AutoModel

from clipseek_cosmos_processor import assert_video_processor, load_cosmos_processor


def _script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def resolve_cosmos_model_path():
    base = _script_dir()
    parent = os.path.dirname(base)
    candidates = [
        os.path.join(base, "cosmos_model"),
        os.path.join(parent, "embed_app", "cosmos_model"),
    ]
    for p in candidates:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            return p
    return "nvidia/Cosmos-Embed1-448p"


class CosmosEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32
        if self.device == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model_id = resolve_cosmos_model_path()
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return
        self.processor = load_cosmos_processor(self.model_id, trust_remote_code=True)
        assert_video_processor(self.processor)
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self._warm_text_encoder()

    def _warm_text_encoder(self):
        """Pay the first text-embedding/kernel setup cost before the first user search."""
        if self.processor is None or self.model is None:
            return
        try:
            text_inputs = self.processor(text=[""])
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            with self._autocast():
                with torch.inference_mode():
                    text_out = self.model.get_text_embeddings(**text_inputs)
            _ = text_out.text_proj.float()
            if self.device == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"ClipSeek: text warmup skipped: {e}", flush=True)

    def _autocast(self):
        return torch.amp.autocast(
            device_type="cuda",
            enabled=self.device == "cuda",
            dtype=self.dtype,
        )

    def get_text_feat(self, text):
        """Single L2-normalized text embedding (1, D). `text` is str or list of str fragments (joined)."""
        self.load()
        if isinstance(text, list):
            joined = " ".join(t.strip() for t in text if t and str(t).strip())
            captions = [joined] if joined else [""]
        else:
            captions = [text or ""]
        text_inputs = self.processor(text=captions)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with self._autocast():
            with torch.inference_mode():
                text_out = self.model.get_text_embeddings(**text_inputs)
        feat = text_out.text_proj.float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feat

    def get_image_feat(self, image_path):
        """L2-normalized image embedding as pseudo-video (1, D)."""
        self.load()
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        frames = np.stack([arr] * 8, axis=0)
        batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))
        video_inputs = self.processor(videos=batch)
        video_inputs = {k: v.to(self.device) for k, v in video_inputs.items()}
        with self._autocast():
            with torch.inference_mode():
                video_out = self.model.get_video_embeddings(**video_inputs)
        feat = video_out.visual_proj.float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feat

    def get_vid_feat_tensor(self, video_path, chunk_size=10.0, overlap=0.0):
        """
        Chunk embeddings as (N, D) float tensor, L2-normalized per row.
        Matches embed_app VideoEmbedder.get_vid_feat chunking (seconds-based).
        """
        self.load()
        try:
            vr = decord.VideoReader(video_path)
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            duration = total_frames / fps
        except Exception as e:
            raise RuntimeError(f"Failed to read video {video_path}: {e}") from e

        stride = chunk_size - overlap
        if stride <= 0:
            stride = chunk_size

        start_times = []
        t = 0.0
        while t < duration:
            start_times.append(t)
            t += stride

        rows = []
        for start_time in start_times:
            end_time = min(start_time + chunk_size, duration)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            if start_frame >= total_frames:
                break
            chunk_len = end_frame - start_frame
            if chunk_len < 1:
                continue
            actual_frames = min(8, chunk_len)
            frame_ids = np.linspace(
                start_frame, min(end_frame, total_frames - 1), actual_frames, dtype=int
            ).tolist()
            try:
                frames = vr.get_batch(frame_ids).asnumpy()
                if frames.shape[0] == 0:
                    continue
                batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))
                video_inputs = self.processor(videos=batch)
                video_inputs = {k: v.to(self.device) for k, v in video_inputs.items()}
                with self._autocast():
                    with torch.inference_mode():
                        video_out = self.model.get_video_embeddings(**video_inputs)
                chunk_emb = video_out.visual_proj.float().cpu()
                chunk_emb = chunk_emb / chunk_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                rows.append(chunk_emb)
            except Exception:
                continue

        del vr
        dim = getattr(self.model.config, "embed_dim", 768)
        if not rows:
            return torch.zeros(0, dim, dtype=torch.float32)
        return torch.cat(rows, dim=0).float()
