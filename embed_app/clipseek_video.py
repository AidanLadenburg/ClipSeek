"""
Shared on-disk embedding record for ClipSeek (embed GUI + search).
Pickle class path is stable so io.exe can load files written by the embedder app.
"""
import os
import pickle
from datetime import datetime


def fix_path(path):
    return os.path.abspath(path)


class ClipseekVideo:
    def __init__(
        self,
        path,
        embedder,
        chunk_size=10,
        overlap=0,
        progress_callback=None,
        gpu_index=0,
    ):
        self.title = os.path.basename(path)
        self.path = fix_path(path)
        stride = chunk_size - overlap
        self.chunk_stride_sec = stride if stride > 0 else float(chunk_size)
        self.chunks = self.encode_chunks(
            path, embedder, chunk_size, overlap, progress_callback, gpu_index
        )
        self.datetime = datetime.fromtimestamp(os.stat(self.path).st_mtime)
        self.audio = None
        self.shot_type = []
        self.annotations = []
        self.transcript = ""
        self.sims = []
        self.proxy = ""

    def encode_chunks(
        self, path, embedder, chunk_size, overlap, progress_callback, gpu_index=0
    ):
        return embedder.get_vid_feat(
            path,
            chunk_size=chunk_size,
            overlap=overlap,
            progress_callback=progress_callback,
            gpu_index=gpu_index,
        )

    def save(self, out):
        with open(out, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)


class _ClipseekUnpickler(pickle.Unpickler):
    """Map legacy `__main__.video_obj` (and similar) to ClipseekVideo."""

    def find_class(self, module, name):
        if name in ("video_obj", "ClipseekVideo"):
            return ClipseekVideo
        return super().find_class(module, name)


def load_clipseek_video_pickle(file_path):
    with open(file_path, "rb") as f:
        try:
            return pickle.load(f)
        except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError):
            f.seek(0)
            return _ClipseekUnpickler(f).load()
