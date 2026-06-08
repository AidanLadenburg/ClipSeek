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
    def __init__(self, path, embedder, chunk_size=10, overlap=0, progress_callback=None):
        self.title = os.path.basename(path)
        self.path = fix_path(path)
        stride = chunk_size - overlap
        self.chunk_stride_sec = stride if stride > 0 else float(chunk_size)
        self.chunks = self.encode_chunks(path, embedder, chunk_size, overlap, progress_callback)
        self.datetime = datetime.fromtimestamp(os.stat(self.path).st_mtime)
        self.audio = None
        self.shot_type = []
        self.annotations = []
        self.transcript = ""
        self.sims = []
        self.proxy = ""

    def encode_chunks(self, path, embedder, chunk_size, overlap, progress_callback):
        return embedder.get_vid_feat(path, chunk_size=chunk_size, overlap=overlap, progress_callback=progress_callback)

    def save(self, out):
        with open(out, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)


class _ClipseekUnpickler(pickle.Unpickler):
    """Map legacy ClipSeek classes and NumPy 2 private pickle paths."""

    def find_class(self, module, name):
        if name in ("video_obj", "ClipseekVideo"):
            return ClipseekVideo
        modules = _compatible_numpy_pickle_modules(module)
        last_error = None
        for candidate in modules:
            try:
                return super().find_class(candidate, name)
            except (AttributeError, ImportError, ModuleNotFoundError) as e:
                last_error = e
                if len(modules) == 1:
                    raise
        if last_error is not None:
            raise last_error
        return super().find_class(module, name)


def _compatible_numpy_pickle_modules(module):
    if module == "numpy.core":
        return [module, "numpy._core"]
    elif module.startswith("numpy.core."):
        return [module, "numpy._core." + module[len("numpy.core.") :]]
    elif module == "numpy._core":
        return [module, "numpy.core"]
    elif module.startswith("numpy._core."):
        return [module, "numpy.core." + module[len("numpy._core.") :]]
    return [module]


def load_pickle_compat(file_obj):
    return _ClipseekUnpickler(file_obj).load()


def load_clipseek_video_pickle(file_path):
    with open(file_path, "rb") as f:
        return load_pickle_compat(f)
