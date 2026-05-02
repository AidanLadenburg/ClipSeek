"""
Cosmos-Embed1: load the real CosmosEmbed1Processor, not AutoTokenizer.

If `processor_config.json` has `processor_class` but no `auto_map`, Hugging Face
`AutoProcessor.from_pretrained` resolves the class name to None (Cosmos is not
in the built-in PROCESSOR_MAPPING), never reads `config.json`'s auto_map, and
falls back to `AutoTokenizer`. Then `processor(videos=...)` hits the tokenizer
with text=None and text_target=None → "You need to specify either `text` or
`text_target`."
"""
from transformers import AutoConfig, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def load_cosmos_processor(model_id: str, trust_remote_code: bool = True):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    am = getattr(cfg, "auto_map", None) or {}
    auto_proc = am.get("AutoProcessor")
    if auto_proc:
        proc_cls = get_class_from_dynamic_module(
            auto_proc, model_id, trust_remote_code=trust_remote_code
        )
        return proc_cls.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def assert_video_processor(proc):
    name = type(proc).__name__
    if "Tokenizer" in name and "Processor" not in name:
        raise TypeError(
            f"Loaded object is {name}, not a video processor. "
            "Fix cosmos_model/processor_config.json (add auto_map from config.json) or reinstall with download.py."
        )
