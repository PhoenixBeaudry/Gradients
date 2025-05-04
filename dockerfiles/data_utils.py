# data_utils.py
"""
Generic dataset → (input_ids, labels) loader compatible with Axolotl‑style SFT.
Usage
-----
>>> from data_utils import load_and_tokenise_dataset
>>> train_ds, eval_ds = load_and_tokenise_dataset(cfg, tokenizer)
"""

from typing import Tuple, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from itertools import chain
# ──────────────────────────────────────────────────────────────────────────────
# Chat‑template helpers
# ──────────────────────────────────────────────────────────────────────────────
_LL3_START, _LL3_END, _EOT = "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
_CHATML_BEG, _CHATML_END   = "<|im_start|>", "<|im_end|>"

def tmpl_llama3(sys_msg, user_msg, assistant=""):
    return (f"{_LL3_START}system{_LL3_END}\n{sys_msg}{_EOT}\n"
            f"{_LL3_START}user{_LL3_END}\n{user_msg}{_EOT}\n"
            f"{_LL3_START}assistant{_LL3_END}\n{assistant}")

def tmpl_llama2(sys_msg, user_msg, assistant=""):
    sys_block = f"<<SYS>>\n{sys_msg}\n<</SYS>>\n\n" if sys_msg else ""
    return f"<s>[INST] {sys_block}{user_msg} [/INST] {assistant}"

def tmpl_qwen(sys_msg, user_msg, assistant=""):
    return (f"{_CHATML_BEG}system\n{sys_msg}{_CHATML_END}\n"
            f"{_CHATML_BEG}user\n{user_msg}{_CHATML_END}\n"
            f"{_CHATML_BEG}assistant\n{assistant}")

def tmpl_phi(sys_msg, user_msg, assistant=""):
    return f"<|system|>\n{sys_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n{assistant}"

def tmpl_plain(sys_msg, user_msg, assistant=""):
    return f"{user_msg}\n{assistant}"

_TEMPLATE_TABLE = {
    "llama‑3": tmpl_llama3, "llama3": tmpl_llama3,
    "llama‑2": tmpl_llama2, "llama2": tmpl_llama2,
    "mistral": tmpl_llama2, "mixtral": tmpl_llama2,
    "qwen": tmpl_qwen, "zephyr": tmpl_qwen,
    "phi": tmpl_phi,
}

def _select_template(model_name: str, explicit: str | None):
    key = (explicit or model_name).lower()
    for sub, fn in _TEMPLATE_TABLE.items():
        if sub in key:
            return fn
    return tmpl_plain

# ──────────────────────────────────────────────────────────────────────────────
# Schema detection
# ──────────────────────────────────────────────────────────────────────────────
def _detect_schema(example: Dict[str, Any]) -> str:
    e = {k.lower(): v for k, v in example.items()}
    k = set(e)
    if {"instruction", "output"} <= k:
        return "instruct"
    if {"question", "answer"} <= k:
        return "qa"
    if {"prompt", "completion"} <= k or {"prompt", "response"} <= k:
        return "prompt_completion"
    if {"chosen", "rejected"} <= k:
        return "dpo"
    return "plain"

# ──────────────────────────────────────────────────────────────────────────────
# Processor factory
# ──────────────────────────────────────────────────────────────────────────────
def _build_processor(cfg: dict, tokenizer):
    seq_len  = int(cfg.get("sequence_len", 2048))
    mask_inp = not cfg.get("train_on_inputs", False)
    tmpl_fn  = _select_template(cfg["base_model"], cfg.get("chat_template"))
    sys_msg  = cfg.get("system_prompt", "You are a helpful assistant.")

    def proc(ex: Dict[str, Any]):
        schema = _detect_schema(ex)

        # Prompt + answer extraction
        if schema == "instruct":
            prompt = ex["instruction"] + (" " + ex["input"] if ex.get("input") else "")
            answer = ex.get("output", "")
        elif schema == "qa":
            prompt, answer = ex["question"], ex["answer"]
        elif schema == "prompt_completion":
            prompt = ex["prompt"]; answer = ex.get("completion") or ex.get("response", "")
        elif schema == "dpo":
            prompt, answer = ex["prompt"], ex["chosen"]
        else:
            prompt, answer = "", str(ex[next(iter(ex))])

        # Chat wrap
        chat_prompt = tmpl_fn(sys_msg, prompt)
        full_text   = chat_prompt + answer + tokenizer.eos_token

        # Tokenise
        enc        = tokenizer(full_text, truncation=True,
                               max_length=seq_len, return_attention_mask=False)
        input_ids  = enc["input_ids"]
        labels     = input_ids.copy()

        # Mask prompt tokens
        if mask_inp:
            prompt_len = len(tokenizer(chat_prompt, add_special_tokens=False)["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": input_ids, "labels": labels}

    return proc

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def load_and_tokenise_dataset(cfg: dict, tokenizer) -> Tuple[Dataset, Dataset | None]:
    """
    Returns
    -------
    train_ds : datasets.Dataset  (columns: input_ids, labels)
    eval_ds  : same or None
    """
    spec   = cfg["datasets"][0]
    dtype  = spec.get("ds_type", spec.get("type", "hf")).lower()

    # Load raw dataset
    if dtype in {"json", "csv", "text"}:
        raw = load_dataset(dtype, data_files={"train": spec["path"]})["train"]
    else:
        raw = load_dataset(spec["path"], split=spec.get("split", "train"))

    # Train / eval split
    val_ratio = float(cfg.get("val_set_size", 0))
    if val_ratio:
        split   = raw.train_test_split(test_size=val_ratio, seed=42, shuffle=True)
        train, eval_ = split["train"], split["test"]
    else:
        train, eval_ = raw, None

    # Tokenise
    processor = _build_processor(cfg, tokenizer)
    train = train.map(processor, remove_columns=train.column_names)
    if eval_:
        eval_ = eval_.map(processor, remove_columns=eval_.column_names)

    return train, eval_


# All chat special tokens we inject anywhere in data_utils
_CHAT_TOKENS = [
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
    "<|im_start|>", "<|im_end|>",
    "<<SYS>>", "<</SYS>>", "[INST]", "[/INST]",
    "<|system|>", "<|user|>", "<|assistant|>"
]

def prepare_tokenizer(model_name: str, hub_token: str | None, cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        use_auth_token=hub_token,
        trust_remote_code=True,
    )

    # ---------- robust special‑token handling -------------------
    raw_values = tokenizer.special_tokens_map_extended.values()
    # each element may be str / int / list[str] / list[int]
    flat_tokens = chain.from_iterable(
        v if isinstance(v, (list, tuple, set)) else [v] for v in raw_values
    )
    special_set = set(flat_tokens)          # safe: only hashables
    # ------------------------------------------------------------

    # 1) force left‑padding
    tokenizer.padding_side = "left"

    # 2) default truncation
    if hasattr(tokenizer, "enable_truncation"):
        tokenizer.enable_truncation(max_length=int(cfg.get("sequence_len", 2048)))

    # 3) default padding
    if hasattr(tokenizer, "enable_padding"):
        tokenizer.enable_padding()

    # 4) fallback EOS→PAD
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


from typing import List
from transformers import PreTrainedTokenizerBase
import torch

class CausalLMDataCollator:
    """
    Pads input_ids & labels to the longest sequence in the batch.
    Labels are padded with –100, everything else with tokenizer.pad_token_id.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 pad_to_multiple_of: int | None = 8):
        self.tok = tokenizer
        self.mult = pad_to_multiple_of

    def __call__(self, features: List[dict]) -> dict:
        # We rely on tokenizer.pad for input_ids & attention_mask
        batch = self.tok.pad(
            features,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.mult,
        )

        # ---- labels ---------------------------------------------------------
        max_len = batch["input_ids"].size(1)
        padded_labels = []
        for f in features:
            lbl = f["labels"]
            pad = max_len - len(lbl)
            padded_labels.append(lbl + [-100] * pad)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

