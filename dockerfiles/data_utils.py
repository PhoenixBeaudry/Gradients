# data_utils.py
"""
Self‑contained dataset → (input_ids, labels) pipeline that mirrors
Axolotl‑style SFT/DPO prompt construction, masking and tokenisation for *any*
HF dataset / local JSON‑CSV‑TXT file and for the major chat model families
(Llama‑3/2, Mistral/Mixtral, Qwen‑2, Zephyr/ChatML, Phi‑3, plain).

Public API
==========
>>> from data_utils import (
...     prepare_tokenizer,
...     load_and_tokenise_dataset,
...     ChatDataCollator,
... )

1.  **prepare_tokenizer** – add missing special tokens, pad token, pick
    padding side, mark if resize is needed (tokenizer._added_tokens flag).

2.  **load_and_tokenise_dataset** – auto‑detect schema, apply chat template,
    label masking, negative‑answer filtering for QA datasets, optional
    train/val split.

3.  **ChatDataCollator** – memory‑efficient collator that pads *only* to the
    longest sequence in the batch and keeps labels aligned.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
from itertools import chain

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# ────────────────────────────────────────────────────────────────────────────
# Chat‑template helpers
# ────────────────────────────────────────────────────────────────────────────
_LL3_START, _LL3_END, _EOT = (
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
)
_CHATML_BEG, _CHATML_END = "<|im_start|>", "<|im_end|>"


def _tmpl_llama3(sys_msg: str, user_msg: str, assistant: str = "") -> str:
    return (
        f"{_LL3_START}system{_LL3_END}\n{sys_msg}{_EOT}\n"
        f"{_LL3_START}user{_LL3_END}\n{user_msg}{_EOT}\n"
        f"{_LL3_START}assistant{_LL3_END}\n{assistant}"
    )


def _tmpl_llama2(sys_msg: str, user_msg: str, assistant: str = "") -> str:
    sys_block = f"<<SYS>>\n{sys_msg}\n<</SYS>>\n\n" if sys_msg else ""
    return f"<s>[INST] {sys_block}{user_msg} [/INST] {assistant}"


def _tmpl_qwen(sys_msg: str, user_msg: str, assistant: str = "") -> str:
    return (
        f"{_CHATML_BEG}system\n{sys_msg}{_CHATML_END}\n"
        f"{_CHATML_BEG}user\n{user_msg}{_CHATML_END}\n"
        f"{_CHATML_BEG}assistant\n{assistant}"
    )


def _tmpl_phi(sys_msg: str, user_msg: str, assistant: str = "") -> str:
    return f"<|system|>\n{sys_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n{assistant}"


def _tmpl_plain(sys_msg: str, user_msg: str, assistant: str = "") -> str:  # fallback
    return f"{user_msg}\n{assistant}"


_TEMPLATE_TABLE = {
    "llama-3": _tmpl_llama3,
    "llama3": _tmpl_llama3,
    "llama-2": _tmpl_llama2,
    "llama2": _tmpl_llama2,
    "mistral": _tmpl_llama2,
    "mixtral": _tmpl_llama2,
    "qwen": _tmpl_qwen,
    "zephyr": _tmpl_qwen,
    "phi": _tmpl_phi,
}


def _select_template(model_name: str, explicit: str | None) -> callable:
    """Pick chat template based on explicit cfg or substrings in model name."""
    key = (explicit or model_name).lower()
    for substr, fn in _TEMPLATE_TABLE.items():
        if substr in key:
            return fn
    return _tmpl_plain


# ────────────────────────────────────────────────────────────────────────────
# Schema detection helper
# ────────────────────────────────────────────────────────────────────────────

def _detect_schema(example: Dict[str, Any]) -> str:
    e = {k.lower(): v for k, v in example.items()}
    keys = set(e)
    if {"instruction", "output"} <= keys:
        return "instruct"
    if {"question", "answer"} <= keys:
        return "qa"
    if ("prompt" in keys and ("completion" in keys or "response" in keys)):
        return "prompt_completion"
    if {"chosen", "rejected"} <= keys:
        return "dpo"
    return "plain"


# ────────────────────────────────────────────────────────────────────────────
# Processor factory
# ────────────────────────────────────────────────────────────────────────────

def _build_processor(cfg: dict, tokenizer: PreTrainedTokenizerBase):
    seq_len = int(cfg.get("sequence_len", 2048))
    mask_inputs = not cfg.get("train_on_inputs", False)
    tmpl_fn = _select_template(cfg["base_model"], cfg.get("chat_template"))
    sys_msg = cfg.get("system_prompt", "You are a helpful assistant.")

    def proc(ex: Dict[str, Any]):  # returns dict or None (to be filtered)
        schema = _detect_schema(ex)

        # -------- 1. extract (prompt, answer) --------------------------------
        if schema == "instruct":
            prompt = ex["instruction"] + (" " + ex["input"] if ex.get("input") else "")
            answer = ex.get("output", "")
        elif schema == "qa":
            # Keep only positive QA pairs if label column present
            if "label" in ex and ex["label"] not in (1, True):
                return None  # drop negative
            prompt, answer = ex["question"], ex["answer"]
        elif schema == "prompt_completion":
            prompt = ex["prompt"]
            answer = ex.get("completion") or ex.get("response", "")
        elif schema == "dpo":
            prompt, answer = ex["prompt"], ex["chosen"]
        else:  # plain → treat first column as a single prompt, no supervised answer
            key = next(iter(ex))
            prompt, answer = "", str(ex[key])

        # -------- 2. wrap in chat template -----------------------------------
        chat_prompt = tmpl_fn(sys_msg, prompt)
        full_text = chat_prompt + answer + (tokenizer.eos_token or "")

        # -------- 3. tokenise -------------------------------------------------
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=seq_len,
            return_attention_mask=False,
        )
        input_ids = enc["input_ids"]
        labels = input_ids.copy()

        # -------- 4. mask prompt tokens --------------------------------------
        if mask_inputs:
            prompt_ids = tokenizer(
                chat_prompt,
                truncation=True,
                max_length=seq_len,
                add_special_tokens=False,
            )["input_ids"]
            keep = min(len(prompt_ids), len(labels))
            labels[:keep] = [-100] * keep

        return {"input_ids": input_ids, "labels": labels}

    return proc


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def load_and_tokenise_dataset(
    cfg: dict, tokenizer: PreTrainedTokenizerBase
) -> Tuple[Dataset, Dataset | None]:
    """Load dataset, apply processor, return (train, eval) HF Datasets."""

    spec = cfg["datasets"][0]
    dtype = spec.get("ds_type", spec.get("type", "hf")).lower()

    # ---------- 1. load raw --------------------------------------------------
    if dtype in {"json", "csv", "text"}:
        raw = load_dataset(dtype, data_files={"train": spec["path"]})["train"]
    else:
        raw = load_dataset(spec["path"], split=spec.get("split", "train"))

    # ---------- 2. optional val split ---------------------------------------
    val_ratio = float(cfg.get("val_set_size", 0))
    if 0.0 < val_ratio < 1.0:
        split = raw.train_test_split(test_size=val_ratio, seed=42, shuffle=True)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = raw, None

    # ---------- 3. filter negatives early (QA) ------------------------------
    if "label" in train_ds.column_names:
        train_ds = train_ds.filter(lambda ex: ex["label"] in (1, True))
        if eval_ds:
            eval_ds = eval_ds.filter(lambda ex: ex["label"] in (1, True))

    # ---------- 4. tokenise --------------------------------------------------
    processor = _build_processor(cfg, tokenizer)
    train_ds = train_ds.map(processor, remove_columns=train_ds.column_names)
    if eval_ds:
        eval_ds = eval_ds.map(processor, remove_columns=eval_ds.column_names)

    # drop any None returned rows (map keeps them as None)
    train_ds = train_ds.filter(lambda ex: ex["input_ids"] is not None)
    if eval_ds:
        eval_ds = eval_ds.filter(lambda ex: ex["input_ids"] is not None)

    return train_ds, eval_ds


# ────────────────────────────────────────────────────────────────────────────
# Tokenizer preparation (adds missing chat tokens, PAD)                       
# ────────────────────────────────────────────────────────────────────────────
_CHAT_TOKENS = [
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
    "<|im_start|>", "<|im_end|>",
    "<<SYS>>", "<</SYS>>", "[INST]", "[/INST]",
    "<|system|>", "<|user|>", "<|assistant|>",
]


def prepare_tokenizer(
    model_name: str,
    hub_token: str | None,
    cfg: dict,
) -> PreTrainedTokenizerBase:
    """Return tokenizer with all chat tokens & PAD present; flag if resized."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        use_auth_token=hub_token,
        trust_remote_code=True,
    )

    # 1. gather existing special tokens --------------------------------------
    raw_vals = tokenizer.special_tokens_map_extended.values()
    flat = list(chain.from_iterable(v if isinstance(v, (list, tuple)) else [v] for v in raw_vals))
    existing = set(flat)

    # 2. add missing tokens ---------------------------------------------------
    missing = [tok for tok in _CHAT_TOKENS if tok not in existing]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        tokenizer._added_tokens = True  # signal resize needed
    else:
        tokenizer._added_tokens = False

    # 3. ensure PAD token -----------------------------------------------------
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer._added_tokens = True

    # 4. choose padding side --------------------------------------------------
    tokenizer.padding_side = cfg.get("padding_side", "right")

    return tokenizer


# ────────────────────────────────────────────────────────────────────────────
# Collator (dynamic padding, aligned labels)                                  
# ────────────────────────────────────────────────────────────────────────────
class ChatDataCollator:
    """Batch‑builder that pads to longest seq and aligns labels."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int | None = 8):
        self.tok = tokenizer
        self.mult = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad input_ids / attention_mask fast‑path
        batch_inputs = self.tok(
            [f["input_ids"] for f in features],
            padding=True,
            truncation=False,
            return_tensors="pt",
            pad_to_multiple_of=self.mult,
        )
        max_len = batch_inputs["input_ids"].size(1)

        # Align labels --------------------------------------------------------
        padded_labels = []
        for f in features:
            lbl = f["labels"]
            pad = max_len - len(lbl)
            padded_labels.append(lbl + [-100] * pad)
        batch_inputs["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch_inputs
