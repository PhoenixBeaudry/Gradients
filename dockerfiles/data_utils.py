# data_utils.py
"""data_utils.py – unified, model‑agnostic dataset+tokenizer utilities
---------------------------------------------------------------------
This module gives you three public helpers that align with the *main* script
shown by the user.

    ▸ prepare_tokenizer(model_name, hub_token, cfg) -> tokenizer
      - ensures all chat special‑tokens + PAD exist
      - sets ``tokenizer._added_tokens`` boolean so the caller can decide
        whether to call ``model.resize_token_embeddings``

    ▸ load_and_tokenise_dataset(cfg, tokenizer) -> train_ds, eval_ds
      - auto‑detects schema (instruction, QA, prompt‑completion, DPO, plain)
      - chooses the right chat template for Llama‑3/2, Mistral/Mixtral,
        Qwen‑2 / Zephyr, Phi‑3, or falls back to a plain template
      - applies masking when ``train_on_inputs: false``
      - filters negative QA rows (label != 1) to mimic Axolotl behaviour

    ▸ ChatDataCollator(tokenizer) – dynamic padding collator that keeps
      ``input_ids`` and ``labels`` aligned and pads labels with –100.

They are **stateless**, fully typed and depend only on ``datasets`` and
``transformers``.
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

__all__ = [
    "prepare_tokenizer",
    "load_and_tokenise_dataset",
    "ChatDataCollator",
]

# ---------------------------------------------------------------------------
# 1. Chat‑family → template registry
# ---------------------------------------------------------------------------
_LL3_START, _LL3_END, _EOT = "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"
_CHATML_BEG, _CHATML_END = "<|im_start|>", "<|im_end|>"


def _tmpl_llama3(system: str, user: str, assistant: str = "") -> str:
    return (
        f"{_LL3_START}system{_LL3_END}\n{system}{_EOT}\n"
        f"{_LL3_START}user{_LL3_END}\n{user}{_EOT}\n"
        f"{_LL3_START}assistant{_LL3_END}\n{assistant}"
    )


def _tmpl_llama2(system: str, user: str, assistant: str = "") -> str:
    sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
    return f"<s>[INST] {sys_block}{user} [/INST] {assistant}"


def _tmpl_qwen(system: str, user: str, assistant: str = "") -> str:
    return (
        f"{_CHATML_BEG}system\n{system}{_CHATML_END}\n"
        f"{_CHATML_BEG}user\n{user}{_CHATML_END}\n"
        f"{_CHATML_BEG}assistant\n{assistant}"
    )


def _tmpl_phi(system: str, user: str, assistant: str = "") -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"


def _tmpl_plain(system: str, user: str, assistant: str = "") -> str:
    return f"{user}\n{assistant}"


_TEMPLATE_TABLE: dict[str, Callable[[str, str, str], str]] = {
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


def _select_template(model_name: str, explicit: str | None) -> Callable[[str, str, str], str]:
    key = (explicit or model_name).lower()
    for sub, fn in _TEMPLATE_TABLE.items():
        if sub in key:
            return fn
    return _tmpl_plain


# ---------------------------------------------------------------------------
# 2. Schema detection
# ---------------------------------------------------------------------------

def _detect_schema(example: Dict[str, Any]) -> str:
    keys = {k.lower() for k in example}
    if {"instruction", "output"} <= keys:
        return "instruct"
    if {"question", "answer"} <= keys:
        return "qa"
    if ("prompt" in keys) and ("completion" in keys or "response" in keys):
        return "prompt_completion"
    if {"chosen", "rejected"} <= keys:
        return "dpo"
    return "plain"


# ---------------------------------------------------------------------------
# 3. Processor factory
# ---------------------------------------------------------------------------

def _build_processor(cfg: dict, tokenizer: PreTrainedTokenizerBase):
    seq_len = int(cfg.get("sequence_len", 2048))
    mask_inputs = not cfg.get("train_on_inputs", False)
    tmpl_fn = _select_template(cfg["base_model"], cfg.get("chat_template"))
    sys_msg = cfg.get("system_prompt", "You are a helpful assistant.")

    def _proc(ex: Dict[str, Any]):  # may return None to drop row
        schema = _detect_schema(ex)

        # -------- extract prompt & answer ----------------------------------
        if schema == "instruct":
            prompt = ex["instruction"] + (" " + ex["input"] if ex.get("input") else "")
            answer = ex.get("output", "")
        elif schema == "qa":
            if "label" in ex and ex["label"] not in (1, True):
                return None  # negative example
            prompt, answer = ex["question"], ex["answer"]
        elif schema == "prompt_completion":
            prompt = ex["prompt"]
            answer = ex.get("completion") or ex.get("response", "")
        elif schema == "dpo":
            prompt, answer = ex["prompt"], ex["chosen"]
        else:  # plain – first field is the whole supervised sequence
            first_key = next(iter(ex))
            prompt, answer = "", str(ex[first_key])

        # -------- build chat prompt ----------------------------------------
        chat = tmpl_fn(sys_msg, prompt)
        full = chat + answer + (tokenizer.eos_token or "")

        # -------- tokenise --------------------------------------------------
        enc = tokenizer(full, truncation=True, max_length=seq_len, return_attention_mask=False)
        ids = enc["input_ids"]
        labels = ids.copy()

        if mask_inputs:
            prompt_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
            cutoff = min(len(prompt_ids), len(labels))
            labels[:cutoff] = [-100] * cutoff

        return {"input_ids": ids, "labels": labels}

    return _proc


# ---------------------------------------------------------------------------
# 4. Dataset loader (public)
# ---------------------------------------------------------------------------

def load_and_tokenise_dataset(cfg: dict, tokenizer: PreTrainedTokenizerBase) -> Tuple[Dataset, Dataset | None]:
    """Load dataset, map processor, return (train, eval) Datasets."""

    ds_spec = cfg["datasets"][0]
    dtype = ds_spec.get("ds_type", ds_spec.get("type", "hf")).lower()

    # 1 ◇ raw load -----------------------------------------------------------
    if dtype in {"json", "csv", "text"}:
        raw = load_dataset(dtype, data_files={"train": ds_spec["path"]})["train"]
    else:
        raw = load_dataset(ds_spec["path"], split=ds_spec.get("split", "train"))

    # 2 ◇ optional val split -------------------------------------------------
    val_ratio = float(cfg.get("val_set_size", 0))
    if 0.0 < val_ratio < 1.0:
        split = raw.train_test_split(test_size=val_ratio, seed=42, shuffle=True)
        train, eval_ = split["train"], split["test"]
    else:
        train, eval_ = raw, None

    processor = _build_processor(cfg, tokenizer)

    # 3 ◇ map & drop None rows ----------------------------------------------
    train = train.map(processor, remove_columns=train.column_names)
    train = train.filter(lambda x: x["input_ids"] is not None)
    if eval_:
        eval_ = eval_.map(processor, remove_columns=eval_.column_names)
        eval_ = eval_.filter(lambda x: x["input_ids"] is not None)

    return train, eval_


# ---------------------------------------------------------------------------
# 5. Tokenizer preparation
# ---------------------------------------------------------------------------
_CHAT_TOKENS: Sequence[str] = [
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
    "<|im_start|>", "<|im_end|>",
    "<<SYS>>", "<</SYS>>", "[INST]", "[/INST]",
    "<|system|>", "<|user|>", "<|assistant|>",
]


def prepare_tokenizer(model_name: str, hub_token: str | None, cfg: dict) -> PreTrainedTokenizerBase:
    """Load HF tokenizer, inject missing chat tokens, ensure PAD.

    Sets ``tokenizer._added_tokens`` bool so caller can resize embeddings.
    Returns the tokenizer (single object – main script API).
    """

    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        use_auth_token=hub_token,
        trust_remote_code=True,
    )

    # a) existing specials ---------------------------------------------------
    raw_vals = tok.special_tokens_map_extended.values()
    flat = list(
        chain.from_iterable(val if isinstance(val, (list, tuple)) else [val] for val in raw_vals)
    )
    existing = set(flat)

    # b) add missing chat tokens --------------------------------------------
    missing = [t for t in _CHAT_TOKENS if t not in existing]
    tok._added_tokens = False  # default
    if missing:
        tok.add_special_tokens({"additional_special_tokens": missing})
        tok._added_tokens = True

    # c) ensure PAD token ----------------------------------------------------
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok._added_tokens = True

    # d) padding side --------------------------------------------------------
    tok.padding_side = cfg.get("padding_side", "right")

    return tok


# ---------------------------------------------------------------------------
# 6. Collator
# ---------------------------------------------------------------------------
class ChatDataCollator:
    """Pads to longest sequence in batch & aligns labels (‑100 padding)."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int | None = 8):
        self.tok = tokenizer
        self.mult = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.tok(
            [f["input_ids"] for f in features],
            padding=True,
            truncation=False,
            return_tensors="pt",
            pad_to_multiple_of=self.mult,
        )
        max_len = batch_inputs["input_ids"].shape[1]

        padded_labels = [
            f["labels"] + [-100] * (max_len - len(f["labels"])) for f in features
        ]
        batch_inputs["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch_inputs
