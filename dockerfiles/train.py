#!/usr/bin/env python3
import os
import argparse
import logging
from functools import partial
from datasets import load_dataset, Dataset
from typing import Tuple, Dict, Any, List
import yaml
from ignite.engine import Engine, Events
from ignite.handlers.lr_finder import FastaiLRFinder
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    SchedulerType,
)
import time
from transformers import TrainerCallback, TrainerControl, TrainerState
import bitsandbytes as bnb

# Optional imports for LoRA adapters and DPO
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = get_peft_model = prepare_model_for_kbit_training = None


# Disable parallel tokenizer threads to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


###### Custom Callbacks #####
class TimeLimitCallback(TrainerCallback):
    """Stop training after a fixed number of hours."""

    def __init__(self, max_hours: float):
        """
        Args:
            max_hours: training time budget in hours
        """
        self.max_seconds = max_hours * 3600.0 * 0.95
        self.start_time: float | None = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # record the training start time
        self.start_time = time.time()
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # only check once we've started
        if self.start_time is None:
            return control
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            print(f"\n⏱️  Reached time limit of {self.max_seconds/3600:.2f}h — stopping training.")
            control.should_training_stop = True
        return control


def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal LM with SFT or DPO")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    return logging.getLogger(__name__)


def prepare_tokenizer(model_name: str, hub_token: str = None, max_length: int = 2048):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        use_auth_token=hub_token,
    )
    # 1) force left‐padding
    tokenizer.padding_side = "left"

    # 2) default truncation to max_length
    if hasattr(tokenizer, "enable_truncation"):
        tokenizer.enable_truncation(max_length=max_length)

    # 3) default padding on every call
    if hasattr(tokenizer, "enable_padding"):
        tokenizer.enable_padding()

    # fallback EOS→PAD
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

# ============== 1. Schema detection utilities =================================
def _lower_keys(example: Dict[str, Any]) -> Dict[str, Any]:
    return {k.lower(): v for k, v in example.items()}

def detect_schema(example: Dict[str, Any]) -> str:
    """Return one of: 'instruct', 'qa', 'prompt_completion', 'dpo', 'plain'."""
    e = _lower_keys(example)
    keys = set(e)
    if {"instruction", "output"}.issubset(keys):
        return "instruct"
    if {"question", "answer"}.issubset(keys):
        return "qa"
    if {"prompt", "completion"}.issubset(keys) or {"prompt", "response"}.issubset(keys):
        return "prompt_completion"
    if {"chosen", "rejected"}.issubset(keys):
        return "dpo"
    return "plain"

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Chat‑template helpers  (multi‑model)
# ──────────────────────────────────────────────────────────────────────────────
#  ── token strings used by each family ──
_LL3_START  = "<|start_header_id|>"
_LL3_END    = "<|end_header_id|>"
_EOT        = "<|eot_id|>"

_CHATML_BEG = "<|im_start|>"
_CHATML_END = "<|im_end|>"

#  ── concrete builders ──
def llama3_chat(system: str, user: str, assistant: str = "") -> str:
    return (
        f"{_LL3_START}system{_LL3_END}\n{system}{_EOT}\n"
        f"{_LL3_START}user{_LL3_END}\n{user}{_EOT}\n"
        f"{_LL3_START}assistant{_LL3_END}\n{assistant}"
    )

def llama2_chat(system: str, user: str, assistant: str = "") -> str:
    # works for Llama‑2 **and** Mistral chat / Mixtral
    sys = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
    return f"<s>[INST] {sys}{user} [/INST] {assistant}"

def qwen_chat(system: str, user: str, assistant: str = "") -> str:
    return (
        f"{_CHATML_BEG}system\n{system}{_CHATML_END}\n"
        f"{_CHATML_BEG}user\n{user}{_CHATML_END}\n"
        f"{_CHATML_BEG}assistant\n{assistant}"
    )

def phi_chat(system: str, user: str, assistant: str = "") -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"

def plain_chat(system: str, user: str, assistant: str = "") -> str:
    return f"{user}\n{assistant}"

#  ── mapping: model‑name → template fn ──
_TEMPLATE_REGISTRY = {
    "llama‑3":    llama3_chat,
    "llama3":     llama3_chat,
    "llama‑2":    llama2_chat,
    "llama2":     llama2_chat,
    "mistral":    llama2_chat,
    "mixtral":    llama2_chat,
    "qwen":       qwen_chat,
    "phi":        phi_chat,
    "zephyr":     qwen_chat,      # Zephyr uses ChatML tokens
}

def get_template_fn(model_name: str, explicit: str | None = None):
    """Return a template builder based on explicit cfg or model name."""
    key = (explicit or model_name).lower()
    for substr, fn in _TEMPLATE_REGISTRY.items():
        if substr in key:
            return fn
    return plain_chat  # fallback

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Example → (input_ids, labels)
# ──────────────────────────────────────────────────────────────────────────────
def build_processor(cfg: dict, tokenizer):
    seq_len         = int(cfg.get("sequence_len", 2048))
    train_on_inputs = bool(cfg.get("train_on_inputs", False))
    tmpl_fn         = get_template_fn(
        cfg["base_model"],
        cfg.get("chat_template"),     # optional override in YAML
    )
    SYS_MSG         = cfg.get(
        "system_prompt",
        "You are a helpful assistant."
    )

    def proc(example: Dict[str, Any]) -> Dict[str, Any]:
        schema = detect_schema(example)

        # ---- build prompt + answer ------------------------------------------
        if schema == "instruct":
            prompt  = example["instruction"]
            if (inp := example.get("input")):
                prompt = f"{prompt} {inp}"
            answer  = str(example.get("output", ""))
        elif schema == "qa":
            prompt, answer = example["question"], example["answer"]
        elif schema == "prompt_completion":
            prompt  = example["prompt"]
            answer  = example.get("completion") or example.get("response", "")
        elif schema == "dpo":
            prompt, answer = example["prompt"], example["chosen"]
        else:  # 'plain'
            prompt, answer = "", str(example[next(iter(example))])

        chat_prompt = tmpl_fn(SYS_MSG, prompt)
        full_text   = chat_prompt + answer + tokenizer.eos_token

        # ---- tokenise -------------------------------------------------------
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=seq_len,
            return_attention_mask=False,
        )
        input_ids = enc["input_ids"]
        labels    = input_ids.copy()

        # ---- mask prompt tokens, if desired ---------------------------------
        if not train_on_inputs:
            prompt_len = len(
                tokenizer(chat_prompt, add_special_tokens=False)["input_ids"]
            )
            labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": input_ids, "labels": labels}

    return proc

# ============== 4. Public loader =============================================
def load_and_tokenise_dataset(cfg: dict, tokenizer) -> Tuple[Dataset, Dataset | None]:
    ds_cfg = cfg["datasets"][0]
    ds_type = ds_cfg.get("ds_type", ds_cfg.get("type", "hf")).lower()

    # -- Load raw dataset ------------------------------------------------------
    if ds_type in {"json", "csv", "text"}:
        raw = load_dataset(ds_type, data_files={"train": ds_cfg["path"]})["train"]
    else:  # assume HF repo path
        raw = load_dataset(ds_cfg["path"], split=ds_cfg.get("split", "train"))

    val_size = float(cfg.get("val_set_size", 0))
    train_ds, eval_ds = (raw, None)
    if val_size > 0:
        splits = raw.train_test_split(test_size=val_size, seed=42, shuffle=True)
        train_ds, eval_ds = splits["train"], splits["test"]

    # -- Tokenise + label ------------------------------------------------------
    processor = build_processor(cfg, tokenizer)
    train_ds = train_ds.map(processor, remove_columns=train_ds.column_names)
    if eval_ds:
        eval_ds = eval_ds.map(processor, remove_columns=eval_ds.column_names)

    return train_ds, eval_ds


def load_model(model_name: str, cfg: dict) -> AutoModelForCausalLM:
    common_kwargs = {
        'use_auth_token': cfg.get('hub_token'),
        'load_in_8bit': bool(cfg.get('load_in_8bit', False)),
        'torch_dtype': torch.bfloat16 if cfg.get('bf16') and torch.cuda.is_bf16_supported() else None,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='flash_attention_2',
            trust_remote_code=True,
            **common_kwargs
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **common_kwargs)


def apply_lora_adapter(model: AutoModelForCausalLM, cfg: dict) -> AutoModelForCausalLM:
    if get_peft_model is None:
        raise ImportError("peft library is required for LoRA adapters.")

    if cfg.get('load_in_8bit', False):
        model = prepare_model_for_kbit_training(model)

    # Determine target modules for LoRA
    targets = cfg.get('target_modules') or []
    if not targets:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(x in name.lower() for x in ('attn', 'attention')):
                targets.append(name.split('.')[-1])
        targets = list(set(targets))
        if not targets:
            raise ValueError("Could not auto-detect attention modules for LoRA. Please set 'target_modules' in config.")

    peft_config = LoraConfig(
        r=int(cfg.get('lora_r', 16)),
        lora_alpha=int(cfg.get('lora_alpha', 16)),
        target_modules=targets,
        lora_dropout=float(cfg.get('lora_dropout', 0.05)),
        bias='none',
        task_type='CAUSAL_LM'
    )
    return get_peft_model(model, peft_config)



def build_trainer(cfg: dict, model, tokenizer, train_ds, eval_ds, callbacks):
    # ── SFT Trainer branch ────────────────────────────────────────
    tf_args = TrainingArguments(
        output_dir=cfg.get('output_dir', './outputs'),
        auto_find_batch_size=True,
        bf16=bool(cfg.get('bf16', False)),
        gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps', 2)),
        dataloader_num_workers=int(cfg.get('dataloader_num_workers', 8)),
        num_train_epochs=int(cfg.get('num_epochs', 1)),
        learning_rate=float(cfg.get('learning_rate', 5e-5)),
        optim=cfg.get('optimizer', 'lion_8bit'),
        warmup_steps=int(cfg.get('warmup_steps', 25)),
        lr_scheduler_type=SchedulerType.COSINE,
        load_best_model_at_end=True,
        max_steps=int(cfg.get('max_steps', -1)),
        logging_steps=int(cfg.get('logging_steps', 100)),
        eval_strategy='steps' if eval_ds else 'no',
        save_strategy='best',
        eval_steps=int(cfg.get('eval_steps')) if cfg.get('eval_steps') is not None else None,
        save_steps=int(cfg.get('save_steps')) if cfg.get('save_steps') is not None else None,
        save_total_limit=int(cfg.get('save_total_limit')) if cfg.get('save_total_limit') is not None else None,
        metric_for_best_model=cfg.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=bool(cfg.get('greater_is_better', False)),
        weight_decay=float(cfg.get('weight_decay', 0.0)),
        fp16=bool(cfg.get('fp16', False)),
        logging_dir=cfg.get('logging_dir', './logs'),
        push_to_hub=True,
        run_name=cfg.get('wandb_run'),
        hub_model_id=cfg.get('hub_model_id'),
        hub_token=cfg.get('hub_token'),
        hub_strategy='every_save',
        use_liger_kernel=True,
    )
    logger = setup_logger()
    logger.info("Initializing SFT Trainer")

    return Trainer(
        model=model,
        args=tf_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
        callbacks=callbacks,
        processing_class=tokenizer,
    )



def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger()

    # Performance flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logger.info("Loaded config from %s", args.config)
    accelerator = Accelerator(log_with="wandb", mixed_precision="bf16")
    accelerator.init_trackers(cfg.get('wandb_project'), config=cfg)

    # after loading cfg...
    seq_len = int(cfg.get("sequence_len", 2048))
    tokenizer = prepare_tokenizer(cfg["base_model"], cfg.get("hub_token"), max_length=seq_len)
    model = load_model(cfg['base_model'], cfg)

    if cfg.get('adapter') == 'lora':
        model = apply_lora_adapter(model, cfg)

    train_ds, eval_ds = load_and_tokenise_dataset(cfg, tokenizer)

    callbacks = []
    if cfg.get('early_stopping', True):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.get('early_stopping_patience', 8))
        )
    # add time‐limit
    max_hours = int(cfg.get('hours_to_complete'))  # e.g. 4.0
    if max_hours is not None:
        callbacks.append(TimeLimitCallback(max_hours))

    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds, callbacks)

    logger.info("Starting Full Model Training...")

    trainer = accelerator.prepare(trainer)
    trainer.train()



if __name__ == '__main__':
    main()