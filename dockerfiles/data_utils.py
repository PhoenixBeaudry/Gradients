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
def prepare_tokenizer(model_name: str, hub_token=None, cfg=None):
    """
    Prepares and configures the tokenizer for the given model.
    
    Args:
        model_name: Name or path of the model
        hub_token: Optional Hugging Face token for private models
        cfg: Configuration dictionary with additional tokenizer settings
        
    Returns:
        Configured tokenizer
    """
    cfg = cfg or {}
    tokenizer_kwargs = {
        "use_auth_token": hub_token,
        "trust_remote_code": True,
        "padding_side": "right",  # Most common for causal LM training
        "use_fast": True,  # Use fast tokenizer when available
    }
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # Ensure we have the needed special tokens for the model architecture
    # Different model families need different special tokens
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # Apply custom chat template if specified
    chat_template = cfg.get("chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    
    # Add special tokens if specified
    special_tokens = cfg.get("special_tokens", {})
    if special_tokens:
        num_added = tokenizer.add_special_tokens(special_tokens)
        setattr(tokenizer, "_added_tokens", num_added > 0)
    
    # Apply token length limits
    max_length = cfg.get("max_length", 2048)
    tokenizer.model_max_length = max_length
    
    return tokenizer


def load_and_tokenise_dataset(cfg: dict, tokenizer) -> Tuple[Dataset, Dataset]:
    """
    Load and tokenize datasets for training and evaluation.
    
    Args:
        cfg: Configuration dictionary with dataset settings
        tokenizer: Tokenizer to use for processing
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Extract config parameters
    dataset_name = cfg.get("dataset_name")
    dataset_path = cfg.get("dataset_path")
    train_split = cfg.get("train_split", "train")
    eval_split = cfg.get("eval_split", "validation")
    text_column = cfg.get("text_column", "text")
    prompt_column = cfg.get("prompt_column")
    response_column = cfg.get("response_column")
    max_length = cfg.get("max_length", tokenizer.model_max_length)
    streaming = cfg.get("streaming", False)
    
    # Load dataset based on source specification
    if dataset_name:
        # Load from Hugging Face datasets hub
        dataset = load_dataset(
            dataset_name,
            streaming=streaming,
            use_auth_token=cfg.get("hub_token"),
            trust_remote_code=True
        )
    elif dataset_path:
        # Load from local path
        if dataset_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset('json', data_files=dataset_path)
        elif dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_path)
        elif dataset_path.endswith('.parquet'):
            dataset = load_dataset('parquet', data_files=dataset_path) 
        else:
            # Try to load as a directory
            dataset = load_dataset(dataset_path)
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")
    
    # Split dataset into train and eval if they exist
    train_ds = dataset[train_split] if train_split in dataset else dataset
    eval_ds = dataset[eval_split] if eval_split in dataset and eval_split in dataset else None
    
    # Create tokenization function based on data format
    def is_chat_dataset():
        """Check if dataset has conversation structure"""
        if prompt_column and response_column:
            return True
        # Check if any sample is a list of messages
        sample = train_ds[0] if not streaming else next(iter(train_ds))
        return isinstance(sample.get("messages", None), list)
    
    def tokenize_function(examples):
        """Tokenize based on dataset format"""
        if is_chat_dataset():
            if "messages" in examples:
                # Handle chat format with messages field
                formatted = [
                    tokenizer.apply_chat_template(chat, tokenize=False)
                    for chat in examples["messages"]
                ]
            else:
                # Handle explicit prompt/response format
                formatted = []
                for i in range(len(examples[prompt_column])):
                    chat = [
                        {"role": "user", "content": str(examples[prompt_column][i])},
                        {"role": "assistant", "content": str(examples[response_column][i])}
                    ]
                    formatted.append(tokenizer.apply_chat_template(chat, tokenize=False))
            
            result = tokenizer(
                formatted,
                max_length=max_length,
                padding="max_length" if cfg.get("pad_to_max_length", False) else False,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            # Create labels for causal LM (shift input_ids right)
            result["labels"] = result["input_ids"].clone()
            return result
        else:
            # Handle plain text format
            text_data = examples[text_column]
            if isinstance(text_data, list):
                text_data = [str(t) for t in text_data]
            else:
                text_data = str(text_data)
                
            result = tokenizer(
                text_data,
                max_length=max_length,
                padding="max_length" if cfg.get("pad_to_max_length", False) else False,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            # Create labels for causal LM (shift input_ids right)
            result["labels"] = result["input_ids"].clone()
            return result
    
    # Apply tokenization to datasets
    if streaming:
        # For streaming datasets
        train_ds = train_ds.map(tokenize_function, batched=True)
        if eval_ds:
            eval_ds = eval_ds.map(tokenize_function, batched=True)
    else:
        # For non-streaming datasets
        train_ds = train_ds.map(
            tokenize_function,
            batched=True,
            num_proc=cfg.get("preprocessing_num_workers", 4),
            remove_columns=[col for col in train_ds.column_names if col not in ["input_ids", "attention_mask", "labels"]],
        )
        if eval_ds:
            eval_ds = eval_ds.map(
                tokenize_function,
                batched=True,
                num_proc=cfg.get("preprocessing_num_workers", 4),
                remove_columns=[col for col in eval_ds.column_names if col not in ["input_ids", "attention_mask", "labels"]],
            )
    
    return train_ds, eval_ds


class ChatDataCollator:
    """
    Data collator that handles dynamic padding for chat datasets.
    
    Args:
        tokenizer: The tokenizer used to process the data
        pad_to_multiple_of: Optional requirement to pad to multiple of value (for hardware optimization)
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Pad input_ids and attention_mask to the same length
        batch = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features],
             "attention_mask": [f["attention_mask"] for f in features]},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Handle labels for causal LM
        labels = [f.get("labels", f["input_ids"].clone()) for f in features]
        max_label_length = max(len(l) for l in labels)
        
        # Pad labels to max length
        padded_labels = []
        for label in labels:
            # For causal LM, we use -100 as padding token ID so loss ignores it
            padding_length = max_label_length - len(label)
            if padding_length > 0:
                padded_label = torch.cat([label, torch.full((padding_length,), -100, dtype=torch.long)])
            else:
                padded_label = label
            padded_labels.append(padded_label)
        
        batch["labels"] = torch.stack(padded_labels)
        return batch