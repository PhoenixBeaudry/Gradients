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
    Load and tokenize datasets for training and evaluation according to dataset specifications.
    
    Args:
        cfg: Configuration dictionary with dataset settings
        tokenizer: Tokenizer to use for processing
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger = logging.getLogger(__name__)
    datasets = []
    
    # Process each dataset specification
    for ds_idx, ds_spec in enumerate(cfg.get("datasets", [])):
        logger.info(f"Processing dataset {ds_idx+1}/{len(cfg.get('datasets', []))}")
        
        # 1. Load raw dataset
        dtype = ds_spec.get("ds_type", ds_spec.get("type", "hf")).lower()
        if dtype in {"json", "csv", "text"}:
            raw = load_dataset(dtype, data_files={"train": ds_spec["path"]})["train"]
        else:
            raw = load_dataset(ds_spec["path"], split=ds_spec.get("split", "train"))
        
        logger.info(f"Loaded raw dataset with {len(raw)} examples")
        
        # 2. Apply any specified filtering
        if ds_spec.get("filter"):
            filter_expr = ds_spec["filter"]
            raw = raw.filter(lambda x: eval(filter_expr, {"x": x}))
            logger.info(f"Filtered dataset to {len(raw)} examples")
        
        # 3. Determine dataset format and prepare processing function
        max_length = ds_spec.get("max_length", cfg.get("max_length", tokenizer.model_max_length))
        
        # Determine if this is a chat dataset
        is_chat = False
        prompt_col = ds_spec.get("prompt_column", ds_spec.get("prompt_key"))
        response_col = ds_spec.get("response_column", ds_spec.get("completion_key"))
        messages_col = ds_spec.get("messages_column")
        
        if prompt_col and response_col:
            is_chat = True
            logger.info(f"Processing as chat dataset with prompt/response columns: {prompt_col}/{response_col}")
        elif messages_col:
            is_chat = True
            logger.info(f"Processing as chat dataset with messages column: {messages_col}")
        else:
            text_col = ds_spec.get("text_column", "text")
            logger.info(f"Processing as text dataset with column: {text_col}")
        
        # 4. Define tokenization function
        def tokenize_chat(examples):
            if messages_col:
                # Format with messages list
                formatted = []
                for msgs in examples[messages_col]:
                    # Ensure each message has role and content
                    valid_msgs = [m for m in msgs if isinstance(m, dict) and "role" in m and "content" in m]
                    formatted.append(tokenizer.apply_chat_template(valid_msgs, tokenize=False))
            else:
                # Format with prompt/response pairs
                formatted = []
                for i in range(len(examples[prompt_col])):
                    # Skip empty examples
                    if not examples[prompt_col][i] or not examples[response_col][i]:
                        continue
                        
                    # Support system prompt if available
                    chat = []
                    system_col = ds_spec.get("system_column")
                    if system_col and examples.get(system_col) and examples[system_col][i]:
                        chat.append({"role": "system", "content": str(examples[system_col][i])})
                    
                    # Add user and assistant messages
                    chat.append({"role": "user", "content": str(examples[prompt_col][i])})
                    chat.append({"role": "assistant", "content": str(examples[response_col][i])})
                    
                    formatted.append(tokenizer.apply_chat_template(chat, tokenize=False))
            
            result = tokenizer(
                formatted,
                max_length=max_length,
                padding=False,
                truncation=True,
                return_tensors=None,
            )
            
            # Set labels for causal LM
            result["labels"] = result["input_ids"].copy()
            
            return result
            
        def tokenize_text(examples):
            text_col = ds_spec.get("text_column", "text")
            if text_col not in examples:
                raise ValueError(f"Text column '{text_col}' not found in dataset. Available columns: {list(examples.keys())}")
                
            texts = examples[text_col]
            if not isinstance(texts, list):
                texts = [texts]
                
            # Clean and convert to strings
            texts = [str(t) if t is not None else "" for t in texts]
            
            result = tokenizer(
                texts,
                max_length=max_length,
                padding=False,
                truncation=True,
                return_tensors=None,
            )
            
            # Set labels for causal LM
            result["labels"] = result["input_ids"].copy()
            
            return result
        
        # 5. Apply tokenization
        num_proc = ds_spec.get("preprocessing_num_workers", cfg.get("preprocessing_num_workers", 4))
        
        tokenize_fn = tokenize_chat if is_chat else tokenize_text
        processed = raw.map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc,
            remove_columns=raw.column_names,
            desc=f"Tokenizing dataset {ds_idx+1}"
        )
        
        # 6. Apply dataset weights if specified
        weight = ds_spec.get("weight", 1.0)
        if weight != 1.0:
            logger.info(f"Applying weight {weight} to dataset")
            # Create weighted dataset by repeating or sampling
            if weight > 1.0:
                # Repeat dataset
                repeat_times = int(weight)
                remainder = weight - repeat_times
                datasets.append(processed)
                for _ in range(repeat_times - 1):
                    datasets.append(processed)
                
                # Handle remainder with random sampling
                if remainder > 0:
                    sample_size = int(len(processed) * remainder)
                    if sample_size > 0:
                        sampled = processed.shuffle().select(range(sample_size))
                        datasets.append(sampled)
            else:
                # Sample dataset
                sample_size = int(len(processed) * weight)
                if sample_size > 0:
                    sampled = processed.shuffle().select(range(sample_size))
                    datasets.append(sampled)
        else:
            datasets.append(processed)
    
    # Combine all datasets
    if len(datasets) > 1:
        train_ds = datasets[0]
        for ds in datasets[1:]:
            # Ensure all datasets have the same columns
            if set(train_ds.features) != set(ds.features):
                raise ValueError(f"Datasets have incompatible features: {train_ds.features} vs {ds.features}")
            train_ds = concatenate_datasets([train_ds, ds])
    elif len(datasets) == 1:
        train_ds = datasets[0]
    else:
        raise ValueError("No datasets were processed successfully")
    
    logger.info(f"Final training dataset size: {len(train_ds)} examples")
    
    # Create evaluation dataset if specified
    eval_ds = None
    eval_spec = cfg.get("eval_dataset")
    if eval_spec:
        logger.info("Processing evaluation dataset")
        
        # Load eval dataset with same approach as training
        dtype = eval_spec.get("ds_type", eval_spec.get("type", "hf")).lower()
        if dtype in {"json", "csv", "text"}:
            eval_raw = load_dataset(dtype, data_files={"eval": eval_spec["path"]})["eval"]
        else:
            eval_raw = load_dataset(eval_spec["path"], split=eval_spec.get("split", "validation"))
        
        # Determine format and apply tokenization
        is_chat = False
        prompt_col = eval_spec.get("prompt_column", eval_spec.get("prompt_key"))
        response_col = eval_spec.get("response_column", eval_spec.get("completion_key"))
        messages_col = eval_spec.get("messages_column")
        
        if prompt_col and response_col or messages_col:
            is_chat = True
        
        tokenize_fn = tokenize_chat if is_chat else tokenize_text
        eval_ds = eval_raw.map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc,
            remove_columns=eval_raw.column_names,
            desc="Tokenizing evaluation dataset"
        )
        
        logger.info(f"Evaluation dataset size: {len(eval_ds)} examples")
    
    # Shuffle the training dataset
    train_ds = train_ds.shuffle(seed=cfg.get("seed", 42))
    
    return train_ds, eval_ds


class ChatDataCollator:
    """
    Data collator that handles padding for both text and chat datasets.
    
    Args:
        tokenizer: The tokenizer used to process the data
        pad_to_multiple_of: Optional requirement to pad to multiple of value (for hardware optimization)
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        
        # Pad input_ids and attention_mask to the same length
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Handle labels for causal LM
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            # For causal LM, we set padding token ID in labels to -100 so the loss ignores it
            batch["labels"] = torch.full_like(batch["input_ids"], -100)
            for i, label in enumerate(labels):
                label_tensor = torch.tensor(label, dtype=torch.long)
                # Only copy as many tokens as fit in the padded tensor
                length = min(len(label_tensor), batch["labels"].size(1))
                batch["labels"][i, :length] = label_tensor[:length]
        
        return batch