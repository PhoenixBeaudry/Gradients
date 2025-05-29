#!/usr/bin/env python3
import os
import multiprocessing as mp
if __name__ == "__main__":          # important when Accelerate spawns ranks
    mp.set_start_method("forkserver", force=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"    # fork-safe, see below
os.environ.setdefault("DEEPSPEED_TIMEOUT", "3600")   # 1 h NCCL watchdog  :contentReference[oaicite:0]{index=0}
import argparse
import logging
import yaml
from datetime import datetime
import torch
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
from training_helpers.custom_callbacks import TimeLimitCallback
from training_helpers.dataset_helpers import load_sft_datasets, load_dpo_datasets, load_grpo_datasets, load_tokenizer
from training_helpers.model_helpers import load_model, get_lora_adapter
from training_helpers.trainer_helpers import build_trainer_args, reward_functions

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


def build_trainer(cfg: dict, model, peft_config, tokenizer, train_ds, eval_ds):

    #### Callbacks ####
    callbacks = []
    if cfg.get('early_stopping', True):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.get('early_stopping_patience', 8))
        )
    
    # calculate time left for job
    time_remaining = datetime.fromisoformat(cfg['required_finish_time']) - datetime.now()
    seconds_remaining = max(0.0, time_remaining.total_seconds())

    if seconds_remaining is not None:
        callbacks.append(TimeLimitCallback(seconds_remaining*0.95))
    ###################


    ##### Training Arguments ####
    trainer_kwargs = build_trainer_args(cfg)

    

    #####################################
    logger = setup_logger()
    logger.info("Initializing Trainer")
    if cfg["rl"] == "dpo":
        trainer_args = DPOConfig(
            **trainer_kwargs,
        )
        return DPOTrainer(
            model=model,
            ref_model=None,
            args=trainer_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config
        )
    elif cfg["rl"] == "grpo":
        trainer_args = GRPOConfig(
            **trainer_kwargs,
        )
        return GRPOTrainer(
            model=model,
            args=trainer_args,
            reward_funcs=reward_functions(cfg),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config
        )
    else:
        trainer_args = SFTConfig(
            **trainer_kwargs,
        )
        return SFTTrainer(
            model=model,
            args=trainer_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config
        )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    #####################################################

    logger = setup_logger()
    
    # Performance flags
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    logger.info("Loaded config from %s", args.config)
    
    # after loading cfg...
    tokenizer = load_tokenizer(cfg['base_model'], cfg)

    if cfg["rl"] == "dpo":
        train_dataset, eval_dataset = load_dpo_datasets(cfg)
    elif cfg["rl"] == "grpo":
        train_dataset, eval_dataset = load_grpo_datasets(cfg)
    else:
        train_dataset, eval_dataset = load_sft_datasets(cfg)

    model = load_model(cfg['base_model'], cfg)

    if cfg.get('adapter') == 'lora':
        peft_config = get_lora_adapter(model, cfg)
    else:
        peft_config = None

    if cfg["testing"]:
        # ── HPO trial: auto‑subset the corpus ───────────────────────────────────
        # 1. compute target subset sizes
        target_train = int(len(train_dataset)/2)
        target_eval = int(len(eval_dataset)/2)
        # deterministic shuffle → reproducible trials
        train_dataset = train_dataset.shuffle(seed=42).select(range(target_train))
        eval_dataset  = eval_dataset.shuffle(seed=42).select(range(target_eval))

    elif cfg["hpo_run"]:
        # ── HPO trial: auto‑subset the corpus ───────────────────────────────────
        # 1. compute target subset sizes
        SUBSET_FRAC   = 0.02          # 5 %
        MIN_PAIRS     = 1_500         # never go below this
        MAX_PAIRS     = 8_000        # never go above this
        target_train = int(max(MIN_PAIRS, min(MAX_PAIRS, len(train_dataset) * SUBSET_FRAC)))
        target_eval = int(max(MIN_PAIRS, min(MAX_PAIRS, len(eval_dataset) * SUBSET_FRAC)))

        # No lower than dataset size
        target_train = min(target_train, len(train_dataset))
        target_eval = min(target_eval, len(eval_dataset))

        # deterministic shuffle → reproducible trials
        train_dataset = train_dataset.shuffle(seed=42).select(range(target_train))
        eval_dataset  = eval_dataset .shuffle(seed=42).select(range(target_eval))

    

    logger.info("Starting Full Model Training...")
    trainer = build_trainer(cfg, model, peft_config, tokenizer, train_dataset, eval_dataset)
    
    try:
        trainer.train()
    finally:
        if not cfg["hpo_run"]:
            trainer.push_to_hub()
        


if __name__ == '__main__':
    main()