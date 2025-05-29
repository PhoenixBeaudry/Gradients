#!/usr/bin/env python3
import argparse
import logging
import yaml
import torch
from datetime import datetime, timedelta
import torch.distributed as dist
from trl import DPOConfig, DPOTrainer
from transformers import (
    EarlyStoppingCallback,
    SchedulerType
)
from training_helpers.custom_callbacks import TimeLimitCallback, add_optuna_callback_if_needed
from training_helpers.dataset_helpers import load_dpo_datasets, load_tokenizer
from training_helpers.model_helpers import load_model, get_lora_adapter



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
    # ── DPO Trainer ────────────────────────────────────────
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

    if cfg["hpo_run"]:
        add_optuna_callback_if_needed(callbacks)
    ###################

    ##### Training Arguments ####
    hf_kwargs = {}
    if not cfg["hpo_run"]:
        hf_kwargs = {
            'hub_model_id': cfg['hub_model_id'],
            'hub_token': cfg['hub_token'],
            'hub_strategy': "end",
            'push_to_hub': True,
        }
        lr_scheduler=SchedulerType.COSINE
    else:
        lr_scheduler=SchedulerType.CONSTANT_WITH_WARMUP
        
    tf_args = DPOConfig(
        output_dir=cfg['output_dir'],
        gradient_accumulation_steps=int(cfg['gradient_accumulation_steps']),
        per_device_train_batch_size=int(cfg['micro_batch_size']),
        per_device_eval_batch_size=int(cfg['micro_batch_size']),
        max_steps=int(cfg['max_steps']),
        learning_rate=float(cfg['learning_rate']),
        beta=float(cfg['beta']),
        optim=cfg['optimizer'],
        label_smoothing=float(cfg['label_smoothing']),
        lr_scheduler_type=lr_scheduler,
        logging_steps=int(cfg['logging_steps']),
        eval_strategy='steps',
        save_strategy='best',
        eval_steps=int(cfg['eval_steps']),
        save_steps=int(cfg['save_steps']),
        save_total_limit=int(cfg['save_total_limit']),
        metric_for_best_model=cfg['metric_for_best_model'],
        greater_is_better=bool(cfg['greater_is_better']),
        weight_decay=float(cfg['weight_decay']),
        run_name=cfg['wandb_run'],
        warmup_steps=cfg['warmup_steps'],
        report_to="wandb",
        auto_find_batch_size=True,
        gradient_checkpointing=cfg['gradient_checkpointing'],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        ddp_find_unused_parameters=False,
        ddp_timeout=3600,
        dataloader_pin_memory=False,
        use_liger_kernel=cfg['use_liger_kernel'],
        load_best_model_at_end=True,
        dataset_num_proc=6,
        dataloader_num_workers=6,
        **hf_kwargs,
    )
    #####################################
    logger = setup_logger()
    logger.info("Initializing DPO Trainer")
    return DPOTrainer(
        model=model,
        ref_model=None,
        args=tf_args,
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
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(hours=2))

    logger.info("Loaded config from %s", args.config)
    
    # after loading cfg...
    tokenizer = load_tokenizer(cfg['base_model'], cfg)
    train_dataset, eval_dataset = load_dpo_datasets(cfg)
    
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