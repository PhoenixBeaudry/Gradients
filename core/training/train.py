#!/usr/bin/env python3
import os
import argparse
import logging
from hpo_optuna import run_optuna 
import yaml
import torch
from axolotl.common.datasets import load_datasets
from axolotl.train import setup_model_and_tokenizer
from axolotl.cli.config import load_cfg
from axolotl.cli.args import TrainerCliArgs
from accelerate import Accelerator
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    SchedulerType,
)
import time
from transformers import TrainerCallback, TrainerControl, TrainerState
import bitsandbytes as bnb


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



def build_trainer(cfg: dict, model, tokenizer, processor, train_ds, eval_ds, callbacks):
    # ── SFT Trainer branch ────────────────────────────────────────
    tf_args = TrainingArguments(
        output_dir=cfg['output_dir'],
        bf16=bool(cfg['bf16']),
        gradient_accumulation_steps=int(cfg['gradient_accumulation_steps']),
        dataloader_num_workers=int(cfg['dataloader_num_workers']),
        num_train_epochs=int(cfg['num_epochs']),
        learning_rate=float(cfg['learning_rate']),
        optim=cfg['optimizer'],
        warmup_steps=int(cfg['warmup_steps']),
        lr_scheduler_type=SchedulerType.COSINE,
        max_steps=int(cfg['max_steps']),
        logging_steps=int(cfg['logging_steps']),
        eval_strategy='steps',
        save_strategy='best',
        eval_steps=int(cfg['eval_steps']),
        save_steps=int(cfg['save_steps']),
        save_total_limit=int(cfg['save_total_limit']),
        metric_for_best_model=cfg['metric_for_best_model'],
        greater_is_better=bool(cfg['greater_is_better']),
        weight_decay=float(cfg['weight_decay']),
        fp16=bool(cfg['fp16']),
        run_name=cfg['wandb_run'],
        hub_model_id=cfg['hub_model_id'],
        hub_token=cfg['hub_token'],
        hub_strategy='every_save',
        push_to_hub=True,
        use_liger_kernel=True,
        auto_find_batch_size=True,
        load_best_model_at_end=True,
    )
    logger = setup_logger()
    logger.info("Initializing SFT Trainer")
    return Trainer(
        model=model,
        args=tf_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
        processing_class=processor,
        callbacks=callbacks,
    )



accelerator = Accelerator(log_with="wandb", mixed_precision="bf16")
device = accelerator.device
args = parse_args()
cfg = load_config(args.config)
accelerator.init_trackers(cfg.get('wandb_project'), config=cfg)


def main():

    ### Temp Axolotl Config to generate Dataset and Model
    axo_config_alt = load_config(args.config)
    axo_config_alt["save_strategy"] = "steps"
    out_path = os.path.join(f"{args.config}_axo.yml")
    with open(out_path, "w") as f:
        yaml.dump(axo_config_alt, f)
    axo_cfg = load_cfg(f"{args.config}_axo.yml")
    #####################################################

    logger = setup_logger()
    
    # Performance flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logger.info("Loaded config from %s", args.config)
    
    # after loading cfg...
    dataset_meta = load_datasets(cfg=axo_cfg, cli_args=TrainerCliArgs())
    model, tokenizer, peft_config, processor = setup_model_and_tokenizer(cfg=axo_cfg)

    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    max_hours = int(cfg.get('hours_to_complete')) 

    callbacks = []
    if cfg.get('early_stopping', True):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.get('early_stopping_patience', 8))
        )
    
    if max_hours is not None:
        callbacks.append(TimeLimitCallback(max_hours*0.9))

    trainer = build_trainer(cfg, model, tokenizer, processor, train_dataset, eval_dataset, callbacks)

    logger.info("Starting Full Model Training...")

    trainer.train()



if __name__ == '__main__':
    main()