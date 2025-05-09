#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
from math import ceil
import torch
from datetime import datetime
from axolotl.common.datasets import load_preference_datasets
from trl import DPOConfig, DPOTrainer
from axolotl.utils.models import load_tokenizer
from axolotl.cli.config import load_cfg
from axolotl.cli.args import TrainerCliArgs
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    SchedulerType,
    AutoModelForCausalLM
)
import time
from transformers import TrainerCallback, TrainerControl, TrainerState
import optuna
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Disable parallel tokenizer threads to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


###### Custom Callbacks ########################

class TimeLimitCallback(TrainerCallback):
    """Stop training after a fixed number of hours."""

    def __init__(self, max_seconds: float):
        """
        Args:
            max_hours: training time budget in hours
        """
        self.max_seconds = max_seconds
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
    
class OptunaPruningCallback(TrainerCallback):
    """
    Reports ``eval_loss`` back to Optuna at every evaluation and raises
    ``optuna.TrialPruned`` when the trial should stop early.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "eval_loss"):
        self._trial = trial
        self._monitor = monitor

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self._monitor not in metrics:
            return  # nothing to report
        step = state.global_step
        value = float(metrics[self._monitor])
        # Send metric to Optuna
        self._trial.report(value, step)
        # Ask Optuna whether the trial should be pruned
        if self._trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at step {step}: {self._monitor}={value}"
            )
        
def add_optuna_callback_if_needed(callbacks: list[TrainerCallback]):
    storage_url = os.getenv("OPTUNA_STORAGE")
    study_name  = os.getenv("OPTUNA_STUDY_NAME")
    trial_id    = os.getenv("OPTUNA_TRIAL_ID")
    if not (storage_url and study_name and trial_id):
        return  # not an HPO child

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    trial  = optuna.trial.Trial(study, trial_id=int(trial_id))
    callbacks.append(OptunaPruningCallback(trial, monitor="eval_loss"))

#######################################################



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

def load_model(model_name: str, cfg: dict) -> AutoModelForCausalLM:
    device_map = {"": torch.cuda.current_device()} 
    common_kwargs = {
        'use_auth_token': cfg.get('hub_token'),
        'load_in_8bit': bool(cfg.get('load_in_8bit', False)),
        'torch_dtype': torch.bfloat16,
    }
    if any(k in cfg["base_model"].lower() for k in ("qwen")):
        return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map=device_map, **common_kwargs)
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='flash_attention_2',
            trust_remote_code=True,
            device_map=device_map,
            **common_kwargs
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map=device_map, **common_kwargs)


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
        lora_alpha=int(cfg.get('lora_r', 16))*2,
        target_modules=targets,
        lora_dropout=float(cfg.get('lora_dropout', 0.05)),
        bias='none',
        task_type='CAUSAL_LM'
    )
    return get_peft_model(model, peft_config)


def build_trainer(cfg: dict, model, tokenizer, train_ds, eval_ds):
    # ── SFT Trainer ────────────────────────────────────────
    #### Callbacks ####
    callbacks = []
    if cfg.get('early_stopping', True):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.get('early_stopping_patience', 8), early_stopping_threshold=1e-4)
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
            'hub_strategy': cfg['hub_strategy'],
            'push_to_hub': True,
        }
    tf_args = DPOConfig(
        output_dir=cfg['output_dir'],
        gradient_accumulation_steps=int(cfg['gradient_accumulation_steps']),
        per_device_train_batch_size=int(cfg['micro_batch_size']),
        per_device_eval_batch_size=int(cfg['micro_batch_size']),
        dataloader_num_workers=int(cfg['dataloader_num_workers']),
        max_steps=int(cfg['max_steps']),
        learning_rate=float(cfg['learning_rate']),
        beta=float(cfg['beta']),
        optim=cfg['optimizer'],
        lr_scheduler_type=SchedulerType.COSINE,
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
        use_liger_kernel=True,
        load_best_model_at_end=True,
        **hf_kwargs,
    )
    #####################################
    logger = setup_logger()
    logger.info("Initializing DPO Trainer")
    return DPOTrainer(
        model=model,
        args=tf_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )



def main():
    args = parse_args()
    cfg = load_config(args.config)

    ### Temp Axolotl Config to generate Dataset and Model
    axo_cfg = load_cfg(args.config)
    #####################################################

    logger = setup_logger()
    
    # Performance flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logger.info("Loaded config from %s", args.config)
    
    # after loading cfg...
    dataset_meta = load_preference_datasets(cfg=axo_cfg, cli_args=TrainerCliArgs())
    tokenizer = load_tokenizer(axo_cfg)

    if any(k in cfg["base_model"].lower() for k in ("qwen", "mistral", "starcode")):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    model = load_model(cfg['base_model'], cfg)
    if cfg.get('adapter') == 'lora':
        model = apply_lora_adapter(model, cfg)

    if not cfg["hpo_run"]:
        train_dataset = dataset_meta.train_dataset
        eval_frac = min(0.03, 10_000 / len(train_dataset))   # 3 % or 10 k, whichever smaller
        eval_dataset   = train_dataset.shuffle(seed=42).select(range(int(eval_frac * len(train_dataset))))
    else:
        # ── HPO trial: auto‑subset the corpus ───────────────────────────────────
        # 1. compute target subset sizes
        n_train = len(dataset_meta.train_dataset)
        target_train = min(
            50_000,
            max(1_024, ceil(n_train * 0.02))
        )

        n_eval = len(dataset_meta.eval_dataset)
        target_eval = min(
            max(512, ceil(target_train * 0.25)),
            n_eval                                    # never exceed full eval set
        )

        # 2. deterministic shuffle so every trial sees identical data
        train_subset = dataset_meta.train_dataset.shuffle(seed=42)
        eval_subset  = dataset_meta.eval_dataset.shuffle(seed=42)

        # 3. slice
        train_dataset = train_subset.select(range(target_train))
        eval_dataset  = eval_subset.select(range(target_eval))

    trainer = build_trainer(cfg, model, tokenizer, train_dataset, eval_dataset)

    logger.info("Starting Full Model Training...")

    trainer.train()



if __name__ == '__main__':
    main()