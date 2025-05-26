#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import aiohttp
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    SchedulerType,
)
import time
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import TrainerCallback, TrainerControl, TrainerState, AutoTokenizer
import optuna
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
            print(f"\nReached time limit of {self.max_seconds/3600:.2f}h — stopping training.")
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
    try:
        if "phi-3-mini" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    except:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

    model.config.use_cache = False
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None
    model.train()

    return model



def load_tokenizer(model_name: str, cfg: dict):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=cfg.get("hub_token"),
        trust_remote_code=True,
        padding_side="left", 
        truncation_side="right"
    )
    if tok.pad_token_id is None:      # e.g. Llama‑3, Qwen‑2 FlashAttn
        tok.pad_token = tok.eos_token
    tok.add_eos_token = True
    tok.truncation = True
    return tok
        


def get_lora_adapter(model: AutoModelForCausalLM, cfg: dict) -> AutoModelForCausalLM:
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
    return peft_config




def load_sft_datasets(cfg: dict):
    """
    Return (train_ds, eval_ds)
    If cfg["val_set_size"] is 0 → eval_ds is None.
    """
    # Load **only one** split so we always get a Dataset, never a DatasetDict
    ds_train = load_dataset(
        "json",
        data_files=cfg["datasets"][0]["path"],
        split="train",          # guarantees Dataset, not DatasetDict
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
    )

    def combine_prompt(example):
        # Handles the case when "input" (the context) may be empty
        if example["input"]:
            prompt = f"{example['prompt']}\n{example['input']}"
        else:
            prompt = example["prompt"]
        example["prompt"] = prompt
        return example                  
    
    if not cfg["datasets"][0]["field_input"] == None:
        # Standardise column names
        ds_train = ds_train.rename_columns({
            cfg["datasets"][0]["field_instruction"]:   "prompt",
            cfg["datasets"][0]["field_input"]:   "input",
            cfg["datasets"][0]["field_output"]:   "completion",
        })
        ds_train = ds_train.map(combine_prompt)
    else:
        # Standardise column names
        ds_train = ds_train.rename_columns({
            cfg["datasets"][0]["field_instruction"]:   "prompt",
            cfg["datasets"][0]["field_output"]:   "completion",
        })

    
    # Optional random split
    val_size = cfg.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None
    
    return train_ds, eval_ds



def build_trainer(cfg: dict, model, peft_config, tokenizer, train_ds, eval_ds):
    # ── SFT Trainer ────────────────────────────────────────
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
    tf_args = SFTConfig(
        output_dir=cfg['output_dir'],
        gradient_accumulation_steps=int(cfg['gradient_accumulation_steps']),
        per_device_train_batch_size=int(cfg['micro_batch_size']),
        per_device_eval_batch_size=int(cfg['micro_batch_size']),
        max_steps=int(cfg['max_steps']),
        learning_rate=float(cfg['learning_rate']),
        optim=cfg['optimizer'],
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
        dataloader_pin_memory=False,
        use_liger_kernel=cfg['use_liger_kernel'],
        load_best_model_at_end=True,
        packing=cfg['packing'],
        eval_packing=cfg['packing'],
        dataset_num_proc=6,
        dataloader_num_workers=6,
        **hf_kwargs,
    )
    #####################################
    logger = setup_logger()
    logger.info("Initializing SFT Trainer")
    return SFTTrainer(
        model=model,
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
    
    logger.info("Loaded config from %s", args.config)
    
    # after loading cfg...
    tokenizer = load_tokenizer(cfg['base_model'], cfg)
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