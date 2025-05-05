#!/usr/bin/env python3
import os
import argparse
import logging
from hpo_optuna import run_optuna 
import yaml
import torch
from axolotl.common.datasets import load_datasets
from axolotl.utils.models import load_tokenizer
from axolotl.cli.config import load_cfg
from axolotl.cli.args import TrainerCliArgs
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    SchedulerType,
    AutoModelForCausalLM
)
import time
from transformers import TrainerCallback, TrainerControl, TrainerState
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

def load_model(model_name: str, cfg: dict) -> AutoModelForCausalLM:
    device_map = {"": torch.cuda.current_device()} 
    common_kwargs = {
        'use_auth_token': cfg.get('hub_token'),
        'load_in_8bit': bool(cfg.get('load_in_8bit', False)),
        'torch_dtype': torch.bfloat16,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='flash_attention_3',
            trust_remote_code=True,
            device_map=device_map,
            **common_kwargs
        )
    except:
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
        output_dir=cfg['output_dir'],
        gradient_accumulation_steps=int(cfg['gradient_accumulation_steps']),
        per_device_train_batch_size=int(cfg['micro_batch_size']),
        per_device_eval_batch_size=int(cfg['micro_batch_size']),
        dataloader_num_workers=int(cfg['dataloader_num_workers']),
        num_train_epochs=int(cfg['num_epochs']),
        learning_rate=float(cfg['learning_rate']),
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
        hub_model_id=cfg['hub_model_id'],
        hub_token=cfg['hub_token'],
        hub_strategy='every_save',
        report_to="wandb",
        warmup_ratio=0.08,
        auto_find_batch_size=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        push_to_hub=True,
        use_liger_kernel=True,
        load_best_model_at_end=True,
    )
    logger = setup_logger()
    logger.info("Initializing SFT Trainer")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,                     # passes label_pad_token_id=-100 automatically
        padding="longest",               # dynamic per mini‑batch
        pad_to_multiple_of=8,            # keeps TensorCores happy; optional
    )
    return Trainer(
        model=model,
        args=tf_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
        data_collator=data_collator
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
    dataset_meta = load_datasets(cfg=axo_cfg, cli_args=TrainerCliArgs())
    tokenizer = load_tokenizer(axo_cfg)
    if "Qwen" in cfg["base_model"]:
        print("Qwen detected: setting tokenizer padding to left =============")
        tokenizer.padding_side = "left"
    model = load_model(cfg['base_model'], cfg)
    if cfg.get('adapter') == 'lora':
        model = apply_lora_adapter(model, cfg)

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

    trainer = build_trainer(cfg, model, tokenizer, train_dataset, eval_dataset, callbacks)

    logger.info("Starting Full Model Training...")

    trainer.train()



if __name__ == '__main__':
    main()