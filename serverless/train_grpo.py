#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import importlib
import sys
import inspect
import torch
from datetime import datetime
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import RewardFunc
from transformers import (
    EarlyStoppingCallback,
    SchedulerType,
    AutoModelForCausalLM
)
import time
from transformers import TrainerCallback, TrainerControl, TrainerState, AutoTokenizer
import optuna
import bitsandbytes as bnb
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




CONFIG_DIR = os.path.abspath("/workspace/configs/")

##### Custom Funcs for getting GRPO reward functions #####
def reward_functions(cfg):
    """
    Collects and returns a list of functions for GRPOTrainer.
    """
    funcs = []
    for fqn in cfg['trl']['reward_funcs']:
        funcs.append(get_reward_func(fqn))
    return funcs


def get_reward_func(reward_func_fqn: str) -> RewardFunc | str:
    """
    Try to load <module>.py from CONFIG_DIR and return its <func>.
    If the file doesn’t exist, just return the original string (HF model path).
    """
    module_name, func_name = reward_func_fqn.rsplit(".", 1)
    module_path = os.path.join(CONFIG_DIR, f"{module_name}.py")
    print(f"→ looking for {module_name!r} at {module_path!r}, exists? {os.path.isfile(module_path)}")
    # 1) if we have an on-disk file, dynamically import it
    if os.path.isfile(module_path):
        # drop any cached module so we always load the newest version
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # get the function
        if not hasattr(module, func_name):
            raise AttributeError(
                f"Module {module_name!r} has no attribute {func_name!r}"
            )
        reward_func = getattr(module, func_name)

        # sanity check signature
        sig = inspect.signature(reward_func)
        if len(sig.parameters) < 2:
            raise ValueError(
                "Reward function must accept at least two arguments: "
                "prompts: list and completions: list"
            )

        return reward_func

    # 2) otherwise fall back to treating the FQN string as a model-path
    return reward_func_fqn


############################################


def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal LM with SFT or DPO or GRPO")
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
    if any(k in model_name.lower() for k in ("qwen", "phi")):
        return AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True, device_map=device_map, torch_dtype=torch.bfloat16)
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True, device_map=device_map, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    except:
        return AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True, device_map=device_map, torch_dtype=torch.bfloat16)

        
def apply_lora_adapter(model: AutoModelForCausalLM, cfg: dict) -> AutoModelForCausalLM:
    if get_peft_model is None:
        raise ImportError("peft library is required for LoRA adapters.")

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

def load_tokenizer(model_name: str, cfg: dict):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=cfg.get("hub_token"),
        trust_remote_code=True,
        padding_side="left",          # DPO prefers left‑padding
        truncation_side="right"
    )
    if tok.pad_token_id is None:      # e.g. Llama‑3, Qwen‑2 FlashAttn
        tok.pad_token = tok.eos_token
    tok.add_eos_token = True
    tok.truncation = True
    return tok

def load_grpo_datasets(cfg: dict):
    """
    Return (train_ds, eval_ds) ready for TRL‑DPO.
    If cfg["val_set_size"] is 0 → eval_ds is None.
    """
    # Load **only one** split so we always get a Dataset, never a DatasetDict
    ds_train = load_dataset(
        "json",
        data_files=cfg["datasets"][0]["path"],
        split="train"          # guarantees Dataset, not DatasetDict
    )

    # Standardise column names
    ds_train = ds_train.rename_columns({
        cfg["datasets"][0]["field_prompt"]:   "prompt",
    })

    # Optional random split
    val_size = cfg.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None

    return train_ds, eval_ds


def build_trainer(cfg: dict, model, tokenizer, train_ds, eval_ds):
    # ── GRPO Trainer ────────────────────────────────────────
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
            'hub_strategy': "end",
            'push_to_hub': True,
        }
    tf_args = GRPOConfig(
        output_dir=cfg['output_dir'],
        # GRPO params
        max_completion_length=int(cfg["trl"]["max_completion_length"]),
        reward_weights=cfg["trl"]["reward_weights"],
        use_vllm=cfg["trl"]["use_vllm"],
        num_generations=int(cfg["trl"]["num_generations"]),
        #####
        gradient_accumulation_steps=int(cfg['gradient_accumulation_steps']),
        per_device_train_batch_size=int(cfg['micro_batch_size']),
        per_device_eval_batch_size=int(cfg['micro_batch_size']),
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        use_liger_kernel=True,
        load_best_model_at_end=True,
        dataloader_num_workers=16,
        **hf_kwargs,
    )
    #####################################
    logger = setup_logger()
    logger.info("Initializing GRPO Trainer")
    return GRPOTrainer(
        model=model,
        args=tf_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=reward_functions(cfg),
        processing_class=tokenizer,
        callbacks=callbacks,
    )



def main():
    args = parse_args()
    cfg = load_config(args.config)
    #####################################################

    logger = setup_logger()
    
    # Performance flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logger.info("Loaded config from %s", args.config)
    
    # after loading cfg...
    train_dataset, eval_dataset = load_grpo_datasets(cfg)

    tokenizer = load_tokenizer(cfg['base_model'], cfg)

    model = load_model(cfg['base_model'], cfg)
    
    if cfg.get('adapter') == 'lora':
        model = apply_lora_adapter(model, cfg)

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
        SUBSET_FRAC   = 0.05          # 5 %
        MIN_PAIRS     = 1_000         # never go below this
        MAX_PAIRS     = 10_000        # never go above this
        target_train = int(max(MIN_PAIRS, min(MAX_PAIRS, len(train_dataset) * SUBSET_FRAC)))
        target_eval = int(max(MIN_PAIRS, min(MAX_PAIRS, len(eval_dataset) * SUBSET_FRAC)))

        # No lower than dataset size
        target_train = min(target_train, len(train_dataset))
        target_eval = min(target_eval, len(eval_dataset))
        
        # deterministic shuffle → reproducible trials
        train_dataset = train_dataset.shuffle(seed=42).select(range(target_train))
        eval_dataset  = eval_dataset .shuffle(seed=42).select(range(target_eval))

    trainer = build_trainer(cfg, model, tokenizer, train_dataset, eval_dataset)

    logger.info("Starting Full Model Training...")

    trainer.train()
    if not cfg["hpo_run"]:
        trainer.push_to_hub()



if __name__ == '__main__':
    main()