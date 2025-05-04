# hpo_optuna.py
"""
One‑hour Optuna sweep for causal‑LM fine‑tuning.

Public API
----------
run_optuna(...)  →  dict
"""

from __future__ import annotations
import os
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from functools import partial
from typing import Tuple, Dict, Callable, Any
from datasets import Dataset

# -------------------------------------------------------------------
# Search‑space
# -------------------------------------------------------------------
def sample_hparams(trial: optuna.Trial, _cfg: dict) -> Dict[str, Any]:
    return {
        # optimiser & scheduler
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        "warmup_steps":  trial.suggest_int("warmup_steps", 0, 500),
        "optimizer":     trial.suggest_categorical(
            "optimizer", ["adamw_torch_fused", "lion_8bit", "paged_adamw_8bit"]
        ),

        # batch‑related
        "gradient_accumulation_steps": trial.suggest_int("ga_steps", 1, 8),
        "micro_batch_size":            trial.suggest_int("micro_bs", 2, 32),

        # LoRA params (ignored if adapter≠lora)
        "lora_r":       trial.suggest_int("lora_r", 4, 64),
        "lora_alpha":   trial.suggest_int("lora_alpha", 8, 128),
        "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.15),

        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.2),
    }

# -------------------------------------------------------------------
# Objective
# -------------------------------------------------------------------
def _objective(
    trial,
    base_cfg,
    dataset_pair,
    tokenizer,
    logger,
    load_model_fn,
    apply_adapter_fn,
    build_trainer_fn,
):
    import wandb  # local import keeps wandb optional for other use‑cases

    cfg = {**base_cfg, **sample_hparams(trial, base_cfg)}

    # HPO‑specific overrides
    cfg.update(
        num_epochs     = 1,
        max_steps      = min(int(base_cfg.get("hpo_max_steps", 300)),
                             int(base_cfg.get("max_steps", 10_000))),
        save_strategy  = "no",
        logging_steps  = 0,
        push_to_hub    = False,
        hub_strategy   = "never",
        wandb_run      = f"trial_{trial.number}",
        wandb_project  = f"{base_cfg.get('wandb_project', 'project')}-hpo",
        report_to      = ["wandb"],     # Transformers → active run
    )

    # start a standalone W&B run for this trial
    run = wandb.init(
        project = cfg["wandb_project"],
        name    = cfg["wandb_run"],
        config  = cfg,
        reinit  = True,
    )

    train_ds, eval_ds = dataset_pair
    model = load_model_fn(cfg["base_model"], cfg)
    if cfg.get("adapter") == "lora":
        model = apply_adapter_fn(model, cfg)

    trainer = build_trainer_fn(cfg, model, tokenizer, train_ds, eval_ds, callbacks=[])
    trainer.train()
    eval_loss = trainer.evaluate().get("eval_loss", float("inf"))

    # log final metric and close the run
    run.log({"eval_loss": eval_loss})
    wandb.finish()

    del trainer, model
    torch.cuda.empty_cache()
    return eval_loss

# -------------------------------------------------------------------
# Public entry
# -------------------------------------------------------------------
def run_optuna(
    cfg: dict,
    dataset_pair: Tuple[Dataset, Dataset],
    tokenizer,
    logger,
    timeout_sec: int,
    load_model_fn:    Callable,
    apply_adapter_fn: Callable,
    build_trainer_fn: Callable,
) -> dict:
    """
    Launch a time‑boxed Optuna sweep and return the best parameter dict.
    """
    logger.info("▶️  Optuna sweep started (timeout = %.1f min)…", timeout_sec / 60)

    study = optuna.create_study(
        direction = "minimize",
        sampler   = TPESampler(seed=cfg.get("seed", 42)),
        pruner    = HyperbandPruner(
            min_resource=1,
            max_resource=cfg.get("hpo_max_steps", 300),
        ),
    )

    study.optimize(
        partial(
            _objective,
            base_cfg         = cfg,
            dataset_pair     = dataset_pair,
            tokenizer        = tokenizer,
            logger           = logger,
            load_model_fn    = load_model_fn,
            apply_adapter_fn = apply_adapter_fn,
            build_trainer_fn = build_trainer_fn,
        ),
        timeout           = timeout_sec,
        n_jobs            = 1,
        show_progress_bar = True,
    )

    logger.info("✅  Optuna done. Best eval_loss = %.4f", study.best_value)
    logger.info("🏆  Best parameters: %s", study.best_params)
    return study.best_params
