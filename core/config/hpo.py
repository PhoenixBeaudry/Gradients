#!/usr/bin/env python3
# hpo_fixed.py

import os
import subprocess
import tempfile
import yaml
import re

import optuna
from optuna.pruners import HyperbandPruner
from optuna.exceptions import TrialPruned

#─── CONFIGURATION ───────────────────────────────────────────────────────────────
CONFIG_DIR = os.environ.get("CONFIG_DIR", "/workspace/axolotl/configs")
JOB_ID = os.environ["JOB_ID"]  # e.g. "my_job"
TEMPLATE = os.path.join(CONFIG_DIR, f"{JOB_ID}.yml")

N_TRIALS = int(os.environ.get("HPO_TRIALS", 20))
MAX_EVAL_STEPS = int(os.environ.get("HPO_MAX_STEPS", 20))
PRUNER = HyperbandPruner(min_resource=1, max_resource=MAX_EVAL_STEPS, reduction_factor=3)

BASE_WANDB_PROJ = os.environ.get("WANDB_PROJECT", "Gradients-On-Demand")
HPO_WANDB_PROJ = os.environ.get("HPO_WANDB_PROJECT", BASE_WANDB_PROJ + "-hpo")

#─── HELPERS ─────────────────────────────────────────────────────────────────────
def run_trial(cfg_path: str, trial_num: int) -> float:
    """
    Run a short, single-trial Axolotl training and return the final eval_loss.
    """
    # each trial writes to its own output dir to avoid collisions
    trial_out = f"./outputs/trial_{trial_num}"
    os.makedirs(trial_out, exist_ok=True)

    cmd = [
        "axolotl", "train", cfg_path,
        "--max_steps", str(MAX_EVAL_STEPS),
        "--output_dir", trial_out,
        "--no_push_hf"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # crash or other failure
    if proc.returncode != 0:
        raise RuntimeError(f"Trial {trial_num} failed:\n{proc.stderr}")

    logs = proc.stdout + proc.stderr

    # parse the last "{'eval_loss': X.XXXX, ...}" line
    pattern = re.compile(r"'eval_loss'\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    for line in reversed(logs.splitlines()):
        m = pattern.search(line)
        if m:
            return float(m.group(1))

    # if we get here, parsing failed
    raise RuntimeError(f"Could not parse eval_loss from logs of trial {trial_num}:\n{logs}")

#─── OBJECTIVE ───────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    # 1) sample HPO space
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    mb = trial.suggest_categorical("micro_batch_size", [4, 8, 16, 32])
    ga = trial.suggest_int("gradient_accumulation_steps", 1, 8)
    lora_r = trial.suggest_int("lora_r", 4, 64)
    lora_alpha = trial.suggest_int("lora_alpha", 8, 128)
    dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)

    # 2) load & patch base YAML
    with open(TEMPLATE, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.update({
        "learning_rate": lr,
        "micro_batch_size": mb,
        "gradient_accumulation_steps": ga,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": dropout,
        # disable HF push
        "hub_strategy": "none",
        "hub_model_id": "",
        "hub_token": "",
        # separate W&B project
        "wandb_project": HPO_WANDB_PROJ,
    })

    # 3) dump to a temp YAML
    fd, tmp_path = tempfile.mkstemp(suffix=".yml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f)

        # 4) run & report
        try:
            loss = run_trial(tmp_path, trial.number)
        except RuntimeError as e:
            trial.set_user_attr("axolotl_error", str(e))  
            # this will be recorded as a pruned trial rather than an outright failure
            raise optuna.exceptions.TrialPruned()
        
        trial.report(loss, step=MAX_EVAL_STEPS)

        if trial.should_prune():
            raise TrialPruned()

        return loss

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

#─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        pruner=PRUNER,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print(">> Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"   • {k} = {v}")

    # write best config
    with open(TEMPLATE, "r") as f:
        best_cfg = yaml.safe_load(f)
    best_cfg.update(study.best_trial.params)
    best_cfg["hub_strategy"] = "checkpoint"
    best_cfg["wandb_project"] = BASE_WANDB_PROJ

    out_path = os.path.join(CONFIG_DIR, f"{JOB_ID}_best.yml")
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f)
    print(f"Wrote best config to {out_path}")
