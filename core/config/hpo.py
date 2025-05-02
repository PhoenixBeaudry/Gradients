#!/usr/bin/env python3

import os
import subprocess
import tempfile
import yaml
import shutil
import time
import wandb
import optuna
from optuna.pruners import HyperbandPruner
from optuna.exceptions import TrialPruned

#─── CONFIGURATION ───────────────────────────────────────────────────────────────
CONFIG_DIR = os.environ.get("CONFIG_DIR", "/workspace/axolotl/configs")
JOB_ID = os.environ["JOB_ID"]  # e.g. "my_job"
TEMPLATE = os.path.join(CONFIG_DIR, f"{JOB_ID}.yml")

MAX_EVAL_STEPS = int(os.environ.get("HPO_MAX_STEPS", 10))
PRUNER = HyperbandPruner(min_resource=1, max_resource=MAX_EVAL_STEPS, reduction_factor=3)

BASE_WANDB_PROJ = os.environ.get("WANDB_PROJECT", "Gradients-On-Demand")
HPO_WANDB_PROJ = os.environ.get("HPO_WANDB_PROJECT", BASE_WANDB_PROJ + "-hpo")

#─── HELPERS ─────────────────────────────────────────────────────────────────────
def run_trial(cfg_path: str, trial_num: int) -> float:
    """
    Run a short, single-trial Axolotl training and return the final eval_loss.
    """
    trial_out = f"./outputs/trial_{trial_num}"
    os.makedirs(trial_out, exist_ok=True)
    try:
        cmd = [
            "axolotl", "train", cfg_path,
            "--max-steps", str(MAX_EVAL_STEPS),
            "--output-dir", trial_out,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        _, _ = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Trial #{trial_num} failed (exit {proc.returncode})")

        # Fetch eval_loss from W&B instead of parsing logs
        api = wandb.Api()
        entity = os.environ.get("WANDB_ENTITY")
        project_path = f"{entity}/{HPO_WANDB_PROJ}" if entity else HPO_WANDB_PROJ
        run_name = f"trial_{trial_num}"

        run = None
        for _ in range(12):  # up to ~1 minute polling
            runs = api.runs(project_path, filters={"display_name": run_name})
            if runs:
                run = runs[0]
                break
            time.sleep(5)

        if run is None:
            raise RuntimeError(f"No W&B run named {run_name} found in {project_path}")

        loss = (run.summary_metrics.get("eval_loss") or
                run.summary.get("eval_loss"))
        if loss is None:
            raise RuntimeError(f"No eval_loss in summary for run {run_name}")
        return float(loss)
    finally:
        shutil.rmtree(trial_out, ignore_errors=True)

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
        "eval_steps": 2,
        "max_steps": MAX_EVAL_STEPS,
        "warmup_steps": 2,
        "logging_steps": 2,
        # disable HF push
        "hub_strategy": "none",
        "hub_model_id": "",
        "hub_token": "",
        # separate W&B project & run name
        "wandb_project": HPO_WANDB_PROJ,
        "wandb_run_name": f"trial_{trial.number}",
        "run_name":        f"trial_{trial.number}",
    })

    fd, tmp_path = tempfile.mkstemp(suffix=".yml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f)

        try:
            loss = run_trial(tmp_path, trial.number)
        except RuntimeError as e:
            trial.set_user_attr("axolotl_error", str(e))
            print(f"Trial Pruned due to error: {e}")
            raise TrialPruned()

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

    with open(TEMPLATE, "r") as f:
        base_cfg = yaml.safe_load(f)
    hours = base_cfg.get("hours_to_complete")
    if hours is None:
        raise ValueError(
            f"'hours_to_complete' not found in your base config ({TEMPLATE})."
        )

    time_budget_sec = (hours / 10) * 3600
    print(f"Running HPO for up to {time_budget_sec:.0f}s ({hours/10:.2f}h) ...")

    study.optimize(objective, timeout=time_budget_sec)

    print(">> Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"   • {k} = {v}")

    with open(TEMPLATE, "r") as f:
        best_cfg = yaml.safe_load(f)
    best_cfg.update(study.best_trial.params)

    out_path = os.path.join(CONFIG_DIR, f"{JOB_ID}_best.yml")
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f)
    print(f"Wrote best config to {out_path}")
