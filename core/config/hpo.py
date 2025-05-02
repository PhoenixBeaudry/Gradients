#!/usr/bin/env python3
# hpo_fixed.py

import os
import subprocess
import tempfile
import yaml
from pathlib import Path
import re
import shutil
import optuna
from subprocess import TimeoutExpired
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


from optuna.exceptions import TrialPruned

LOG_PATTERN = re.compile(r"'eval_loss'\s*:\s*([0-9]+(?:\.[0-9]+)?)")

def _parse_loss(log_text: str) -> float:
    for line in reversed(log_text.splitlines()):
        m = LOG_PATTERN.search(line)
        if m:
            return float(m.group(1))
    raise RuntimeError(f"Could not parse eval_loss:\n{log_text!r}")

def run_trial(cfg: dict, trial_num: int) -> float:
    """
    1) Write a one-off temp YAML from `cfg`.  
    2) Create a trial workspace under ./outputs/trial_{trial_num}/.  
    3) Launch axolotl with separate train+eval step caps, streaming its stdout.  
    4) Enforce a hard `timeout` on the whole subprocess.  
    5) Parse out the final eval_loss and return it.  
    6) Clean up everything, no matter what.
    """
    # 1) Dump temp config
    tmp_cfg = tempfile.NamedTemporaryFile(suffix=".yml", delete=False)
    try:
        yaml.safe_dump(cfg, tmp_cfg)
        tmp_cfg.flush()
        tmp_path = tmp_cfg.name
    finally:
        tmp_cfg.close()

    # 2) Prepare trial dir
    trial_dir = Path("outputs") / f"trial_{trial_num}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "axolotl", "train", tmp_path,
        "--max_train_steps", str(cfg["eval_steps"]),
        "--max_eval_steps",  str(cfg["eval_steps"]),
        "--output-dir",      str(trial_dir),
    ]

    try:
        # 3) Launch & stream logs
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # 4) Wait up to `timeout` seconds for completion
        try:
            logs, _ = proc.communicate()
        except TimeoutExpired:
            proc.kill()
            raise TrialPruned(f"Trial #{trial_num} timed out after {timeout}s")

        if proc.returncode != 0:
            raise RuntimeError(f"Trial #{trial_num} failed (exit {proc.returncode})\n{logs}")

        # 5) Extract loss
        return _parse_loss(logs)

    finally:
        # 6) Cleanup both temp config and trial outputs
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        shutil.rmtree(trial_dir, ignore_errors=True)


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
            print(f"Trial Pruned due to error: {str(e)}")
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
    # 1) Create the study as before
    study = optuna.create_study(
        direction="minimize",
        pruner=PRUNER,
    )

    # 2) Load your base config to read hours_to_complete
    with open(TEMPLATE, "r") as f:
        base_cfg = yaml.safe_load(f)
    hours = base_cfg.get("hours_to_complete")
    if hours is None:
        raise ValueError(
            "'hours_to_complete' not found in your base config "
            f"({TEMPLATE})."
        )

    # 3) Compute a time budget: 1/10th of total hours, in seconds
    time_budget_sec = 600 #(hours / 10) * 3600
    print(f"Running HPO for up to {time_budget_sec:.0f}s "
          f"({hours/10:.2f}h) ...")

    # 4) Optimize until timeout rather than trial-count
    study.optimize(objective, timeout=time_budget_sec)

    # 5) Report best trial as before
    print(">> Best hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"   • {k} = {v}")

    # 6) Write out the best config
    with open(TEMPLATE, "r") as f:
        best_cfg = yaml.safe_load(f)
    best_cfg.update(study.best_trial.params)

    out_path = os.path.join(CONFIG_DIR, f"{JOB_ID}_best.yml")
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f)
    print(f"Wrote best config to {out_path}")

