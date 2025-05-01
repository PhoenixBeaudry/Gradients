# hpo.py

import os
import subprocess
import tempfile
import yaml

import optuna
from optuna.pruners import HyperbandPruner
from optuna.exceptions import TrialPruned

#─── CONFIGURATION ───────────────────────────────────────────────────────────────

# Where your “base” Axolotl config lives
CONFIG_DIR = os.environ.get("CONFIG_DIR", "/workspace/axolotl/configs")
JOB_ID     = os.environ["JOB_ID"]  # e.g. “my_job”
TEMPLATE   = os.path.join(CONFIG_DIR, f"{JOB_ID}.yml")

# HPO settings
N_TRIALS         = int(os.environ.get("HPO_TRIALS", 20))
MAX_EVAL_STEPS   = int(os.environ.get("HPO_MAX_STEPS", 20))
PRUNER           = HyperbandPruner()

# W&B project base name
BASE_WANDB_PROJ  = os.environ.get("WANDB_PROJECT", "Gradients-On-Demand")
HPO_WANDB_PROJ   = os.environ.get("HPO_WANDB_PROJECT", BASE_WANDB_PROJ + "-hpo")

#─── HELPERS ─────────────────────────────────────────────────────────────────────

def run_trial(cfg_path: str, trial_num: int) -> float:
    """
    Run a short, single‐trial training and return the final eval_loss.
    """
    # each trial writes to its own output dir to avoid collisions
    trial_out = f"./outputs/trial_{trial_num}"
    cmd = (
        f"axolotl train {cfg_path} "
        f"--max_steps {MAX_EVAL_STEPS} "
        f"--output_dir {trial_out} "
        f"--no_push_hf"            # disable any HF pushing
    )
    # run and capture logs
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    logs = proc.stdout + proc.stderr

    # parse the last “eval_loss: X.XXXX”
    for line in reversed(logs.splitlines()):
        if "eval_loss" in line:
            return float(line.strip().split("eval_loss:")[-1])
    raise RuntimeError("Could not parse eval_loss from logs")

#─── OBJECTIVE ───────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    # 1) sample HPO space
    lr   = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    mb   = trial.suggest_categorical("micro_batch_size", [4, 8, 16, 32])
    ga   = trial.suggest_int("gradient_accumulation_steps", 1, 8)
    lora_r     = trial.suggest_int("lora_r", 4, 64)
    lora_alpha = trial.suggest_int("lora_alpha", 8, 128)
    dropout    = trial.suggest_uniform("lora_dropout", 0.0, 0.3)

    # 2) load & patch base YAML
    with open(TEMPLATE, "r") as f:
        cfg = yaml.safe_load(f)

    # override trial hyperparams
    cfg["learning_rate"]                = lr
    cfg["micro_batch_size"]             = mb
    cfg["gradient_accumulation_steps"]  = ga
    cfg["lora_r"]                       = lora_r
    cfg["lora_alpha"]                   = lora_alpha
    cfg["lora_dropout"]                 = dropout

    # disable any HF push during HPO
    cfg["hub_strategy"] = "none"
    cfg["hub_model_id"]  = ""
    cfg["hub_token"]     = ""

    # separate W&B project
    cfg["wandb_project"] = HPO_WANDB_PROJ

    # 3) dump to a temp YAML
    fd, tmp_path = tempfile.mkstemp(suffix=".yml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f)

    # 4) run & report
    loss = run_trial(tmp_path, trial.number)
    trial.report(loss, step=0)

    # 5) prune unpromising trials
    if trial.should_prune():
        raise TrialPruned()

    return loss

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

    # optionally write out best to a YAML for your final run
    best_cfg = yaml.safe_load(open(TEMPLATE))
    best_cfg.update(study.best_trial.params)
    best_cfg["hub_strategy"]   = "checkpoint"  # or your desired push strategy
    best_cfg["wandb_project"]  = BASE_WANDB_PROJ  # back to main project

    with open(os.path.join(CONFIG_DIR, f"{JOB_ID}_best.yml"), "w") as f:
        yaml.dump(best_cfg, f)
