#!/usr/bin/env python3
"""
hpo_optuna.py  –  1‑hour Optuna sweep → full training (multi‑GPU compatible)
--------------------------------------------------------------------------
* Trials log to <WANDB_PROJECT>-hpo and never push to Hugging Face.
* eval_loss is extracted (in this order):
    1) wandb-summary.json   2) stdout regex   3) trainer_state.json
"""
from __future__ import annotations
import argparse, copy, json, logging, os, re, shutil, subprocess, tempfile, uuid, time
from pathlib import Path
import yaml, optuna
from datetime import datetime, timedelta
from optuna.pruners import HyperbandPruner
from optuna.storages import RDBStorage      

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("hpo_optuna")

MAX_TRIALS_TO_RUN = 20
TRIAL_MAX_STEPS = 180
TRIAL_EVAL_STEPS = 30
TIMEOUT_PERCENTAGE_OF_TOTAL = 0.20
MAX_MINUTES_PER_TRIAL = 40
                   

# ╭──────────────────────── Hyper‑parameter space ───────────────────────────╮
def sample_space(trial: optuna.Trial, cfg: dict) -> dict:
    if cfg["rl"] == "dpo" or cfg["rl"] == "grpo":
        params = {
            "learning_rate":               trial.suggest_float("learning_rate", 3e-7, 7e-6, log=True),
            "micro_batch_size":            trial.suggest_categorical("micro_batch_size", [4, 8, 16, 32, 64]),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8]),
            "weight_decay":                trial.suggest_float("weight_decay", 0.0, 0.05),
            "beta":                        trial.suggest_float("beta", 0.05, 0.5)
        }
    else:
        params = {
        "learning_rate":               trial.suggest_float("learning_rate", 3e-6, 7e-5, log=True),
        "micro_batch_size":            trial.suggest_categorical("micro_batch_size", [4, 8, 16, 32, 64]),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8]),
        "weight_decay":                trial.suggest_float("weight_decay", 0.0, 0.05),
    }
        
    if cfg["adapter"] == "lora":
        params |= {
            "lora_r":       trial.suggest_int("lora_r", 16, 1024),
            "lora_dropout":       trial.suggest_float("lora_dropout", 0.0, 0.2),
        }

    return params
# ╰──────────────────────────────────────────────────────────────────────────╯




# ╭────────────────── helpers for eval_loss extraction ───────────────────────╮
_EVAL_RE = re.compile(r"eval_loss[^0-9]*([0-9]+\.[0-9]+)")

def loss_from_wandb(out_dir: Path) -> float | None:
    p = out_dir / "wandb" / "latest-run" / "files" / "wandb-summary.json"
    if p.exists():
        with p.open() as f:
            js = json.load(f)
        if "eval_loss" in js:
            return float(js["eval_loss"])
    return None

def loss_from_stdout(stdout: str) -> float | None:
    matches = _EVAL_RE.findall(stdout)
    return float(matches[-1]) if matches else None

def loss_from_state(out_dir: Path) -> float | None:
    p = out_dir / "trainer_state.json"
    if not p.exists():
        return None
    with p.open() as f:
        js = json.load(f)
    for rec in reversed(js.get("log_history", [])):
        if "eval_loss" in rec:
            return float(rec["eval_loss"])
    return None
# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── Objective (single trial) ─────────────────────────╮
def objective(trial: optuna.Trial,
              base_cfg: dict,
              hpo_project: str,
              study_name: str,
              storage_path: str) -> float:
    cfg          = copy.deepcopy(base_cfg)
    trial_params = sample_space(trial, cfg)
    cfg.update(trial_params)

    trial_id     = f"trial{trial.number}"
    out_dir      = Path(cfg.get("output_root", "./hpo_runs")) / trial_id
    cfg |= {
        "output_dir":        str(out_dir),
        "wandb_run":         f"{cfg['job_id'][:5]}_{cfg['rl']}_{trial_id}",
        "wandb_project":     hpo_project,
        "max_steps":        TRIAL_MAX_STEPS,
        "eval_steps":       TRIAL_EVAL_STEPS,
        "save_steps": 300
    }
    cfg["hpo_run"] = True
    cfg["required_finish_time"] = (datetime.now() + timedelta(minutes=MAX_MINUTES_PER_TRIAL)).isoformat()

    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_cfg = Path(tempfile.mkdtemp()) / f"{trial_id}.yml"
    with tmp_cfg.open("w") as f:
        yaml.safe_dump(cfg, f)

    LOG.info("🔎  Starting trial %d with params: %s", trial.number, trial_params)
     # ── prepare environment for subprocess ──────────────────────────────────
    env = os.environ.copy()
    env["WANDB_PROJECT"] = hpo_project          # override globally
    env.pop("WANDB_RUN_ID",  None)              # avoid carry‑over
    env.pop("WANDB_NAME",    None)
    env["OPTUNA_STORAGE"]   = storage_path
    env["OPTUNA_STUDY_NAME"] = study_name
    env["OPTUNA_TRIAL_ID"]   = str(trial._trial_id)

    if cfg["rl"] == "dpo":
        path_to_train_file = "/workspace/training/train_dpo.py"
    elif cfg["rl"] == "grpo":
        cfg["trl"]["max_completion_length"] = 16
        path_to_train_file = "/workspace/training/train_grpo.py"
    else:
        path_to_train_file = "/workspace/training/train.py"

    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--use_deepspeed",
        "--zero_stage", "2",
        path_to_train_file,
        "--config", str(tmp_cfg),
    ]

    try:
        cp = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, check=True)
        stdout = cp.stdout
    except subprocess.CalledProcessError as e:
        LOG.warning("⚠️  Trial %d failed:\n%s", trial.number, e.stdout)
        LOG.info("⚠️  Waiting 3s before starting next trial for cleanup...")
        time.sleep(10)
        return float("inf")

    # ── extract eval_loss (3 fallback methods) ──────────────────────────────
    for extractor in (loss_from_wandb, lambda _: loss_from_stdout(stdout), loss_from_state):
        val = extractor(out_dir) if extractor is loss_from_wandb or extractor is loss_from_state else extractor(None)
        if val is not None:
            LOG.info("✅  Trial %d completed – eval_loss: %.4f", trial.number, val)
            shutil.rmtree(tmp_cfg.parent, ignore_errors=True)
            return val

    LOG.warning("⚠️  eval_loss not found for trial %d – penalising.", trial.number)
    return float("inf")
# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── Run Optuna sweep ─────────────────────────────────╮
def run_optuna(base_cfg_path: str) -> dict:
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    study_name   = base_cfg.get("job_id", "optuna")
    hpo_root     = Path(base_cfg.get("output_root", "./hpo_runs")) / study_name
    hpo_root.mkdir(parents=True, exist_ok=True)
    storage_path = f"sqlite:///{hpo_root / 'hpo.db'}"
    base_project = os.environ.get("WANDB_PROJECT", "Gradients")
    hpo_project  = f"{base_project}-HPO-Trials"

    LOG.info("🚦  HPO sweep starting  (project: %s)…", hpo_project)
    storage = RDBStorage(url=storage_path, engine_kwargs={"connect_args": {"timeout": 30}, "pool_pre_ping": True})

    if base_cfg["rl"] == "grpo":
        direction = "maximize"
    else:
        direction = "minimize"

    study = optuna.create_study(direction=direction,
                                study_name=base_cfg["job_id"],
                                load_if_exists=False,
                                storage=storage,
                                pruner=HyperbandPruner(min_resource=2, max_resource=int(TRIAL_MAX_STEPS/TRIAL_EVAL_STEPS), reduction_factor=3))
    
    # calculate how much time we have left for job:
    time_remaining = datetime.fromisoformat(base_cfg['required_finish_time']) - datetime.now()
    seconds_remaining = max(0.0, time_remaining.total_seconds())
    LOG.info(f"Time allocated to HPO Search {seconds_remaining/3600*TIMEOUT_PERCENTAGE_OF_TOTAL}")
    study.optimize(lambda t: objective(t, base_cfg, hpo_project, study_name, storage_path),
                   timeout=int(seconds_remaining * TIMEOUT_PERCENTAGE_OF_TOTAL),
                   n_trials=MAX_TRIALS_TO_RUN,
                   show_progress_bar=True)

    LOG.info("🏆  HPO finished – best eval_loss %.5f with params %s",
            study.best_value, study.best_params)
        
    return study.best_params
# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────── Write optimised YAML & launch main run ───────────────╮
def write_opt_cfg(base_cfg: str, best: dict) -> str:
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg.update(best)
    opt_path = base_cfg.replace(".yml", "_opt.yml")
    with open(opt_path, "w") as f:
        yaml.safe_dump(cfg, f)
    LOG.info("💾  Wrote optimised config → %s", opt_path)
    return opt_path

def launch_training(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if cfg["rl"] == "dpo":
        path_to_train_file = "/workspace/training/train_dpo.py"
    elif cfg["rl"] == "grpo":
        path_to_train_file = "/workspace/training/train_grpo.py"
    else:
        path_to_train_file = "/workspace/training/train.py"


    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--use_deepspeed",
        "--zero_stage", "2",
        path_to_train_file,
        "--config", cfg_path,
    ]

    LOG.info("🚀  Starting full training run")
    subprocess.run(cmd, check=True)
# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── CLI entry‑point ──────────────────────────────╮
def main():
    ap = argparse.ArgumentParser(description="HPO then full training")
    ap.add_argument("--config",          required=True, help="Base YAML config file")
    args = ap.parse_args()
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
        
    if base_cfg["do_hpo"] == False:
        launch_training(args.config)
        return
    
    best_params   = run_optuna(args.config)
    optimised_cfg = write_opt_cfg(args.config, best_params)
    time.sleep(10)
    launch_training(optimised_cfg)

if __name__ == "__main__":
    main()
