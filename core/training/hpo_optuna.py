#!/usr/bin/env python3
"""
hpo_optuna.py  –  1‑hour Optuna sweep → full training (multi‑GPU compatible)

Usage:
    python hpo_optuna.py \
        --config          /workspace/configs/my_job.yml \
        --accelerate_yaml /workspace/configs/accelerate.yaml \
        --timeout_hours   1
"""
from __future__ import annotations
import argparse, copy, json, logging, os, shutil, subprocess, tempfile, uuid
from pathlib import Path
import yaml, optuna
from optuna.pruners import HyperbandPruner

# ───────────────────────── global logging ──────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("hpo_optuna")

# ╭──────────────────────── Hyper‑parameter space ───────────────────╮
def sample_space(trial: optuna.Trial, cfg: dict) -> dict:
    """Optuna search‑space; tweak as desired."""
    params = {
        "learning_rate":               trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        "micro_batch_size":            trial.suggest_categorical("micro_batch_size", [2, 4, 8, 16, 32]),
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 8),
        "weight_decay":                trial.suggest_float("weight_decay", 0.0, 0.1),
    }
    if cfg.get("adapter") == "lora":
        params |= {
            "lora_r":       trial.suggest_int("lora_r", 4, 64),
            "lora_alpha":   trial.suggest_int("lora_alpha", 4, 128),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.15),
        }
    return params
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── Objective (single trial) ────────────────╮
def objective(trial: optuna.Trial,
              base_cfg: dict,
              acc_yaml: str,
              hpo_project: str) -> float:
    """Launch one accelerate‑based training subprocess, return eval_loss."""
    # 1) Build trial‑specific config -----------------------------------------
    cfg         = copy.deepcopy(base_cfg)
    trial_params = sample_space(trial, cfg)
    cfg.update(trial_params)

    trial_id    = f"trial{trial.number}_{uuid.uuid4().hex[:4]}"
    cfg["output_dir"]        = str(Path(cfg.get("output_root", "./hpo_runs")) / trial_id)
    cfg["wandb_run"]         = f"{cfg.get('job_id', 'job')}_{trial_id}"
    cfg["num_epochs"]        = 1               # speedy
    cfg["hours_to_complete"] = 0.1            # ~6 minutes via TimeLimitCallback
    cfg["hpo_run"] = True

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Write temp YAML ---------------------------------------------------------
    tmp_cfg = Path(tempfile.mkdtemp()) / f"{trial_id}.yml"
    with tmp_cfg.open("w") as f:
        yaml.safe_dump(cfg, f)

    LOG.info("🔎  Starting trial %d with params: %s", trial.number, trial_params)

    # 2) Launch training ------------------------------------------------------
    cmd = [
        "accelerate", "launch",
        "--config_file", acc_yaml,
        "--mixed_precision", "bf16",
        Path(__file__).with_name("train.py"),
        "--config", str(tmp_cfg),
    ]
    env = os.environ.copy()
    env["WANDB_PROJECT"] = hpo_project     # log to separate project

    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        LOG.warning("⚠️  Trial %d failed:\n%s", trial.number, e.stdout.decode("utf‑8", "ignore"))
        return float("inf")

    # 3) Parse eval_loss from trainer_state.json ------------------------------
    state_path = Path(cfg["output_dir"]) / "trainer_state.json"
    if not state_path.exists():
        LOG.warning("⚠️  trainer_state.json missing for trial %d", trial.number)
        return float("inf")

    try:
        with state_path.open() as f:
            state = json.load(f)
        # last log_history entry with eval_loss
        eval_loss = next(
            x["eval_loss"] for x in reversed(state["log_history"]) if "eval_loss" in x
        )
    except Exception as err:  # noqa: BLE001
        LOG.warning("⚠️  Could not parse eval_loss for trial %d: %s", trial.number, err)
        return float("inf")

    LOG.info("✅  Trial %d completed – eval_loss: %.4f", trial.number, eval_loss)
    # Be a good citizen: tidy up temp config dir
    shutil.rmtree(tmp_cfg.parent, ignore_errors=True)
    return eval_loss
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── Run Optuna sweep ────────────────────────╮
def run_optuna(base_cfg_path: str,
               acc_yaml: str,
               timeout_hours: float = 1.0) -> dict:
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    base_project = os.environ.get("WANDB_PROJECT", "UnnamedProject")
    hpo_project  = f"{base_project}-hpo"

    LOG.info("🚦  HPO sweep starting  (project: %s, budget: %.1fh)…", hpo_project, timeout_hours)

    study = optuna.create_study(direction="minimize",
                                pruner=HyperbandPruner(min_resource=1, reduction_factor=3))

    study.optimize(
        lambda t: objective(t, base_cfg, acc_yaml, hpo_project),
        timeout=int(timeout_hours * 3600),
        show_progress_bar=True,
    )

    LOG.info("🏆  HPO finished – best eval_loss %.5f with params %s",
             study.best_value, study.best_params)
    return study.best_params
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────── Write optimised YAML & launch main run ──────╮
def write_optimised_cfg(base_path: str, best: dict) -> str:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg.update(best)
    opt_path = base_path.replace(".yml", "_opt.yml")
    with open(opt_path, "w") as f:
        yaml.safe_dump(cfg, f)
    LOG.info("💾  Wrote optimised config → %s", opt_path)
    return opt_path

def launch_training(acc_yaml: str, cfg_path: str):
    cmd = [
        "accelerate", "launch",
        "--config_file", acc_yaml,
        "--mixed_precision", "bf16",
        Path(__file__).with_name("train.py"),
        "--config", cfg_path,
    ]
    LOG.info("🚀  Starting full training run")
    subprocess.run(cmd, check=True)
# ╰──────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── CLI entry‑point ─────────────────────╮
def main():
    ap = argparse.ArgumentParser(description="1‑hour HPO then full training")
    ap.add_argument("--config",          required=True, help="Base YAML config file")
    ap.add_argument("--accelerate_yaml", required=True, help="accelerate.yaml for launch")
    ap.add_argument("--timeout_hours",   type=float, default=1.0, help="Wall‑clock HPO budget")
    args = ap.parse_args()

    best_params  = run_optuna(args.config, args.accelerate_yaml, timeout_hours=args.timeout_hours)
    optimised_cfg = write_optimised_cfg(args.config, best_params)
    launch_training(args.accelerate_yaml, optimised_cfg)

if __name__ == "__main__":
    main()
