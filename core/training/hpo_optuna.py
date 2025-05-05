#!/usr/bin/env python3
"""
hpo_optuna.py  ·  1‑hour Optuna sweep → full training
────────────────────────────────────────────────────────
* Runs a 1‑hour hyper‑parameter search on a *slice* of the data.
* Logs every trial to a **separate WandB project** named
      "<original‑WANDB_PROJECT>-hpo".
* Disables any Hugging Face Hub push during trials.
* After HPO, writes <config>_opt.yml with the best params and
  launches the normal multi‑GPU training run via `accelerate`.
"""
from __future__ import annotations
import argparse, copy, gc, importlib.util, os, subprocess, tempfile, uuid, logging
from pathlib import Path
import yaml, optuna, torch
from optuna.pruners import HyperbandPruner
from accelerate import Accelerator
from axolotl.common.datasets import load_datasets
from axolotl.cli.args     import TrainerCliArgs
from axolotl.cli.config   import load_cfg

# ── bring in train.py helpers ────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("train", Path(__file__).with_name("train.py"))
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

# ── global logger config ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("hpo_optuna")

# ╭───────────────────────────── Hyper‑parameter search space ────────────────╮
def _sample_space(trial: optuna.Trial, cfg: dict) -> dict:
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
# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Objective (single Optuna trial) ────────────────╮
def _objective(trial: optuna.Trial, base_cfg: dict, hpo_project: str) -> float:
    """Run a *short* train+eval cycle and return eval_loss."""
    cfg = copy.deepcopy(base_cfg)
    cfg.update(_sample_space(trial, cfg))

    LOG.info("🔎  Starting trial %d with params: %s", trial.number, {k: cfg[k] for k in _sample_space(trial, cfg)})

    cfg["num_epochs"]        = 1
    cfg["hours_to_complete"] = 0.05
    trial_id                 = f"trial{trial.number}_{uuid.uuid4().hex[:4]}"
    cfg["output_dir"]        = str(Path(cfg.get("output_root", "./hpo_runs")) / trial_id)
    cfg["wandb_run"]         = f"{cfg.get('job_id', 'job')}_{trial_id}"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    cfg["push_to_hub"]  = False
    cfg["hub_strategy"] = "none"

    os.environ["WANDB_PROJECT"] = hpo_project

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_cfg_path = tmp.name

    axo_cfg   = load_cfg(tmp_cfg_path)
    data_meta = load_datasets(cfg=axo_cfg, cli_args=TrainerCliArgs())
    train_ds  = data_meta.train_dataset.select(range(min(1024, len(data_meta.train_dataset))))
    eval_ds   = data_meta.eval_dataset.select(range(min(256,  len(data_meta.eval_dataset))))
    tokenizer = train.load_tokenizer(axo_cfg)

    if any(k in cfg["base_model"].lower() for k in ("qwen", "mistral")):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    accelerator = Accelerator()
    model = train.load_model(cfg["base_model"], cfg)
    if cfg.get("adapter") == "lora":
        model = train.apply_lora_adapter(model, cfg)
    model = accelerator.prepare(model)

    trainer = train.build_trainer(cfg, model, tokenizer, train_ds, eval_ds, callbacks=[])
    trainer.accelerator = accelerator
    trainer.train()
    eval_loss = trainer.evaluate().get("eval_loss", float("inf"))

    LOG.info("✅  Trial %d completed – eval_loss: %.4f", trial.number, eval_loss)

    del model, trainer, tokenizer, train_ds, eval_ds
    gc.collect()
    torch.cuda.empty_cache()
    return eval_loss
# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────────── Optuna driver ───────────────────────────────╮
def run_optuna(config_path: str, timeout_hours: float = 1.0) -> tuple[dict, str]:
    LOG.info("🚀  Starting Optuna Hyperparameter Optimization.....")
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    base_project = os.environ.get("WANDB_PROJECT", "UnnamedProject")
    hpo_project  = f"{base_project}-hpo"
    os.environ["WANDB_PROJECT"] = hpo_project

    LOG.info("🚦  Beginning HPO sweep (project: %s, budget: %.1fh)…", hpo_project, timeout_hours)

    study = optuna.create_study(direction="minimize",
                                pruner=HyperbandPruner(min_resource=1, reduction_factor=3))
    study.optimize(lambda t: _objective(t, base_cfg, hpo_project),
                   timeout=int(timeout_hours * 3600),
                   show_progress_bar=True)

    LOG.info("🏆  HPO finished – best eval_loss %.5f with params %s",
             study.best_value, study.best_params)

    os.environ["WANDB_PROJECT"] = base_project
    return study.best_params, hpo_project
# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────────── helpers ─────────────────────────────────────╮
def _write_optimised_cfg(base_path: str, best: dict) -> str:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg.update(best)
    opt_path = base_path.replace(".yml", "_opt.yml")
    with open(opt_path, "w") as f:
        yaml.safe_dump(cfg, f)
    LOG.info("💾  Wrote optimised config → %s", opt_path)
    return opt_path

def _launch_training(acc_yaml: str, cfg_path: str):
    cmd = [
        "accelerate", "launch",
        "--config_file", acc_yaml,
        "--mixed_precision", "bf16",
        Path(__file__).with_name("train.py"),
        "--config", cfg_path,
    ]
    LOG.info("🚀  Starting full training run")
    subprocess.run(list(map(str, cmd)), check=True)
# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── entry‑point ───────────────────────────────────╮
def main():
    ap = argparse.ArgumentParser(description="1‑hour HPO then full training")
    ap.add_argument("--config",          required=True, help="Base YAML config file")
    ap.add_argument("--accelerate_yaml", required=True, help="accelerate.yaml for launch")
    ap.add_argument("--timeout_hours",   type=float, default=1.0, help="Wall‑clock HPO budget")
    args = ap.parse_args()

    best_params, _ = run_optuna(args.config, timeout_hours=args.timeout_hours)
    optimised_cfg  = _write_optimised_cfg(args.config, best_params)
    _launch_training(args.accelerate_yaml, optimised_cfg)

if __name__ == "__main__":
    main()
