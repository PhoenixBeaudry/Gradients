#!/usr/bin/env python3
"""
hpo_optuna.py  –  1‑hour Optuna sweep → full training (multi‑GPU compatible)
--------------------------------------------------------------------------
* Trials log to <WANDB_PROJECT>-hpo and never push to Hugging Face.
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
import gc
import torch
import signal
from contextlib import contextmanager
import threading
import psutil

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("hpo_optuna")

MAX_TRIALS_TO_RUN = 30
TRIAL_MAX_STEPS = 400
TRIAL_EVAL_STEPS = 80
TESTING_TRIAL_MAX_STEPS = 50
TESTING_TRIAL_EVAL_STEPS = 25
PERCENT_TIME_FOR_HPO = 0.30
MAX_MINUTES_PER_TRIAL = 30
GPU_CLEANUP_WAIT_TIME = 10  # seconds to wait for GPU cleanup

# ── Stability utilities ─────────────────────────────────────────────────────
def cleanup_resources():
    """Force cleanup of GPU memory and other resources"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        # Kill any zombie processes
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        time.sleep(2)  # Give processes time to terminate
        for child in current_process.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception as e:
        LOG.warning(f"Resource cleanup error: {e}")

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class ProcessMonitor:
    """Monitor subprocess and kill if it exceeds resource limits"""
    def __init__(self, proc, max_seconds, check_interval=5):
        self.proc = proc
        self.max_seconds = max_seconds
        self.check_interval = check_interval
        self.start_time = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()
    
    def _monitor(self):
        while not self._stop.is_set() and self.proc.poll() is None:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_seconds:
                LOG.warning(f"Process exceeded time limit ({self.max_seconds}s), terminating...")
                self.proc.terminate()
                time.sleep(5)
                if self.proc.poll() is None:
                    self.proc.kill()
                break
            time.sleep(self.check_interval)
    
    def stop(self):
        self._stop.set()
        self._thread.join()

# ╭──────────────────────── Hyper‑parameter space ───────────────────────────╮
def sample_space(trial: optuna.Trial, cfg: dict) -> dict:
    # Model variant params
    if cfg["model_params_count"]:
        model_params_count = cfg["model_params_count"] 
    else:
        model_params_count = 1_000_000_000

    # Invariant Params
    params = {
        "optimizer": trial.suggest_categorical("optimizer", ["adamw_8bit", "lion_8bit", "adamw_torch"]),
    }

    # DPO Params
    if cfg["rl"] == "dpo":
        params |= {
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
            "beta": trial.suggest_float("beta", 0.01, 0.5, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        }
    # GRPO Params
    elif cfg["rl"] == "grpo":
        params |= {
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
            "beta": trial.suggest_float("beta", 0.01, 0.3, log=True),
        }
    # SFT Params
    else:
        params |= {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.15),
        }
        
    # Always lora for models > 8b
    if model_params_count > 8_000_000_000:
        params |= {
            "adapter": trial.suggest_categorical("adapter", ["lora"]),
        }
    elif "bloomz" in cfg["base_model"].lower():
        params |= {
            "adapter": trial.suggest_categorical("adapter", ["None"]),
        }
    else:
        params |= {
            "adapter": trial.suggest_categorical("adapter", ["lora", "None"]),
        }

    # LORA Params
    if params["adapter"] == "lora":
        params |= {
            "lora_r": trial.suggest_int("lora_r", 16, 1024, step=16),
            "lora_alpha": trial.suggest_int("lora_alpha", 16, 1024, step=16),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.1),
        }

    return params
# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭────────────────── helpers for eval_loss extraction ───────────────────────╮
_EVAL_RE = re.compile(r"eval_loss[^0-9]*([0-9]+\.[0-9]+)")

def loss_from_wandb(out_dir: Path) -> float | None:
    """Extract loss from wandb with retry logic"""
    p = out_dir / "wandb" / "latest-run" / "files" / "wandb-summary.json"
    for attempt in range(3):
        if p.exists():
            try:
                with p.open() as f:
                    js = json.load(f)
                if "eval_loss" in js:
                    return float(js["eval_loss"])
            except (json.JSONDecodeError, IOError) as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                LOG.warning(f"Failed to read wandb summary: {e}")
    return None

def loss_from_stdout(stdout: str) -> float | None:
    matches = _EVAL_RE.findall(stdout)
    return float(matches[-1]) if matches else None

def loss_from_state(out_dir: Path) -> float | None:
    """Extract loss from trainer state with retry logic"""
    p = out_dir / "trainer_state.json"
    for attempt in range(3):
        if p.exists():
            try:
                with p.open() as f:
                    js = json.load(f)
                for rec in reversed(js.get("log_history", [])):
                    if "eval_loss" in rec:
                        return float(rec["eval_loss"])
            except (json.JSONDecodeError, IOError) as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                LOG.warning(f"Failed to read trainer state: {e}")
    return None
# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── Objective (single trial) ─────────────────────────╮
def objective(
    trial: optuna.Trial,
    base_cfg: dict,
    hpo_project: str,
    study_name: str,
    storage_path: str,
    time_when_hpo_finished: datetime
) -> float:
    """Run a single trial with enhanced stability and retry logic"""
    
    # Check if we have enough time left
    time_left = (time_when_hpo_finished - datetime.now()).total_seconds()
    if time_left < 60:  # Less than 1 minute left
        LOG.warning("Not enough time left for new trial")
        raise optuna.exceptions.OptunaError("Time limit reached")
    
    cfg = copy.deepcopy(base_cfg)
    trial_params = sample_space(trial, cfg)
    cfg.update(trial_params)

    trial_id = f"trial{trial.number}"
    out_dir = Path(cfg.get("output_root", "./hpo_runs")) / trial_id
    cfg |= {
        "output_dir": str(out_dir),
        "wandb_run": f"{cfg['job_id'][:5]}_{cfg['rl']}_{trial_id}",
        "wandb_project": hpo_project,
        "max_steps": TRIAL_MAX_STEPS,
        "eval_steps": TRIAL_EVAL_STEPS,
        "save_steps": 500,
        "logging_steps": 10,  # More frequent logging for monitoring
        "save_total_limit": 1,  # Save disk space
        "load_best_model_at_end": False,  # Speed up for HPO
    }

    if cfg["testing"] == True:
        cfg |= {
            "max_steps": TESTING_TRIAL_MAX_STEPS,
            "eval_steps": TESTING_TRIAL_EVAL_STEPS,
        }

    cfg["hpo_run"] = True
    cfg["required_finish_time"] = (
        datetime.now() + timedelta(minutes=MAX_MINUTES_PER_TRIAL)
    ).isoformat()

    # Ensure clean output directory
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_cfg = Path(tempfile.mkdtemp()) / f"{trial_id}.yml"
    with tmp_cfg.open("w") as f:
        yaml.safe_dump(cfg, f)

    LOG.info("Starting trial %d with params: %s", trial.number, trial_params)
    
    # ── prepare environment for subprocess ─────────────────────────────
    env = os.environ.copy()
    env["WANDB_PROJECT"] = hpo_project
    env.pop("WANDB_RUN_ID", None)
    env.pop("WANDB_NAME", None)
    env["OPTUNA_STORAGE"] = storage_path
    env["OPTUNA_STUDY_NAME"] = study_name
    env["OPTUNA_TRIAL_ID"] = str(trial._trial_id)
    

    if cfg["rl"] == "grpo":
        cfg["trl"]["max_completion_length"] = 32

    path_to_train_file = "/workspace/training/train.py"

    cmd = [
        "accelerate", "launch",
        "--use_deepspeed",
        "--zero_stage", "2",
        "--mixed_precision", "bf16",
        "--num_processes", str(torch.cuda.device_count()),  # Explicit GPU count
        path_to_train_file,
        "--config", str(tmp_cfg),
    ]

    # ── Run subprocess with monitoring ────────────────────────────────
    stdout = ""
    try:
        proc = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,  # Line buffered
            preexec_fn=os.setsid  # Create new process group for cleanup
        )
        
        # Monitor the process
        monitor = ProcessMonitor(proc, MAX_MINUTES_PER_TRIAL * 60)
        
        # Collect output
        stdout_lines = []
        for line in proc.stdout:
            stdout_lines.append(line)
            # Check for early termination signals
            if "eval_loss" in line:
                LOG.info(f"Trial {trial.number}: {line.strip()}")
                # Report intermediate value for pruning
                match = _EVAL_RE.search(line)
                if match:
                    intermediate_value = float(match.group(1))
                    trial.report(intermediate_value, len(stdout_lines))
                    if trial.should_prune():
                        proc.terminate()
                        raise optuna.exceptions.TrialPruned()
        
        stdout = "".join(stdout_lines)
        proc.wait()
        monitor.stop()
        
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stdout)

    # ── Error handling with categorization ──────────────────────────
    except subprocess.CalledProcessError as e:
        msg = stdout or str(e)
        
        # Categorize errors
        penalty_value = float("-inf") if cfg["rl"] == "grpo" else float("inf")
        
        if "torch.OutOfMemoryError" in msg:
            LOG.warning("Trial %d failed: OOM error.", trial.number)
            # OOM might be due to specific hyperparameters, don't retry
            cleanup_resources()
            time.sleep(GPU_CLEANUP_WAIT_TIME)
            
        elif "Watchdog caught collective operation timeout" in msg or "NCCL" in msg:
            LOG.warning("Trial %d failed: NCCL/Communication error.", trial.number)
            LOG.warning(f"Error: {str(e)}")
            cleanup_resources()
            time.sleep(GPU_CLEANUP_WAIT_TIME)
            
        elif "Reached time limit" in msg or "exceeded max_seconds" in msg:
            LOG.info("Trial %d ran out of time: attempting to find last loss...", trial.number)
            # Try to extract partial results
            for extractor in (loss_from_wandb, lambda _: loss_from_stdout(stdout), loss_from_state):
                val = extractor(out_dir) if extractor is loss_from_wandb or extractor is loss_from_state else extractor(None)
                if val is not None:
                    LOG.info("Partial result found for trial %d: %.4f", trial.number, val)
                    return val
                    
        else:
            LOG.warning("Trial %d failed with unknown error:\n%s", trial.number, msg[:500])
            
        return penalty_value
        
    except optuna.exceptions.TrialPruned:
        LOG.info("Trial %d was pruned.", trial.number)
        cleanup_resources()
        return float("-inf") if cfg["rl"] == "grpo" else float("inf")
        
    except Exception as e:
        LOG.error(f"Unexpected error in trial {trial.number}: {e}")
        cleanup_resources()
        return float("-inf") if cfg["rl"] == "grpo" else float("inf")

    # ── extract eval_loss (3 fallback methods) ─────────────────────
    for extractor in (loss_from_wandb, lambda _: loss_from_stdout(stdout), loss_from_state):
        val = extractor(out_dir) if extractor is loss_from_wandb or extractor is loss_from_state else extractor(None)
        if val is not None:
            LOG.info("Trial %d completed – eval_loss: %.4f", trial.number, val)
            # Cleanup temporary files
            shutil.rmtree(tmp_cfg.parent, ignore_errors=True)
            # Save successful hyperparameters
            save_trial_results(trial, val, out_dir)
            cleanup_resources()
            return val

    LOG.warning("eval_loss not found for trial %d – penalising.", trial.number)
    return float("-inf") if cfg["rl"] == "grpo" else float("inf")

def save_trial_results(trial, value, out_dir):
    """Save trial results for later analysis"""
    results = {
        "trial_number": trial.number,
        "value": value,
        "params": trial.params,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "trial_results.json", "w") as f:
        json.dump(results, f, indent=2)
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

    LOG.info("HPO sweep starting  (project: %s)…", hpo_project)
    
    # Use more robust storage settings
    storage = RDBStorage(
        url=storage_path, 
        engine_kwargs={
            "connect_args": {
                "timeout": 60,  # Increased timeout
                "check_same_thread": False  # Allow multi-threading
            }, 
            "pool_pre_ping": True,
            "pool_size": 5,
            "max_overflow": 10
        }
    )

    if base_cfg["rl"] == "grpo":
        direction = "maximize"
    else:
        direction = "minimize"

    # Create study with more aggressive pruning for stability
    study = optuna.create_study(
        direction=direction,
        study_name=base_cfg["job_id"],
        load_if_exists=True,  # Allow resuming interrupted studies
        storage=storage,
        pruner=HyperbandPruner(
            min_resource=1,  # Allow early pruning
            max_resource=int(TRIAL_MAX_STEPS/TRIAL_EVAL_STEPS), 
            reduction_factor=3
        )
    )
    
    # Save study configuration
    study_config = {
        "base_cfg_path": base_cfg_path,
        "study_name": study_name,
        "direction": direction,
        "start_time": datetime.now().isoformat(),
    }
    with open(hpo_root / "study_config.json", "w") as f:
        json.dump(study_config, f, indent=2)
    
    # Calculate time budget
    time_remaining = datetime.fromisoformat(base_cfg['required_finish_time']) - datetime.now()
    seconds_remaining = max(0.0, time_remaining.total_seconds() * PERCENT_TIME_FOR_HPO)
    time_when_hpo_finished = datetime.now() + timedelta(seconds=seconds_remaining)

    LOG.info(f"Time allocated to HPO Search: {seconds_remaining/3600:.2f}h")
    
    # Run optimization with exception handling
    try:
        study.optimize(
            lambda t: objective(t, base_cfg, hpo_project, study_name, storage_path, time_when_hpo_finished),
            timeout=int(seconds_remaining),
            n_trials=MAX_TRIALS_TO_RUN,
            show_progress_bar=True,
            catch=(Exception,),  # Catch all exceptions to prevent study crash
            callbacks=[lambda study, trial: cleanup_resources()]  # Cleanup after each trial
        )
    except Exception as e:
        LOG.error(f"Study optimization failed: {e}")
        # Try to get best value so far
        if len(study.trials) > 0:
            LOG.info("Attempting to use best trial found so far...")
        else:
            raise

    # Save final results
    if study.best_trial:
        LOG.info("HPO finished – best eval_loss %.5f with params %s",
                study.best_value, study.best_params)
        
        # Save all trial history
        trial_history = []
        for trial in study.trials:
            trial_history.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "duration": (trial.datetime_complete - trial.datetime_start).total_seconds() 
                           if trial.datetime_complete else None
            })
        
        with open(hpo_root / "trial_history.json", "w") as f:
            json.dump(trial_history, f, indent=2)
            
        return study.best_params
    else:
        raise ValueError("No successful trials completed")
# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────── Write optimised YAML & launch main run ───────────────╮
def write_opt_cfg(base_cfg: str, best: dict) -> str:
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg.update(best)
    
    # Add stability configurations for full training
    cfg["dataloader_pin_memory"] = False  # Avoid potential memory issues
    cfg["dataloader_num_workers"] = 0  # Avoid multiprocessing issues
    
    opt_path = base_cfg.replace(".yml", "_opt.yml")
    with open(opt_path, "w") as f:
        yaml.safe_dump(cfg, f)
    LOG.info("💾  Wrote optimised config → %s", opt_path)
    return opt_path

def launch_training(cfg_path: str):
    """Launch full training with stability measures"""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    path_to_train_file = "/workspace/training/train.py"
    
    # Ensure clean GPU state before full training
    cleanup_resources()
    time.sleep(GPU_CLEANUP_WAIT_TIME)

    cmd = [
        "accelerate", "launch",
        "--use_deepspeed",
        "--zero_stage", "2",
        "--mixed_precision", "bf16",
        "--num_processes", str(torch.cuda.device_count()),
        path_to_train_file,
        "--config", cfg_path,
    ]

    LOG.info("🚀  Starting full training run")
    
    # Set stability environment variables
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_LAUNCH_BLOCKING"] = "0"
    env["NCCL_TIMEOUT"] = "3600"  # 1 hour for full training
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        LOG.error(f"Full training failed: {e}")
        raise
# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── CLI entry‑point ──────────────────────────────╮
def main():
    ap = argparse.ArgumentParser(description="HPO then full training")
    ap.add_argument("--config", required=True, help="Base YAML config file")
    ap.add_argument("--resume", action="store_true", help="Resume interrupted HPO study")
    args = ap.parse_args()
    
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
        
    if base_cfg["do_hpo"] == False:
        launch_training(args.config)
        return
    
    try:
        best_params = run_optuna(args.config)
        optimised_cfg = write_opt_cfg(args.config, best_params)
        
        # Clean pause before full training
        LOG.info("Pausing before full training run...")
        cleanup_resources()
        time.sleep(GPU_CLEANUP_WAIT_TIME * 2)
        
        launch_training(optimised_cfg)
    except Exception as e:
        LOG.error(f"HPO pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()