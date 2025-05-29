from transformers import TrainerCallback, TrainerControl, TrainerState
import optuna
import time
import os

class TimeLimitCallback(TrainerCallback):
    """
    Stop training after `max_seconds` of wall-clock time.

    • Checks the clock only every `check_interval` steps (default 50).
    • Computes `deadline` once instead of recomputing `elapsed` each step.
    • Runs the check only on the local-process-zero rank to avoid redundant work
      in distributed/DeepSpeed jobs; other ranks receive the stop signal
      through the shared `TrainerControl` object.
    """

    __slots__ = ("deadline", "check_interval", "next_check_step")

    def __init__(self, max_seconds: float, check_interval: int = 10) -> None:
        self.deadline = time.perf_counter() + max_seconds
        self.check_interval = max(1, check_interval)          # at least 1 step
        self.next_check_step = self.check_interval

    # ────────────────────────────────────────────────────────────────
    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Only rank-0 performs the lightweight time check.
        if not state.is_local_process_zero:
            return control

        # Skip until the next scheduled check step.
        if state.global_step < self.next_check_step:
            return control

        # Schedule the following check now, before doing any work.
        self.next_check_step += self.check_interval

        # Single high-resolution clock read.
        if time.perf_counter() >= self.deadline:
            control.should_training_stop = True
            print(f"\nReached time limit — stopping training.")

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