from transformers import SchedulerType
from trl.trainer.grpo_trainer import RewardFunc
import os
import importlib
import sys
import inspect



def build_trainer_args(cfg: dict):

    # Hub Args for HPO Trials
    hub_kwargs = {}
    if not cfg["hpo_run"]:
        hub_kwargs = {
            'hub_model_id': cfg['hub_model_id'],
            'hub_token': cfg['hub_token'],
            'hub_strategy': "end",
            'push_to_hub': True,
        }
        lr_scheduler=SchedulerType.COSINE
    else:
        lr_scheduler=SchedulerType.CONSTANT_WITH_WARMUP


    # Build Main Invariant Training Arguments
    trainer_kwargs = {
        # Training Length Args
        "max_steps": int(cfg['max_steps']),
        "logging_steps": int(cfg['logging_steps']),

        # Optimizer Args
        "optim": cfg['optimizer'],
        "weight_decay": float(cfg['weight_decay']),
        "gradient_checkpointing": cfg['gradient_checkpointing'],
        "gradient_checkpointing_kwargs": {'use_reentrant':False},

        # LR Args
        "learning_rate": float(cfg['learning_rate']),
        "lr_scheduler_type": lr_scheduler,
        "warmup_steps": cfg['warmup_steps'],

        # Batch and Memory Args
        "per_device_train_batch_size": int(cfg['micro_batch_size']),
        "per_device_eval_batch_size": int(cfg['micro_batch_size']),
        "gradient_accumulation_steps": int(cfg['gradient_accumulation_steps']),
        "packing": cfg['packing'],
        "eval_packing": cfg['packing'],    
        "dataset_num_proc": 6,
        "dataloader_num_workers": 6,

        # Evaluation and Saving Args
        "eval_strategy": 'steps', 
        "save_strategy": 'best',
        "eval_steps": int(cfg['eval_steps']),
        "save_steps": int(cfg['save_steps']),
        "save_total_limit": int(cfg['save_total_limit']),

        # Best Metric Args
        "metric_for_best_model": cfg['metric_for_best_model'],
        "greater_is_better": False,
        "load_best_model_at_end": True,

        # Optimization Args
        "bf16": True,
        "ddp_find_unused_parameters": False,
        "ddp_timeout": 3600,
        "dataloader_pin_memory": False,
        "use_liger_kernel": cfg['use_liger_kernel'],
        "auto_find_batch_size": True,

        # Misc Args
        "output_dir": cfg['output_dir'],
        "run_name": cfg['wandb_run'],
        "report_to": "wandb",
    }

    trainer_kwargs |= hub_kwargs

    # Training Type Specific Args
    type_spec_args = {}
    if cfg["rl"] == "dpo":
        type_spec_args = {
            'beta': float(cfg['beta']),
            'label_smoothing': float(cfg['label_smoothing'])
        }
    elif cfg["rl"] == "grpo":
        type_spec_args = {
            'beta': float(cfg['beta']),
            'num_generations': int(cfg["trl"]["num_generations"]),
            'max_completion_length': int(cfg["trl"]["max_completion_length"]),
            'reward_weights': cfg["trl"]["reward_weights"],
            'use_vllm': False,
            'greater_is_better': True,
        }

    trainer_kwargs |= type_spec_args

    return trainer_kwargs


#### GRPO Specific ####
CONFIG_DIR = os.path.abspath("/workspace/configs/")


##### Custom Funcs for getting GRPO reward functions #####
def reward_functions(cfg):
    """
    Collects and returns a list of functions for GRPOTrainer.
    """
    funcs = []
    for fqn in cfg['trl']['reward_funcs']:
        funcs.append(get_reward_func(fqn))
    return funcs


def get_reward_func(reward_func_fqn: str) -> RewardFunc | str:
    """
    Try to load <module>.py from CONFIG_DIR and return its <func>.
    If the file doesn’t exist, just return the original string (HF model path).
    """
    module_name, func_name = reward_func_fqn.rsplit(".", 1)
    module_path = os.path.join(CONFIG_DIR, f"{module_name}.py")
    print(f"→ looking for {module_name!r} at {module_path!r}, exists? {os.path.isfile(module_path)}")
    # 1) if we have an on-disk file, dynamically import it
    if os.path.isfile(module_path):
        # drop any cached module so we always load the newest version
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # get the function
        if not hasattr(module, func_name):
            raise AttributeError(
                f"Module {module_name!r} has no attribute {func_name!r}"
            )
        reward_func = getattr(module, func_name)

        # sanity check signature
        sig = inspect.signature(reward_func)
        if len(sig.parameters) < 2:
            raise ValueError(
                "Reward function must accept at least two arguments: "
                "prompts: list and completions: list"
            )

        return reward_func

    # 2) otherwise fall back to treating the FQN string as a model-path
    return reward_func_fqn
