from pydantic import BaseModel
from pydantic import Field
import uuid
from datetime import datetime
from enum import Enum
from datetime import timedelta
import os
import uuid
import re
import toml
import yaml
from transformers import AutoTokenizer
from transformers import AutoConfig
from huggingface_hub import HfApi


hf_api = HfApi()

CONFIG_DIR = "/workspace/configs/"
CONFIG_TEMPLATE_PATH = CONFIG_DIR + "base.yml"
CONFIG_TESTING_PATH = CONFIG_DIR + "base_testing.yml"
OUTPUT_DIR = "/workspace/outputs/"
TRAIN_DIR = "/workspace/training/"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")
HUGGINGFACE_USERNAME = os.getenv("HUGGINGFACE_USERNAME")
CUSTOM_DATASET_TYPE = "custom"

# DPO default dataset type
DPO_DEFAULT_DATASET_TYPE = "chatml.intel"
DPO_DEFAULT_FIELD_PROMPT = "question"
DPO_DEFAULT_FIELD_SYSTEM = "system"
DPO_DEFAULT_FIELD_CHOSEN = "chosen"
DPO_DEFAULT_FIELD_REJECTED = "rejected"

GRPO_DEFAULT_FIELD_PROMPT = "prompt"
    
class FileFormat(str, Enum):
    CSV = "csv"  # needs to be local file
    JSON = "json"  # needs to be local file
    HF = "hf"  # Hugging Face dataset
    S3 = "s3"


class InstructTextDatasetType(BaseModel):
    system_prompt: str | None = ""
    system_format: str | None = "{system}"
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    field: str | None = None


class RewardFunction(BaseModel):
    """Model representing a reward function with its metadata"""
    reward_func: str = Field(
        ...,
        description="String with the python code of the reward function to use",
        examples=[
            "def reward_func_conciseness(completions, **kwargs):",
            "\"\"\"Reward function that favors shorter, more concise answers.\"\"\"",
            "    return [100.0/(len(completion.split()) + 10) for completion in completions]"
        ]
    )
    reward_weight: float = Field(..., ge=0)
    func_hash: str | None = None
    is_generic: bool | None = None


class GrpoDatasetType(BaseModel):
    field_prompt: str | None = None
    reward_functions: list[RewardFunction] | None = []


class DpoDatasetType(BaseModel):
    field_prompt: str | None = None
    field_system: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    prompt_format: str | None = "{prompt}"
    chosen_format: str | None = "{chosen}"
    rejected_format: str | None = "{rejected}"


def create_dataset_entry(
    dataset: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType,
    file_format: FileFormat,
    is_eval: bool = False,
) -> dict:
    dataset_entry = {"path": dataset}

    if isinstance(dataset_type, InstructTextDatasetType):
        print("Process Type: Instruct")
        dataset_entry.update(_process_instruct_dataset_fields(dataset_type))
    elif isinstance(dataset_type, DpoDatasetType):
        print("Process Type: DPO")
        dataset_entry.update(_process_dpo_dataset_fields(dataset_type))
    elif isinstance(dataset_type, GrpoDatasetType):
        print("Process Type: GRPO")
        dataset_entry.update(_process_grpo_dataset_fields(dataset_type))
    else:
        raise ValueError("Invalid dataset_type provided.")

    return dataset_entry


def update_flash_attention(config: dict, model: str):
    # You might want to make this model-dependent
    config["flash_attention"] = True
    return config


def update_model_info(config: dict, model: str, job_id: str = "", expected_repo_name: str | None = None):
    print("WE ARE UPDATING THE MODEL INFO")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}

    config["model_params_count"] = None
    try:
        model_info = hf_api.model_info(model)
        size = model_info.safetensors.total
        config["model_params_count"] = size
    except Exception as e:
        print(f"Error getting model size from safetensors: {e}")
        model_size = re.search(r"(\d+)(?=[bB])", model)
        model_size = int(model_size.group(1)) * 1_000_000_000 if model_size else None
        print(f"Model size from regex: {model_size}")
        config["model_params_count"] = model_size

    config["base_model"] = model
    if any(k in model.lower() for k in ("meta-llama-3.1")):
        config["packing"] = False
        config["use_liger_kernel"] = False
    config["base_model_config"] = model
    config["wandb_runid"] = f"{job_id[:6]}_{config['rl']}"
    config["wandb_run"] = f"{job_id[:6]}_{config['rl']}"
    config["wandb_name"] = f"{job_id[:6]}_{config['rl']}"
    config["hub_model_id"] = f"{HUGGINGFACE_USERNAME}/{expected_repo_name or str(uuid.uuid4())}"

    return config


def save_config(config: dict, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def save_config_toml(config: dict, config_path: str):
    with open(config_path, "w") as file:
        toml.dump(config, file)


def _process_grpo_dataset_fields(dataset_type: GrpoDatasetType) -> dict:
    field_prompt = dataset_type.field_prompt

    full_template_config = {"field_prompt": field_prompt}

    return full_template_config


def _process_dpo_dataset_fields(dataset_type: DpoDatasetType) -> dict:

    field_prompt = dataset_type.field_prompt
    field_chosen = dataset_type.field_chosen
    field_rejected = dataset_type.field_rejected
    full_template_config = {"field_prompt": field_prompt,  "field_chosen": field_chosen, "field_rejected": field_rejected}

    return full_template_config


def _process_instruct_dataset_fields(dataset_type: InstructTextDatasetType) -> dict:
    field_instruction = dataset_type.field_instruction
    field_input = dataset_type.field_input
    field_output = dataset_type.field_output
    full_template_config = {"field_instruction": field_instruction,  "field_input": field_input, "field_output": field_output}

    return full_template_config


def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType,
    file_format: FileFormat,
    task_id: str,
    expected_repo_name: str | None,
    required_finish_time: str,
    testing: bool
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """

    print("Loading config template")
    if not testing:
        with open(CONFIG_TEMPLATE_PATH, "r") as file:
            config = yaml.safe_load(file)
            config["testing"] = False
        config["required_finish_time"] = required_finish_time
    else:
        with open(CONFIG_TESTING_PATH, "r") as file:
            config = yaml.safe_load(file)
            config["testing"] = True
            config["required_finish_time"] = (datetime.now() + timedelta(hours=4)).isoformat()

    # RL specific params
    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
        config["learning_rate"] = 1e-6
        config["label_smoothing"] = 0.0
        config["beta"] = 0.04
    elif isinstance(dataset_type, GrpoDatasetType):
        config["rl"] = "grpo"
        config["learning_rate"] = 1e-6
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions], task_id
            )
        config["eval_steps"] = 200
        config["save_steps"] = 100
        config["trl"] = {}
        config["trl"]["beta"] = 0.04
        config["trl"]["max_completion_length"] = 128
        config["trl"]["use_vllm"] = False
        config["trl"]["num_generations"] = 2
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]
        config["rl_beta"] = 0.1
        config["beta"] = 0.04


    config["job_id"] = task_id
    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    config = update_model_info(config, model, task_id, expected_repo_name)
    
    hf_cfg = AutoConfig.from_pretrained(model)
    
    # Calculate max sequence length
    max_pos = getattr(hf_cfg, "max_position_embeddings", None) or getattr(hf_cfg, "n_ctx", None)

    # clamp sequence_len to the model’s max
    desired_len = config["sequence_len"]
    if max_pos is not None and desired_len > max_pos:
        print(f"Requested seq_len={desired_len} > model max {max_pos}; falling back to {max_pos}")
        config["sequence_len"] = max_pos
        print(f"Sequence Length set to: {max_pos}")
    else:
        config["sequence_len"] = desired_len
    ######################

    if config["adapter"] == "lora":
        config = setup_lora_config(config)

    return config


def create_reward_funcs_file(reward_funcs: list[str], task_id: str) -> list[str]:
    """
    Create a Python file with reward functions for GRPO training.

    Args:
        reward_funcs: List of strings containing Python reward function implementations
        task_id: Unique task identifier
    """
    filename = f"rewards_{task_id}"
    filepath = os.path.join(CONFIG_DIR, f"{filename}.py")

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    with open(filepath, "w") as f:
        f.write("# Auto-generated reward functions file\n\n")
        for reward_func in reward_funcs:
            f.write(f"{reward_func}\n\n")

    return filename, func_names


def setup_lora_config(config):
    """Setup QLoRA configuration for more efficient adaptation"""
    config["adapter"] = "lora"
    config["lora_r"] = 32
    config["lora_alpha"] = config["lora_alpha"]
    config["lora_dropout"] = 0.05
    return config


def setup_config(
    dataset: str,
    model: str,
    dataset_type: dict,
    file_format: str,
    task_id: str,
    expected_repo_name: str | None,
    required_finish_time: str,
    testing: bool,
    hpo: bool
):
    # Deserialize dataset_type based on class_type
    if isinstance(dataset_type, dict) and "class_type" in dataset_type:
        dataset_type_class = dataset_type["class_type"]
        class_attributes = dataset_type.get("attributes", {})
        
        # Create an instance directly based on the class name
        if dataset_type_class == "DpoDatasetType":
            print("Dataset Type: DPO")
            dataset_type = DpoDatasetType(**class_attributes)
        elif dataset_type_class == "InstructTextDatasetType":
            print("Dataset Type: Instruct")
            dataset_type = InstructTextDatasetType(**class_attributes)
        elif dataset_type_class == "GrpoDatasetType":
            print("Dataset Type: GRPO")
            # Handle nested RewardFunction objects in GrpoDatasetType
            if "reward_functions" in class_attributes and class_attributes["reward_functions"]:
                reward_functions = []
                for reward_func_dict in class_attributes["reward_functions"]:
                    reward_functions.append(RewardFunction(**reward_func_dict))
                class_attributes["reward_functions"] = reward_functions
            dataset_type = GrpoDatasetType(**class_attributes)

    else:
        # Handle error or default case
        print(f"Unable to deserialize dataset_type: {dataset_type}")
        return {
            "success": False,
            "task_id": task_id,
            "error": "Invalid dataset_type format"
        }
    
    if not testing:
        # Modify Config and save
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(CONFIG_DIR, config_filename)
    else:
        config_filename = f"test_{task_id}.yml"
        config_path = os.path.join(CONFIG_DIR, config_filename)
        
    config = _load_and_modify_config(
        dataset,
        model,
        dataset_type,
        file_format,
        task_id,
        expected_repo_name,
        required_finish_time,
        testing
    )
    if not hpo:
        config["do_hpo"] = False
        
    print("Initial Config:")
    print("=======================================")
    print(f"Task ID: {config['job_id']}")
    print(f"Model: {config['base_model']}")
    print(f"Model Params: {config['model_params_count']}")
    print(f"HuggingFace Repo: {config['hub_model_id']}")
    print(f"RL Type: {config['rl']}")
    print("=======================================")

    save_config(config, config_path)