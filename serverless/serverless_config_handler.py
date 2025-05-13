from pydantic import BaseModel
from pydantic import Field
import uuid
from datetime import datetime
from enum import Enum
from uuid import UUID
import os
import uuid
import re
import toml
import yaml
from transformers import AutoTokenizer
from transformers import AutoConfig
from huggingface_hub import HfApi
from urllib.parse import urlparse
from fastapi import HTTPException
import requests


hf_api = HfApi()

CONFIG_DIR = "/workspace/configs/"
CONFIG_TEMPLATE_PATH = CONFIG_DIR + "base.yml"
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


def download_s3_file_sync(file_url: str, save_path: str | None = None,
                          tmp_dir: str = "/tmp") -> str:
    parsed = urlparse(file_url)
    local = os.path.join(save_path or tmp_dir, os.path.basename(parsed.path))
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local

    r = requests.get(file_url, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "wb") as f:
        f.write(r.content)
    return local


def create_dataset_entry(
    dataset: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType,
    file_format: FileFormat,
    is_eval: bool = False,
) -> dict:
    dataset_entry = {"path": dataset}

    if file_format == FileFormat.JSON:
        if not is_eval:
            dataset_entry = {"path": "/workspace/input_data/"}
        else:
            dataset_entry = {"path": f"/workspace/input_data/{os.path.basename(dataset)}"}

    if isinstance(dataset_type, InstructTextDatasetType):
        print("Process Type: DPO")
        instruct_type_dict = {key: value for key, value in dataset_type.model_dump().items() if value is not None}
        dataset_entry.update(_process_instruct_dataset_fields(instruct_type_dict))
    elif isinstance(dataset_type, DpoDatasetType):
        print("Process Type: DPO")
        dataset_entry.update(_process_dpo_dataset_fields(dataset_type))
    elif isinstance(dataset_type, GrpoDatasetType):
        print("Process Type: DPO")
        dataset_entry.update(_process_grpo_dataset_fields(dataset_type))
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry["ds_type"] = file_format.value
        dataset_entry["data_files"] = [os.path.basename(dataset)]

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
    config["base_model_config"] = model
    config["wandb_runid"] = f"{job_id[:5]}_{config['rl']}"
    config["wandb_run"] = f"{job_id[:5]}_{config['rl']}"
    config["wandb_name"] = f"{job_id[:5]}_{config['rl']}"
    config["hub_model_id"] = f"{HUGGINGFACE_USERNAME}/{expected_repo_name or str(uuid.uuid4())}"

    return config


def save_config(config: dict, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def save_config_toml(config: dict, config_path: str):
    with open(config_path, "w") as file:
        toml.dump(config, file)


def _process_grpo_dataset_fields(dataset_type: GrpoDatasetType) -> dict:
    return {"split": "train"}


def _process_dpo_dataset_fields(dataset_type: DpoDatasetType) -> dict:
    # Enable below when https://github.com/axolotl-ai-cloud/axolotl/issues/1417 is fixed
    # context: https://discord.com/channels/1272221995400167588/1355226588178022452/1356982842374226125

    # dpo_type_dict = dataset_type.model_dump()
    # dpo_type_dict["type"] = "user_defined.default"
    # if not dpo_type_dict.get("prompt_format"):
    #     if dpo_type_dict.get("field_system"):
    #         dpo_type_dict["prompt_format"] = "{system} {prompt}"
    #     else:
    #         dpo_type_dict["prompt_format"] = "{prompt}"
    # return dpo_type_dict

    # Fallback to https://axolotl-ai-cloud.github.io/axolotl/docs/rlhf.html#chatml.intel
    # Column names are hardcoded in axolotl: "DPO_DEFAULT_FIELD_SYSTEM",
    # "DPO_DEFAULT_FIELD_PROMPT", "DPO_DEFAULT_FIELD_CHOSEN", "DPO_DEFAULT_FIELD_REJECTED"
    return {"type": DPO_DEFAULT_DATASET_TYPE, "split": "train"}


def _process_instruct_dataset_fields(instruct_type_dict: dict) -> dict:
    if not instruct_type_dict.get("field_output"):
        return {
            "type": "completion",
            "field": instruct_type_dict.get("field_instruction"),
        }

    processed_dict = instruct_type_dict.copy()
    processed_dict.setdefault("no_input_format", "{instruction}")
    if processed_dict.get("field_input"):
        processed_dict.setdefault("format", "{instruction} {input}")
    else:
        processed_dict.setdefault("format", "{instruction}")

    return {"format": "custom", "type": processed_dict}


def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType,
    file_format: FileFormat,
    task_id: str,
    expected_repo_name: str | None,
    required_finish_time: datetime
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    config_path = CONFIG_TEMPLATE_PATH

    print("Loading config template")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["job_id"] = task_id
    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)
    
    
    config["required_finish_time"] = required_finish_time.isoformat()

    config = update_model_info(config, model, task_id, expected_repo_name)

    # Modify config based on Model Size
    if config["model_params_count"] == None:
        config["learning_rate"] = 2e-4
    elif config["model_params_count"] < 2_000_000_000:
        print("Small model detected...updating params...")
        config["learning_rate"] = 2e-4

    elif config["model_params_count"] < 8_000_000_000:
        print("Medium model detected...updating params...")
        config["learning_rate"] = 2e-4

    elif config["model_params_count"] < 15_000_000_000:
        print("Large model detected...updating params...")
        config["learning_rate"] = 1e-5

    elif config["model_params_count"] < 40_000_000_000:
        config["learning_rate"] = 1e-5


    # RL specific params
    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions], task_id
            )
        config["rl"] = "grpo"
        config["max_steps"] = 100
        config["eval_steps"] = 100
        config["save_steps"] = 100
        config["trl"] = {}
        config["trl"]["beta"] = 0.04
        config["trl"]["max_completion_length"] = 32
        config["trl"]["use_vllm"] = False 
        config["trl"]["num_generations"] = 2
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]
        config["rl_beta"] = 0.1
        config["beta"] = 0.04

    hf_cfg = AutoConfig.from_pretrained(model)
 
    max_pos = getattr(hf_cfg, "max_position_embeddings", None) or getattr(hf_cfg, "n_ctx", None)

    # clamp sequence_len to the model’s max
    desired_len = config["sequence_len"]
    if max_pos is not None and desired_len > max_pos:
        print(f"Requested seq_len={desired_len} > model max {max_pos}; falling back to {max_pos}")
        config["sequence_len"] = max_pos
        print(f"Sequence Length set to: {max_pos}")
    else:
        config["sequence_len"] = desired_len

    config["mlflow_experiment_name"] = dataset

    config = setup_lora_config(config, config["model_params_count"])

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

def setup_lora_config(config, model_size):
    """Setup QLoRA configuration for more efficient adaptation"""
    config["adapter"] = "lora"
    config["lora_r"] = min(256, max(64, int(model_size / 50_000_000)))
    config["lora_alpha"] = config["lora_r"] * 2
    config["lora_dropout"] = 0.05
    return config


def setup_config(
    dataset: str,
    model: str,
    dataset_type: dict,
    file_format: str,
    task_id: str,
    expected_repo_name: str | None,
    required_finish_time: datetime
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
    
    # Convert file_format string back to enum
    file_format_str = file_format
    try:
        file_format = FileFormat(file_format_str)
    except ValueError:
        print(f"Invalid file_format: {file_format_str}, using default")
        file_format = FileFormat.JSON  # Default

    # Download dataset
    try:
        print(file_format)
        if file_format != FileFormat.HF:
            if file_format == FileFormat.S3:
                dataset = download_s3_file_sync(dataset, "/workspace")
                print(dataset)
                file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    
    # Modify Config and save
    config_filename = f"{task_id}.yml"
    config_path = os.path.join(CONFIG_DIR, config_filename)
    config = _load_and_modify_config(
        dataset,
        model,
        dataset_type,
        file_format,
        task_id,
        expected_repo_name,
        required_finish_time
    )
    print("CONFIG AFTER SETUP")
    print(config)
    save_config(config, config_path)