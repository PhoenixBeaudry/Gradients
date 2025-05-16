import os
from enum import Enum
import toml
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
import json
import yaml
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import get_config
from fiber.miner.dependencies import blacklist_low_stake
from pydantic import ValidationError
import runpod
from pathlib import Path
import core.constants as cst
from core.models.payload_models import MinerTaskOffer
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequestGrpo
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.payload_models import TrainResponse
from core.models.utility_models import TaskType



MAX_NUM_WORKERS = 6 # Ensure this matches max workers on runpod endpoint

logger = get_logger(__name__)

runpod.api_key = os.getenv("ENDPOINT_API_KEY")

async def tune_model_text(
    train_request: TrainRequestText,
):
    logger.info("Starting model tuning.")

    logger.info(f"Job received is {train_request}")

    # Check if file exists if not right temp config with huggingface repo

    CONFIG_DIR = "core/config"
    config_filename = f"{train_request.task_id}.yml"
    config_path = os.path.join(CONFIG_DIR, config_filename)

    with open(f"{CONFIG_DIR}/base.yml", "r") as file:
        config = yaml.safe_load(file)
    config["hub_model_id"] = f"{cst.HUGGINGFACE_USERNAME}/{train_request.expected_repo_name}"

    with open(config_path, "w") as file:
        yaml.dump(config, file)


    # Format the request for RunPod
    # Serialize Dataset Type
    serial_dataset_type = {}
    if isinstance(train_request.dataset_type, InstructTextDatasetType):
        serial_dataset_type["class_type"] = "InstructTextDatasetType"
    elif isinstance(train_request.dataset_type, DpoDatasetType):
        serial_dataset_type["class_type"] = "DpoDatasetType"    
    elif isinstance(train_request.dataset_type, GrpoDatasetType):
        serial_dataset_type["class_type"] = "GrpoDatasetType"
    
    serial_dataset_type["attributes"] = json.loads(train_request.dataset_type.model_dump_json())

    # Serialize file_format (Enum)
    file_format_str = train_request.file_format.value if isinstance(train_request.file_format, Enum) else str(train_request.file_format)

    runpod_request = {
        "model": train_request.model,
        "dataset": train_request.dataset,
        "dataset_type": serial_dataset_type,
        "file_format": file_format_str,
        "expected_repo_name": train_request.expected_repo_name,
        "hours_to_complete": train_request.hours_to_complete,
        "task_id": str(train_request.task_id)
    }
    
    try:
        # Create a RunPod endpoint instance
        endpoint = runpod.Endpoint("3f7fu7em4zi8m3")
        
        # Submit the job to RunPod
        job = endpoint.run(runpod_request)
        
        logger.info(f"Submitted job to RunPod Serverless with ID: {train_request.task_id}")
        
        return {"message": "Training job enqueued on RunPod Serverless.", "task_id": str(train_request.task_id)}
        
    except Exception as e:
        logger.error(f"Error submitting job to RunPod: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting job to RunPod: {str(e)}")



async def tune_model_grpo(
    train_request: TrainRequestGrpo,
):
    logger.info("Starting model tuning.")

    logger.info(f"Job received is {train_request}")
    
    CONFIG_DIR = "core/config"
    config_filename = f"{train_request.task_id}.yml"
    config_path = os.path.join(CONFIG_DIR, config_filename)

    with open(f"{CONFIG_DIR}/base.yml", "r") as file:
        config = yaml.safe_load(file)
    config["hub_model_id"] = f"{cst.HUGGINGFACE_USERNAME}/{train_request.expected_repo_name}"

    with open(config_path, "w") as file:
        yaml.dump(config, file)

    # Format the request for RunPod
    # Serialize Dataset Type
    serial_dataset_type = {}
    if isinstance(train_request.dataset_type, InstructTextDatasetType):
        serial_dataset_type["class_type"] = "InstructTextDatasetType"
    elif isinstance(train_request.dataset_type, DpoDatasetType):
        serial_dataset_type["class_type"] = "DpoDatasetType"    
    elif isinstance(train_request.dataset_type, GrpoDatasetType):
        serial_dataset_type["class_type"] = "GrpoDatasetType"
    
    serial_dataset_type["attributes"] = json.loads(train_request.dataset_type.model_dump_json())

    # Serialize file_format (Enum)
    file_format_str = train_request.file_format.value if isinstance(train_request.file_format, Enum) else str(train_request.file_format)

    runpod_request = {
        "model": train_request.model,
        "dataset": train_request.dataset,
        "dataset_type": serial_dataset_type,
        "file_format": file_format_str,
        "expected_repo_name": train_request.expected_repo_name,
        "hours_to_complete": train_request.hours_to_complete,
        "task_id": str(train_request.task_id)
    }
    
    try:
        # Create a RunPod endpoint instance
        endpoint = runpod.Endpoint("3f7fu7em4zi8m3")
        
        # Submit the job to RunPod
        job = endpoint.run(runpod_request)
        
        logger.info(f"Submitted job to RunPod Serverless with ID: {train_request.task_id}")
        
        return {"message": "Training job enqueued on RunPod Serverless.", "task_id": str(train_request.task_id)}
        
    except Exception as e:
        logger.error(f"Error submitting job to RunPod: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting job to RunPod: {str(e)}")


async def get_latest_model_submission(task_id: str) -> str:
    try:
        # Temporary work around in order to not change the vali a lot
        # Could send the task type from vali instead of matching file names
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
                return config_data.get("hub_model_id", None)
        else:
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)
            with open(config_path, "r") as file:
                config_data = toml.load(file)
                return config_data.get("huggingface_repo_id", None)

    except FileNotFoundError as e:
        logger.error(f"No submission found for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest model submission: {str(e)}",
        )


async def task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
) -> MinerTaskResponse:
    try:
        logger.info("An offer has come through")
        logger.info(f"Model: {request.model.lower()}, Time: {request.hours_to_complete}")
        if request.task_type == TaskType.INSTRUCTTEXTTASK:
            logger.info("Task Type: Instruct")
        if request.task_type == TaskType.DPOTASK:
            logger.info("Task Type: DPO")
        if request.task_type == TaskType.GRPOTASK:
            logger.info("Task Type: GRPO")


        ######### Rejections #########
        if request.task_type not in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK]:
            return MinerTaskResponse(message=f"This endpoint only accepts text tasks.", accepted=False)
        
        # Check model parameter count
        # Reject if model size is 32B or larger
        if request.model_params_count is not None and request.model_params_count >= 30_000_000_000:
            logger.info(f"Rejecting offer: Model size too large ({request.model_params_count / 1_000_000_000:.1f}B >= 40B)")
            return MinerTaskResponse(message="Model size too large (>= 40B)", accepted=False)
        
        # Reject if unsupported model
        # if any(k in request.model.lower() for k in ("neo", "stella", "falcon", "gpt-j")):
        #     logger.info(f"Rejecting offer: Unsupported Model: ({request.model.lower()})")
        #     return MinerTaskResponse(
        #         message=f"This endpoint does not currently support that model.", accepted=False)
        
        # Reject absurdly long jobs
        if request.hours_to_complete >= 48:
            logger.info(f"Rejecting offer: too long ({request.hours_to_complete}h)")
            return MinerTaskResponse(message="Job too long", accepted=False)
        
        #################################


        # Create a RunPod endpoint instance
        endpoint = runpod.Endpoint("3f7fu7em4zi8m3")
        endpoint_health = endpoint.health()
        active_workers = int(endpoint_health["workers"]["running"])
        logger.info(f"Currently Active Runpod Workers: {active_workers}")

        # Reject if all workers busy
        if active_workers >= MAX_NUM_WORKERS:
            logger.info("Rejecting offer as all workers are busy.")
            return MinerTaskResponse(message=f"All our workers are busy... Sorry!", accepted=False)
        
        CONFIG_DIR = "core/config"
        config_filename = f"{request.task_id}.yml"
        config_path = os.path.join(CONFIG_DIR, config_filename)
        file_path = Path(config_path)
        if file_path.exists():
            logger.info("Rejecting offer a miner has already taken.")
            # Another miner already took the job
            return MinerTaskResponse(message=f"No thank you.", accepted=False)
        else:
            #Write the config file to make sure other miners don't accept
            with open(f"{CONFIG_DIR}/base.yml", "r") as file:
                config = yaml.safe_load(file)
            with open(config_path, "w") as file:
                yaml.dump(config, file)
        
        
        # otherwise accept
        logger.info(f"Accepting offer): {request.model} ({request.hours_to_complete}h)")
        return MinerTaskResponse(message="-----:)-----", accepted=True)


    except ValidationError as e:
        logger.error(f"Validation error in task_offer: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def task_offer_image(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    # worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info("An image offer has come through")
        return MinerTaskResponse(message=f"No images :(", accepted=False)

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer_image: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")
    


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake)],
    )

    router.add_api_route(
        "/task_offer_image/",
        task_offer_image,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake)],
    )

    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=str,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake)],
    )
    router.add_api_route(
        "/start_training/",  # TODO: change to /start_training_text or similar
        tune_model_text,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake)],
    )
    router.add_api_route(
        "/start_training_grpo/",
        tune_model_grpo,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake)],
    )
    

    return router
