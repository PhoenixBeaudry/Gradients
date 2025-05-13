import runpod
import os
import yaml
import json
from datetime import datetime, timedelta
from serverless_config_handler import setup_config
import subprocess
import asyncio

# You'll need to adapt your existing training code for the serverless environment
def handler(job):
    """
    Process incoming training job requests in RunPod Serverless
    
    Args:
        job (dict): Contains job information including:
            - input: Configuration for training
            - id: Unique job identifier
    
    Returns:
        dict: Results of the training job
    """
    job_input = job["input"]
    job_id = job_input.get("task_id")
    
    print(f"Starting training job: {job_id}")
    
    # Extract training parameters from the job input
    model = job_input.get("model")
    dataset = job_input.get("dataset")
    dataset_type = job_input.get("dataset_type")
    file_format = job_input.get("file_format")
    expected_repo_name = job_input.get("expected_repo_name")
    hours_to_complete = job_input.get("hours_to_complete", 24)
    
    # Calculate required finish time
    required_finish_time = (datetime.now() + timedelta(hours=hours_to_complete))
    
    # Load configuration, setup training, etc.
    CONFIG_DIR = "/workspace/configs"
    config_filename = f"{job_id}.yml"
    config_path = os.path.join(CONFIG_DIR, config_filename)

    asyncio.run(setup_config(
        dataset,
        model,
        dataset_type,
        file_format,
        job_id,
        expected_repo_name,
        required_finish_time
    ))
    # Execute the training process
     # Run the HPO script
    try:
        # Assuming hpo_optuna.py is in /workspace
        cmd = [
            "python", 
            "/workspace/training/hpo_optuna.py", 
            "--config", config_path
        ]
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream logs
        log_output = []
        for line in iter(process.stdout.readline, ''):
            print(line, end='')  # Print to RunPod logs
            log_output.append(line)
            if len(log_output) > 1000:  # Keep a rolling buffer of last 1000 lines
                log_output.pop(0)
        
        # Wait for process to complete
        process.wait()
        
        # Check if process completed successfully
        if process.returncode != 0:
            raise Exception(f"HPO process failed with return code {process.returncode}")
        
        # Return results
        return {
            "success": True,
            "task_id": job_id,
            "model_repo": expected_repo_name,
            "training_completed": datetime.now().isoformat(),
            "last_logs": ''.join(log_output[-100:])  # Return last 100 lines of logs
        }
    
    except Exception as e:
        print(f"Error running HPO: {str(e)}")
        return {
            "success": False,
            "task_id": job_id,
            "error": str(e),
            "last_logs": ''.join(log_output[-100:]) if 'log_output' in locals() else "No logs captured"
        }


# Start the serverless worker
runpod.serverless.start({"handler": handler})