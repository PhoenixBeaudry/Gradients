#!/usr/bin/env python3
"""
RunPod Serverless Handler for Training Jobs
Enhanced with stability, monitoring, and resource management
"""
import runpod
import os
import sys
import json
import time
import signal
import traceback
import threading
import subprocess
import psutil
import torch
import gc
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import logging
from contextlib import contextmanager
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/runpod_handler.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_LOG_BUFFER_SIZE = 2000
PROGRESS_UPDATE_INTERVAL = 30  # seconds
HEALTH_CHECK_INTERVAL = 60  # seconds
GPU_MEMORY_THRESHOLD = 0.95  # 95% memory usage threshold
CLEANUP_WAIT_TIME = 10  # seconds

@dataclass
class JobConfig:
    """Validated job configuration"""
    task_id: str
    model: str
    dataset: str
    dataset_type: str
    file_format: str
    expected_repo_name: str
    hours_to_complete: float
    testing: bool
    hpo: bool
    config_path: Path
    required_finish_time: datetime
    
    @classmethod
    def from_job_input(cls, job_input: Dict[str, Any]) -> 'JobConfig':
        """Create JobConfig from job input with validation"""
        # Required fields
        task_id = job_input.get("task_id")
        if not task_id:
            raise ValueError("task_id is required")
            
        model = job_input.get("model")
        if not model:
            raise ValueError("model is required")
            
        dataset = job_input.get("dataset")
        if not dataset:
            raise ValueError("dataset is required")
            
        # Optional fields with defaults
        dataset_type = job_input.get("dataset_type", "general")
        file_format = job_input.get("file_format", "json")
        expected_repo_name = job_input.get("expected_repo_name", f"{model}-{task_id}")
        hours_to_complete = float(job_input.get("hours_to_complete", 24))
        testing = bool(job_input.get("testing", False))
        hpo = bool(job_input.get("hpo", True))
        
        # Determine config path
        config_dir = Path("/workspace/configs")
        config_dir.mkdir(exist_ok=True)
        config_filename = f"{'test_' if testing else ''}{task_id}.yml"
        config_path = config_dir / config_filename
        
        # Calculate deadline
        required_finish_time = datetime.now() + timedelta(hours=hours_to_complete)
        
        return cls(
            task_id=task_id,
            model=model,
            dataset=dataset,
            dataset_type=dataset_type,
            file_format=file_format,
            expected_repo_name=expected_repo_name,
            hours_to_complete=hours_to_complete,
            testing=testing,
            hpo=hpo,
            config_path=config_path,
            required_finish_time=required_finish_time
        )

class ResourceManager:
    """Manage system resources and cleanup"""
    
    @staticmethod
    def setup_environment(num_gpus: int) -> Dict[str, str]:
        """Setup environment variables for optimal performance"""
        env_vars = {}
        
        # CPU configuration
        num_phys_cpus = psutil.cpu_count(logical=False)
        # Reserve 2 cores per GPU for data loading, leave some for system
        num_omp_threads = max(1, (num_phys_cpus - num_gpus * 2 - 2) // num_gpus)
        env_vars['OMP_NUM_THREADS'] = str(num_omp_threads)
        
        # CUDA configuration
        env_vars['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        env_vars['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(num_gpus))
        
        # PyTorch configuration
        env_vars['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6;8.9;9.0'  # Support various GPU architectures
        env_vars['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Distributed training
        env_vars['NCCL_TIMEOUT'] = '3600'  # 1 hour
        env_vars['NCCL_DEBUG'] = 'INFO'
        
        # Other optimizations
        env_vars['TOKENIZERS_PARALLELISM'] = 'false'
        env_vars['WANDB_SILENT'] = 'true'
        
        # Apply environment variables
        os.environ.update(env_vars)
        
        logger.info(f"Environment configured: {num_gpus} GPUs, {num_phys_cpus} CPUs, {num_omp_threads} OMP threads per GPU")
        return env_vars
    
    @staticmethod
    def cleanup_resources():
        """Clean up GPU memory and system resources"""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Kill any orphaned processes
            current_process = psutil.Process()
            for child in current_process.children(recursive=True):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Wait for termination
            time.sleep(2)
            
            # Force kill if necessary
            for child in current_process.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
                    
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
    
    @staticmethod
    def check_gpu_memory() -> List[Dict[str, float]]:
        """Check GPU memory usage"""
        gpu_stats = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    usage_percent = (allocated / total) * 100
                    
                    gpu_stats.append({
                        'gpu_id': i,
                        'allocated_gb': allocated,
                        'reserved_gb': reserved,
                        'total_gb': total,
                        'usage_percent': usage_percent
                    })
        return gpu_stats

class ProcessMonitor:
    """Monitor and manage the training subprocess"""
    
    def __init__(self, process: subprocess.Popen, job_config: JobConfig, progress_callback=None):
        self.process = process
        self.job_config = job_config
        self.progress_callback = progress_callback
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.is_alive = True
        self.monitor_thread = None
        self.health_thread = None
        
    def start_monitoring(self):
        """Start monitoring threads"""
        self.monitor_thread = threading.Thread(target=self._monitor_process)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.health_thread = threading.Thread(target=self._health_check)
        self.health_thread.daemon = True
        self.health_thread.start()
    
    def _monitor_process(self):
        """Monitor process and enforce time limits"""
        while self.is_alive and self.process.poll() is None:
            elapsed = time.time() - self.start_time
            remaining = (self.job_config.required_finish_time - datetime.now()).total_seconds()
            
            # Check if we're approaching the deadline
            if remaining < 300:  # 5 minutes buffer
                logger.warning(f"Approaching deadline, {remaining/60:.1f} minutes remaining")
                if remaining < 60:  # 1 minute left
                    logger.error("Time limit exceeded, terminating process")
                    self.terminate()
                    break
            
            # Send progress update
            if time.time() - self.last_progress_time > PROGRESS_UPDATE_INTERVAL:
                if self.progress_callback:
                    self.progress_callback({
                        'elapsed_seconds': elapsed,
                        'remaining_seconds': remaining,
                        'status': 'running'
                    })
                self.last_progress_time = time.time()
            
            time.sleep(5)
    
    def _health_check(self):
        """Perform periodic health checks"""
        while self.is_alive and self.process.poll() is None:
            try:
                # Check GPU memory
                gpu_stats = ResourceManager.check_gpu_memory()
                for gpu in gpu_stats:
                    if gpu['usage_percent'] > GPU_MEMORY_THRESHOLD * 100:
                        logger.warning(f"GPU {gpu['gpu_id']} memory usage high: {gpu['usage_percent']:.1f}%")
                
                # Check system memory
                mem = psutil.virtual_memory()
                if mem.percent > 90:
                    logger.warning(f"System memory usage high: {mem.percent}%")
                
                # Check disk space
                disk = psutil.disk_usage('/workspace')
                if disk.percent > 90:
                    logger.warning(f"Disk usage high: {disk.percent}%")
                    
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(HEALTH_CHECK_INTERVAL)
    
    def terminate(self):
        """Gracefully terminate the process"""
        self.is_alive = False
        if self.process.poll() is None:
            self.process.terminate()
            time.sleep(5)
            if self.process.poll() is None:
                self.process.kill()

class TrainingExecutor:
    """Execute and manage the training process"""
    
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config
        self.log_buffer = []
        self.checkpoints = []
        
    def execute(self, progress_callback=None) -> Dict[str, Any]:
        """Execute the training job"""
        cmd = [
            sys.executable,
            "/workspace/training/hpo_optuna.py",
            "--config", str(self.job_config.config_path)
        ]
        
        # Add resume flag if checkpoint exists
        checkpoint_dir = Path("/workspace/checkpoints") / self.job_config.task_id
        if checkpoint_dir.exists():
            cmd.append("--resume")
            logger.info("Resuming from checkpoint")
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Create process with proper configuration
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Start monitoring
        monitor = ProcessMonitor(process, self.job_config, progress_callback)
        monitor.start_monitoring()
        
        # Process output
        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                    
                line = line.rstrip()
                print(line)  # Print to RunPod logs
                self._process_log_line(line)
                
                # Keep buffer size manageable
                if len(self.log_buffer) > MAX_LOG_BUFFER_SIZE:
                    self.log_buffer.pop(0)
            
            # Wait for process completion
            return_code = process.wait()
            monitor.is_alive = False
            
            if return_code == 0:
                return self._handle_success()
            else:
                return self._handle_failure(f"Process exited with code {return_code}")
                
        except Exception as e:
            monitor.terminate()
            return self._handle_failure(f"Execution error: {str(e)}")
    
    def _process_log_line(self, line: str):
        """Process a log line for important information"""
        self.log_buffer.append(line)
        
        # Extract eval loss
        if "eval_loss" in line:
            logger.info(f"Progress: {line}")
        
        # Extract checkpoint saves
        if "Saving model checkpoint" in line:
            self.checkpoints.append(line)
        
        # Extract final model location
        if "model uploaded to" in line.lower() or "pushed to hub" in line.lower():
            logger.info(f"Model location: {line}")
    
    def _handle_success(self) -> Dict[str, Any]:
        """Handle successful completion"""
        # Find model artifacts
        model_artifacts = self._find_model_artifacts()
        
        return {
            "success": True,
            "task_id": self.job_config.task_id,
            "model_repo": self.job_config.expected_repo_name,
            "training_completed": datetime.now().isoformat(),
            "duration_hours": (datetime.now() - self.job_config.required_finish_time + 
                             timedelta(hours=self.job_config.hours_to_complete)).total_seconds() / 3600,
            "artifacts": model_artifacts,
            "checkpoints": self.checkpoints[-5:],  # Last 5 checkpoints
            "last_logs": '\n'.join(self.log_buffer[-200:])  # Last 200 lines
        }
    
    def _handle_failure(self, error_msg: str) -> Dict[str, Any]:
        """Handle training failure"""
        # Check for specific error types
        error_type = "unknown"
        if "OutOfMemoryError" in error_msg:
            error_type = "oom"
        elif "NCCL" in error_msg or "timeout" in error_msg.lower():
            error_type = "timeout"
        elif "CUDA" in error_msg:
            error_type = "cuda"
        
        return {
            "success": False,
            "task_id": self.job_config.task_id,
            "error": error_msg,
            "error_type": error_type,
            "checkpoints": self.checkpoints,  # All checkpoints for recovery
            "last_logs": '\n'.join(self.log_buffer[-500:])  # More logs for debugging
        }
    
    def _find_model_artifacts(self) -> Dict[str, str]:
        """Find and catalog model artifacts"""
        artifacts = {}
        
        # Check for optimized config
        opt_config = self.job_config.config_path.with_name(
            self.job_config.config_path.stem + "_opt.yml"
        )
        if opt_config.exists():
            artifacts["optimized_config"] = str(opt_config)
        
        # Check for final model
        output_root = Path("/workspace/outputs") / self.job_config.task_id
        if output_root.exists():
            model_dirs = list(output_root.glob("*/pytorch_model.bin")) + \
                        list(output_root.glob("*/model.safetensors"))
            if model_dirs:
                artifacts["final_model"] = str(model_dirs[-1].parent)
        
        # Check for HPO results
        hpo_db = output_root / "hpo.db"
        if hpo_db.exists():
            artifacts["hpo_results"] = str(hpo_db)
        
        return artifacts

@contextmanager
def error_handler():
    """Context manager for consistent error handling"""
    try:
        yield
    except Exception as e:
        logger.error(f"Unhandled exception: {traceback.format_exc()}")
        raise

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating cleanup...")
        ResourceManager.cleanup_resources()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

def handler(job):
    """
    Enhanced RunPod Serverless handler for training jobs
    
    Args:
        job (dict): Job specification from RunPod
        
    Returns:
        dict: Job results
    """
    start_time = time.time()
    setup_signal_handlers()
    
    try:
        # Validate and parse job configuration
        job_input = job.get("input", {})
        job_config = JobConfig.from_job_input(job_input)
        
        logger.info(f"Starting training job: {job_config.task_id}")
        logger.info(f"Configuration: model={job_config.model}, dataset={job_config.dataset}, "
                   f"hours={job_config.hours_to_complete}, hpo={job_config.hpo}")
        
        # Setup environment
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available")
        
        logger.info(f"Found {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        
        # Configure environment
        ResourceManager.setup_environment(num_gpus)
        
        # Initial resource cleanup
        ResourceManager.cleanup_resources()
        time.sleep(CLEANUP_WAIT_TIME)
        
        # Setup configuration
        from serverless_config_handler import setup_config
        setup_config(
            job_config.dataset,
            job_config.model,
            job_config.dataset_type,
            job_config.file_format,
            job_config.task_id,
            job_config.expected_repo_name,
            job_config.required_finish_time.isoformat(),
            job_config.testing,
            job_config.hpo
        )
        
        # Verify config was created
        if not job_config.config_path.exists():
            raise FileNotFoundError(f"Config file not created: {job_config.config_path}")
        
        # Log initial resource state
        gpu_stats = ResourceManager.check_gpu_memory()
        for gpu in gpu_stats:
            logger.info(f"GPU {gpu['gpu_id']}: {gpu['usage_percent']:.1f}% used")
        
        # Execute training with progress updates
        def progress_callback(status):
            """Send progress updates to RunPod"""
            runpod.serverless.progress_update(job, status)
        
        executor = TrainingExecutor(job_config)
        result = executor.execute(progress_callback)
        
        # Final cleanup
        ResourceManager.cleanup_resources()
        
        # Add execution metrics
        result["execution_time_seconds"] = time.time() - start_time
        result["gpu_count"] = num_gpus
        
        return result
        
    except ValueError as e:
        # Configuration errors
        logger.error(f"Configuration error: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "configuration",
            "task_id": job.get("input", {}).get("task_id", "unknown")
        }
        
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "unexpected",
            "traceback": traceback.format_exc(),
            "task_id": job.get("input", {}).get("task_id", "unknown")
        }
    
    finally:
        # Always cleanup resources
        try:
            ResourceManager.cleanup_resources()
        except:
            pass

# Start the serverless worker
if __name__ == "__main__":
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})