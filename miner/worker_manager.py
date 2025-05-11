import os
import sys

# Add project root directory to sys.path to allow importing 'core'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import subprocess
import time
import signal
import os
import sys
from fiber.logging_utils import get_logger
from core import constants as cst

logger = get_logger(__name__)

# Define GPU pairs for an 8xH100 system
GPU_PAIRS = [
    "0"
]

# RQ queue name to listen to
QUEUE_NAME = "default"

worker_processes = []

def signal_handler(sig, frame):
    """Handle termination signals to gracefully shut down workers."""
    logger.info("Received termination signal. Stopping RQ workers...")
    for p in worker_processes:
        if p.poll() is None: # Check if process is still running
            try:
                # Send SIGTERM first for graceful shutdown
                p.terminate()
                logger.info(f"Sent SIGTERM to worker PID {p.pid}")
            except ProcessLookupError:
                logger.warning(f"Worker PID {p.pid} not found.")
    
    # Wait a bit for processes to terminate
    time.sleep(5)

    # Force kill any remaining processes
    for p in worker_processes:
         if p.poll() is None:
            try:
                p.kill()
                logger.warning(f"Sent SIGKILL to worker PID {p.pid}")
            except ProcessLookupError:
                 pass # Already gone

    logger.info("All workers stopped.")
    sys.exit(0)

def start_workers():
    """Launch RQ worker processes, one for each GPU pair."""
    # Construct Redis URL, including password if it exists
    if cst.REDIS_PASSWORD:
        redis_url = f"redis://:{cst.REDIS_PASSWORD}@{cst.REDIS_HOST}:{cst.REDIS_PORT}/0"
    else:
        redis_url = f"redis://{cst.REDIS_HOST}:{cst.REDIS_PORT}/0"
        
    logger.info(f"Starting {len(GPU_PAIRS)} RQ workers for queue '{QUEUE_NAME}' on Redis at {cst.REDIS_HOST}:{cst.REDIS_PORT}...") # Don't log password

    for gpu_pair in GPU_PAIRS:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_pair
        
        # Command to start the RQ worker, including Redis URL
        # Assumes 'rq' is in the system PATH or virtual environment
        command = ["rq", "worker", "--url", redis_url, QUEUE_NAME]
        
        logger.info(f"Launching worker with CUDA_VISIBLE_DEVICES={gpu_pair} using command: {' '.join(command)}")
        
        try:
            # Use Popen for non-blocking execution
            process = subprocess.Popen(command, env=env, stdout=sys.stdout, stderr=sys.stderr)
            worker_processes.append(process)
            logger.info(f"Launched worker PID {process.pid} for GPUs {gpu_pair}")
        except FileNotFoundError:
             logger.error(f"Error: 'rq' command not found. Make sure RQ is installed and in your PATH.")
             # Stop any already started workers before exiting
             signal_handler(signal.SIGTERM, None)
        except Exception as e:
            logger.error(f"Failed to launch worker for GPUs {gpu_pair}: {e}")
            signal_handler(signal.SIGTERM, None)


    logger.info("All workers launched. Monitoring...")
    
    # Keep the main script running and monitor workers
    # Basic monitoring: just wait for interrupt
    while True:
        # Optional: Add more sophisticated monitoring here, e.g., check p.poll() and restart if needed
        time.sleep(60) 

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    start_workers()