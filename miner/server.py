import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from fiber.logging_utils import get_logger
from fiber.miner.core import configuration

# from miner.config import factory_worker_config # Removed
from miner.endpoints.tuning import factory_router as tuning_factory_router
# from miner.logic.training_worker import load_job_backups # Removed - File deleted


logger = get_logger(__name__)


def factory_app(debug: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = configuration.factory_config()
        metagraph = config.metagraph
        sync_thread = None
        if metagraph.substrate is not None:
            sync_thread = threading.Thread(target=metagraph.periodically_sync_nodes, daemon=True)
            sync_thread.start()
        
        logger.info("Miner server started. Ready to enqueue jobs to RunPod.")

        yield

        logger.info("Shutting down...")

        metagraph.shutdown()
        if metagraph.substrate is not None and sync_thread is not None:
            sync_thread.join()

    app = FastAPI(lifespan=lifespan, debug=debug)

    # Middleware to restrict access by client IP
    ALLOWED_IPS = ["185.141.218.75"]

    @app.middleware("http")
    async def ip_restrict_middleware(request: Request, call_next):
        client_ip = request.client.host
        if client_ip not in ALLOWED_IPS:
            logger.info(f"Access denied from IP {client_ip}.")
            return JSONResponse(
                status_code=403,
                content={"detail": f"Access denied from IP {client_ip}."}
            )
        return await call_next(request)

    return app


logger = get_logger(__name__)

app = factory_app(debug=True)


tuning_router = tuning_factory_router()

app.include_router(tuning_router)

# if os.getenv("ENV", "prod").lower() == "dev":
#    configure_extra_logging_middleware(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7999)
