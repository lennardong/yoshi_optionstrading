import logging
import os
from datetime import datetime

from prefect import flow, task
from prefect.deployments import Deployment, run_deployment
from prefect.filesystems import RemoteFileSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://localhost:8080/api")
REMOTE_STORAGE_HOST = os.getenv("REMOTE_STORAGE_HOST", "http://localhost:8080")


@task
async def print_current_time():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Current time: {current_time}")
    return current_time


@flow(name="simple_flow")
async def simple_flow():
    time = await print_current_time()


async def create_and_run_deployment():
    remote_storage = RemoteFileSystem(
        basepath=f"{REMOTE_STORAGE_HOST}/flows/",
    )

    try:
        deployment = await Deployment.build_from_flow(
            flow=simple_flow,
            name="simple_flow_deployment",
            version="1",
            work_queue_name="default",
            tags=["demo"],
            storage=remote_storage,
        )

        deployment_id = await deployment.apply()
        logger.info(f"Deployment created with ID: {deployment_id}")

        run = await run_deployment(name="simple_flow/simple_flow_deployment")
        logger.info(f"Flow run created with ID: {run.id}")
    except Exception as e:
        logger.error(f"Error creating or running deployment: {e}")
        raise


if __name__ == "__main__":
    import asyncio

    asyncio.run(create_and_run_deployment())
    logger.info(
        f"Flow 'simple_flow' has been registered with the Prefect server at {PREFECT_API_URL}"
    )
