import asyncio
import logging
import os
from dataclasses import asdict, dataclass, fields

import numpy as np
from model import mlflow_optimize_model
from prefect import flow, get_client, task
from prefect.deployments import Deployment
from prefect.filesystems import RemoteFileSystem
from prefect.infrastructure import Process
from prefect.server.schemas.actions import WorkPoolCreate
from prefect.server.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner

# from prefect_aws.s3 import S3Bucket

s3_bucket_block = S3Bucket.load("minio-block")
print(s3_bucket_block)


@dataclass
class PrefectConfig:
    """
    Represents the configuration for the Prefect setup, including the storage, work pool, and Prefect server URL.

    The `PrefectConfig` class sets up the necessary Prefect infrastructure, including a MinIO storage block and a work pool, during initialization.

    The `setup_prefect` method is an asynchronous function that performs the following tasks:
    - Sets the `PREFECT_API_URL` environment variable to the configured Prefect server URL.
    - Creates a MinIO storage block with the specified settings.
    - Checks if the configured work pool already exists, and creates it if it doesn't.
    """

    storage_block: RemoteFileSystem = None
    workpool_name: str = "my-work-pool"
    prefect_server_url: str = "http://127.0.0.1:4040/api"
    s3_url: str = "http://127.0.0.1:9010"

    def __post_init__(self):
        asyncio.run(self.setup_prefect())

    async def setup_prefect(
        self,
        storage_block_name="minio-storage",
    ):
        logging.info("\n#### Setting up Prefect infrastructure...")
        os.environ["PREFECT_API_URL"] = self.prefect_server_url
        logging.info(f"Setting PREFECT_API_URL to {self.prefect_server_url}")

        # Set up MinIO storage
        minio_block = RemoteFileSystem(
            basepath="s3://prefect-flows",
            settings={
                "key": "minioadmin",
                "secret": "minioadmin",
                "client_kwargs": {"endpoint_url": self.s3_url},
            },
        )
        await minio_block.save(storage_block_name, overwrite=True)
        self.storage_block = minio_block
        logging.info(
            f"Created MinIO storage block '{storage_block_name}', {self.storage_block}."
        )

        # Create or get work pool
        async with get_client() as client:
            workpools = await client.read_work_pools()
            print(workpools)
            try:
                await client.read_work_pool(self.workpool_name)
                logging.info(f"Work pool '{self.workpool_name}' already exists.")
            except Exception:
                work_pool_create = WorkPoolCreate(
                    name=self.workpool_name,
                    type="process",
                )
                await client.create_work_pool(work_pool_create)
                logging.info(f"Work pool '{self.workpool_name}' created.")


def create_deployment(config: PrefectConfig, flow_func, deployment_name_suffix=None):
    flow_name = flow_func.name
    deployment_name = (
        f"{flow_name}-{deployment_name_suffix}" if deployment_name_suffix else flow_name
    )

    deployment = Deployment.build_from_flow(
        flow=flow_func,
        name=deployment_name,
        work_pool_name=config.workpool_name.name,
        storage=config.storage_block,
        infrastructure=Process(),
        schedule=CronSchedule(cron="0 0 * * *"),  # Run daily at midnight
    )
    deployment.apply()
    print(f"Deployment '{deployment_name}' created for flow '{flow_name}'.")
    return deployment


# ... rest of the code remains the same


def setup_prefect():
    pass


@task
def check_reoptimization_criteria(
    monitoring_results, rmse_threshold, directionality_threshold
):
    if len(monitoring_results) < 3:
        return False

    recent_results = monitoring_results[-3:]
    wrong_directions = sum(
        1 for result in recent_results if result["directionality"] < 0.5
    )

    rmse_values = [result["rmse"] for result in monitoring_results]
    rmse_mean = np.mean(rmse_values)
    rmse_std = np.std(rmse_values)
    scaled_rmse_threshold = rmse_mean + rmse_threshold * rmse_std

    return (
        wrong_directions >= directionality_threshold
        or recent_results[-1]["rmse"] > scaled_rmse_threshold
    )


@flow(task_runner=SequentialTaskRunner())
def monitor_and_reoptimize(
    X, y, predictions, actuals, rmse_threshold=2, directionality_threshold=3
):
    monitoring_result = monitor_model_performance(predictions, actuals)
    monitoring_results.append(monitoring_result)

    reoptimize = check_reoptimization_criteria(
        monitoring_results, rmse_threshold, directionality_threshold
    )

    if reoptimize:
        best_params = mlflow_optimize_model(X, y)
        return best_params
    else:
        return None


############
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    prefect = PrefectConfig()
    print(prefect)
