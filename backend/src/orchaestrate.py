import asyncio
import logging
import os
import subprocess
import threading
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from textwrap import dedent
from time import sleep
from typing import Callable, Dict, List, Optional

import numpy as np
from model import mlflow_optimize_model
from prefect import flow, get_client, task
from prefect.client import get_client
from prefect.client.orchestration import PrefectClient
from prefect.context import FlowRunContext
from prefect.deployments import Deployment
from prefect.filesystems import RemoteFileSystem
from prefect.infrastructure import Process
from prefect.server.schemas.actions import WorkPoolCreate
from prefect.server.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner

# from prefect_aws import AwsCredentials

####################


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
    workpool_name: str = f"workpool_test_autorun"
    prefect_server_url: str = "http://127.0.0.1:8080/api"
    s3_url: str = "http://127.0.0.1:9010"

    def __post_init__(self):
        logging.info("\n#### Setting up Prefect infrastructure...")
        # asyncio.run(self.setup_prefect())
        asyncio.run(self.setup_config())
        test = asyncio.run(self.test_connection())
        if test == False:
            raise Exception("Prefect connection failed")
        asyncio.run(self.setup_storage())
        asyncio.run(self.setup_workpool())
        self.start_worker()

    def start_worker(self):
        """
        Starts a Prefect worker in a separate thread on client side, which will pull tasks from the configured work pool and execute them.
        The `start_worker` method creates a new thread that runs the `prefect worker start` command with the configured work pool name.
        This allows the worker to start processing tasks in the background without blocking the main program execution.
        """
        worker_command = f"prefect worker start --pool {self.workpool_name}"
        threading.Thread(
            target=lambda: subprocess.run(worker_command, shell=True), daemon=True
        ).start()

    async def setup_config(self):
        os.environ["PREFECT_API_URL"] = self.prefect_server_url
        os.environ["PREFECT__BACKEND"] = "server"
        logging.info(f"Setting PREFECT_API_URL to {self.prefect_server_url}")

    async def setup_storage(self, storage_block_name="minio-storage"):
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

    async def setup_workpool(self):
        async with get_client() as client:
            workpools = await client.read_work_pools()
            print(workpools)
            try:
                await client.read_work_pool(self.workpool_name)
                logging.info(f"Work pool '{self.workpool_name}' already exists.")
            except Exception:
                workpool = WorkPoolCreate(
                    name=self.workpool_name,
                    type="process",
                )
                await client.create_work_pool(workpool)
                logging.info(f"Work pool '{self.workpool_name}' created.")

    async def test_connection(self):
        async with get_client() as client:
            try:
                # This will ping the server and return basic information
                health_info = await client.api_healthcheck()
                print("Connection successful!")
                print("Server health info:", health_info)
                return True
            except Exception as e:
                print("Connection failed:", str(e))
                return False


async def get_flows():
    async with get_client() as client:
        flows = await client.read_flows()
        print(f"\n#### get_flows(): Found {len(flows)} flows:")
        for flow in flows:
            print(flow.name, flow.id)
        return flows


async def get_deployments():
    async with get_client() as client:
        deployments = await client.read_deployments()
        print(f"\n#### get_deployments(): Found {len(deployments)} deployments:")
        for deployment in deployments:
            print(deployment.name, deployment.id)
        return deployments


async def create_flow_deployment(
    flow_: Callable,
    name: Optional[str] = None,
    tags: List[str] = None,
    cron: Optional[str] = None,
    description: Optional[str] = None,
    **flow_args,
):
    """
    Creates a new flow deployment in the system.

    Args:
        flow_ (Callable): The flow function to deploy.
        name (Optional[str]): The name of the deployment. If not provided, it will be generated from the flow name.
        tags (List[str]): A list of tags to associate with the deployment.
        cron (Optional[str]): A cron expression to schedule the deployment.
        description (Optional[str]): A description for the deployment.
        **flow_args: Additional arguments to pass to the flow function.

    Returns:
        The created deployment.
    """
    flow_name = flow_.__name__
    deployment_name = name or f"{flow_name}_deployment"

    existing_deployments = await get_deployments()
    matching_deployments = [
        d for d in existing_deployments if d.name == deployment_name
    ]
    version = max([int(d.version) for d in matching_deployments], default=0) + 1

    deployment = await flow_.serve(
        name=deployment_name,
        tags=tags,
        schedule=(CronSchedule(cron=cron) if cron else None),
        description=description,
        version=str(version),
        parameters=flow_args,
    )
    logging.info(
        f"Flow {flow_name} deployment created and served: {deployment_name} (version {version})"
    )
    return deployment


def run_deployments(deployments):
    """
    Runs a list of deployments in parallel using separate threads.

    Args:
        deployments (list): A list of deployment functions to run.

    Returns:
        list: A list of results from running the deployments.
    """

    def _run_deployment(deployment):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(deployment)
        loop.close()
        return result

    threads = []
    results = []
    for deployment in deployments:
        thread = threading.Thread(
            target=lambda: results.append(_run_deployment(deployment))
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results


####################
# OLD


def z_serve_flow_threading(
    flow_: Callable,
    name: Optional[str] = None,
    tags: List[str] = None,
    cron: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    **flow_args,
):
    flow_name = flow_.__name__
    deployment_name = name or f"{flow_name}_deploy"

    def deploy():
        asyncio.run(async_deploy())

    async def async_deploy():
        async with get_client() as client:
            deployment = await flow_.serve(
                name=deployment_name,
                tags=tags,
                schedule=(CronSchedule(cron=cron) if cron else None),
                description=description,
                version=version,
                parameters=flow_args,
            )
            logging.info(f"Flow {flow_name} served successfully as {deployment_name}")

    thread = threading.Thread(target=deploy, daemon=True)
    thread.start()
    logging.info(f"Started deployment of {flow_name} in background")

    return thread


def z_run_deployments_no_thread(deployments):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(asyncio.gather(*deployments))
    loop.close()
    return results


def z_run_deployments_needswork(deployments):
    # FIXME - trying to run it as background process does not work
    def deploy_in_background(deployment_coroutine):
        async def _deploy():
            try:
                deployment = await deployment_coroutine
                with FlowRunContext.from_flow(deployment.flow):
                    result = await deployment
                logging.info(f"Deployment completed: {result.name}")
            except Exception as e:
                logging.error(f"Deployment failed: {str(e)}")

        asyncio.run(_deploy())

    threads = []
    for deployment in deployments:
        thread = threading.Thread(target=deploy_in_background, args=(deployment,))
        thread.start()
        threads.append(thread)

    logging.info(f"Started {len(threads)} deployment threads in the background")

    return threads


####################


@flow(
    name=f"test_flow",
    log_prints=True,
)
def test_flow(n):
    for i in range(n):
        res = mock_task(i)
        print(f"Mocktask returns: {res}")


@task
def mock_task(x):
    return x


@task
def check_outliers():
    print("checking outliers")


@task
def optimize_params():
    sleep(3)
    print("optimizing params")


@flow(log_prints=True)
def run_optimization_flow(tag: str = "aleph job"):
    print(f"{tag}..... running")
    check_outliers()
    optimize_params()


####################

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print("\n##### Starting Prefect client...")
    prefect = PrefectConfig()
    print(prefect)

    print("\n##### Testing flows on")
    flows = asyncio.run(get_flows())
    print(type(flows), flows)

    print("\n##### Testing deployments")
    deployments = asyncio.run(get_deployments())
    print(type(deployments), deployments)

    print("\n###### Serve Flow")
    deploy1 = create_flow_deployment(
        flow_=test_flow,
        name="test_flow_deploy1",
        tags=["testing", "aleph"],
        cron="* * * * *",
        description="Testing flow",
        n=10,
    )
    deploy2 = create_flow_deployment(
        flow_=test_flow,
        name="test_flow_deploy2",
        tags=["testing", "aleph"],
        cron="* * * * *",
        description="Testing flow",
        n=100,
    )

    run_deployments([deploy1, deploy2])
    print(f"Deployed  flows")
