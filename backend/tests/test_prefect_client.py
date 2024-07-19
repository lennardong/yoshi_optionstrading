import asyncio
import inspect
import logging
import os
import subprocess
import threading
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from textwrap import dedent
from typing import Callable

from prefect import flow, get_client, task
from prefect.agent import PrefectAgent
from prefect.client import get_client
from prefect.client.orchestration import PrefectClient
from prefect.deployments import Deployment
from prefect.filesystems import RemoteFileSystem
from prefect.infrastructure import Process
from prefect.server.schemas.actions import WorkPoolCreate
from prefect.server.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner


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


async def get_flows(client: PrefectClient):
    flows = await client.read_flows()
    print(f"\n#### get_flows(): Found {len(flows)} flows:")
    for flow in flows:
        print(flow.name, flow.id)
    return flows


async def create_flow_and_run_from_deployment(flow, flow_name, params=None):
    async with get_client() as client:
        flow_id = await client.create_flow(flow)
        deployment = await client.create_deployment(
            flow_id=flow_id, name=f"{flow_name}-deployment", version="1"
        )
        flow_run = await client.create_flow_run_from_deployment(
            deployment_id=deployment, parameters=params
        )
    return flow_run


async def deploy_flow_v1(prefect_config: PrefectConfig, flow, params=None):
    async with get_client() as client:
        client: PrefectClient
        flow_id = await client.create_flow(flow)

        deployment = await client.create_deployment(
            flow_id=flow_id,
            name=f"{flow.name}-deployment",
            version="1",
            work_pool_name=prefect_config.workpool_name,
            storage_document_id=prefect_config.storage_block._block_document_id,
            parameters=params or {},
            is_schedule_active=False,  # Set this to True we want to activate scheduling
        )

        print(f"Deployment created with ID: {deployment}")
        print(f"Work Pool: {prefect_config.workpool_name}")
        print(f"Storage: {prefect_config.storage_block}")

        return deployment


# async def publish_flow_to_storage(prefect_config: PrefectConfig, flow_func):
#     flow_content = inspect.getsource(flow_func)
#     imports = "from prefect import flow\n\n"
#     full_content = imports + dedent(flow_content)
#     flow_path = f"{flow_func.__name__}.py"
#     await prefect_config.storage_block.write_path(flow_path, full_content.encode())
#     print(f"Flow '{flow_func.__name__}' published to storage at path: {flow_path}")


async def publish_flow_to_storage(prefect_config: PrefectConfig, flow_func):
    flow_content = inspect.getsource(flow_func)

    # Add header
    imports = "from prefect import flow\n\n"
    full_content = imports + dedent(flow_content)
    flow_path = f"{flow_func.__name__}.py"
    await prefect_config.storage_block.write_path(flow_path, full_content.encode())

    # Log
    logging.debug("publish_flow_to_strorage() - Published content:")
    logging.debug(full_content)

    # End
    print(f"Flow '{flow_func.__name__}' published to storage at path: {flow_path}")


async def deploy_flow(
    prefect_config: PrefectConfig, flow_func, params=None, run_flow=False
):
    # Publish flow to storage
    flow_path = f"{flow_func.__name__}.py"
    await publish_flow_to_storage(prefect_config, flow_func)

    # Deploy flow
    flow_instance = await flow.from_source(
        source=prefect_config.storage_block,
        entrypoint=f"{flow_path}:{flow_func.__name__}",
    )
    print("Flow Instance Spawned:", type(flow_instance), flow_instance)
    deployment_id = await flow_instance.deploy(
        name=f"{flow_func.__name__}-deployment",
        work_pool_name=prefect_config.workpool_name,
        parameters=params or {},
    )
    print("Deployment Instance Created:", type(deployment_id), deployment_id)

    print(f"Deployment created with ID: {deployment_id}")

    # Run flow if requested
    if run_flow:
        async with get_client() as client:
            client: PrefectClient
            flow_run = await client.create_flow_run_from_deployment(
                deployment_id=deployment_id, parameters=params
            )
            print(f"Started flow run with ID: {flow_run.id}")
            return deployment_id, flow_run.id

    return deployment_id, None


# NOTE this is not needed - to run a flow, just run the function
# async def run_flow(flow, prefect_config: PrefectConfig, params=None):
#     async with get_client() as client:
#         client: PrefectClient = client
#         try:
#             existing_flow = await client.read_flow_by_name(flow.name)
#             flow_id = existing_flow.id
#         except Exception:
#             flow_id = await client.create_flow(flow)

#         flow_run = await client.create_flow_run(flow=flow, parameters=params)

#         # Retrieve detailed information about the flow run
#         flow_run_details = await client.read_flow_run(flow_run.id)

#         print(f"Flow run created with ID: {flow_run.id}")
#         print(f"Infrastructure PID: {flow_run_details.infrastructure_pid}")
#         print(f"Work Pool Name: {flow_run_details.work_pool_name}")
#         print(f"Work Queue Name: {flow_run_details.work_queue_name}")
#         print(f"State: {flow_run_details.state}")

#     return flow_run


async def test_connection():
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


@flow(log_prints=True)
def example_flow(name: str = "World"):
    print(f"Hello, {name}!")


# Create and run the flow
if __name__ == "__main__":
    os.environ["PREFECT_API_URL"] = "127.0.0.1:8080/api"
    os.environ["PREFECT__BACKEND"] = "server"

    prefect_config = PrefectConfig()
    asyncio.run(test_connection())

    print("\n RUNNING A FLOW")
    example_flow()

    print("\n PUBLISHING A FLOW TO STORAGE")
    asyncio.run(publish_flow_to_storage(prefect_config, example_flow))

    print(f"storage doc id: {prefect_config.storage_block._block_document_id}")

    print("\n DEPLOYING A FLOW")
    result = asyncio.run(
        deploy_flow(
            prefect_config=prefect_config,
            flow_func=example_flow,
            params={"name": f"YESSSS! I AM A STRING!"},
            run_flow=True,
        )
    )

    print(result)

    print("\n DEPLOYING A FLOW WITH TASKS")
    result = asyncio.run(
        deploy_flow(
            prefect_config=prefect_config,
            flow_func=test_flow,
            params={"n": 12},
            run_flow=True,
        )
    )

    print(result)

    # flow_run = asyncio.run(
    #     create_flow_and_run_from_deployment(
    #         example_flow, "example-flow_named1", {"name": "Alice"}
    #     )
    # )
    # print(f"Created flow run via deployment: {flow_run.id}")

    # # flow_run = asyncio.run(create_flow_and_run(test_flow, "test_flow_named", {"n": 10}))
    # # print(f"Created flow run: {flow_run.id}")

    # client: PrefectClient = get_client()
    # flows = asyncio.run(get_flows(client))
    # test_flow(15)
