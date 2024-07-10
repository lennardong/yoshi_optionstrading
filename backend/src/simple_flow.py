import asyncio
import logging
import os

from prefect import flow, task
from prefect.client import get_client
from prefect.deployments import Deployment

logging.basicConfig(level=logging.INFO)

os.environ["PREFECT_API_URL"] = "http://localhost:8080/api"


@task
def say_hello(name):
    return f"Hello, {name}!"


@flow
def hello_flow(name: str = "World"):
    result = say_hello(name)
    return result


async def main():
    try:
        # Create a client to interact with the server
        client = get_client()

        # Test the connection
        health = await client.api_healthcheck()
        logging.info(f"Server health check: {health}")

        # Create and apply deployment
        deployment = await Deployment.build_from_flow(
            flow=hello_flow,
            name="hello-flow-deployment",
            version="1",
        )
        deployment_id = await deployment.apply()
        logging.info(f"Deployment created with ID: {deployment_id}")

        # List deployments to verify
        deployments = await client.read_deployments()
        logging.info(f"Existing deployments: {[d.name for d in deployments]}")

        # Run the flow synchronously
        result = hello_flow()
        logging.info(f"Flow result: {result}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
