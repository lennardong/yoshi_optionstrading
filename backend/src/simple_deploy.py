import os
from datetime import timedelta

from prefect import flow, get_client, task
from prefect.client.schemas.schedules import IntervalSchedule
from prefect.deployments import Deployment

# Set up Prefect config
os.environ["PREFECT_API_URL"] = "http://localhost:8080/api"


@task
def say_hello(name):
    print(f"Task say_hello is running with name: {name}")
    return f"Hello, {name}!"


@flow
def python_flow(name: str = "from Python"):
    print(f"Flow python_flow is running with name: {name}")
    result = say_hello(name)
    print(f"Flow result: {result}")
    return result


async def main():
    try:
        # Create a client to interact with the server
        async with get_client() as client:
            print("Client created successfully")

            # Test the connection
            health = await client.api_healthcheck()
            print(f"Server health check: {health}")

            # Create a work pool if it doesn't exist
            work_pool_name = "default"
            try:
                work_pool = await client.read_work_pool(work_pool_name)
                if not work_pool:
                    await client.create_work_pool(name=work_pool_name)
                    print(f"Work pool '{work_pool_name}' created")
                else:
                    print(f"Work pool '{work_pool_name}' already exists")
            except Exception as e:
                print(f"Error reading/creating work pool: {str(e)}", exc_info=True)
                return

            # Create and apply deployment
            try:
                deployment = await Deployment.build_from_flow(
                    flow=python_flow,
                    name="python-first-deployment",
                    work_queue_name=work_pool_name,
                    schedules=[IntervalSchedule(interval=timedelta(seconds=50))],
                )
                deployment_id = await deployment.apply()
                print(f"Deployment created with ID: {deployment_id}")
            except Exception as e:
                print(f"Error creating/applying deployment: {str(e)}", exc_info=True)
                return

            # List deployments to verify
            try:
                deployments = await client.read_deployments()
                print(f"Existing deployments: {[d.name for d in deployments]}")
            except Exception as e:
                print(f"Error reading deployments: {str(e)}", exc_info=True)

        # Run the flow synchronously
        print("Running the flow synchronously")
        flow_run = python_flow()
        print(f"Flow run result: {flow_run}")

    except Exception as e:
        print(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
