import os
import time
from datetime import datetime

from prefect import flow, task


@task
def print_hello_world():
    print(f"hello_world: {datetime.now()}")


@flow
def hello_world_flow(interval: int):
    while True:
        print_hello_world()
        time.sleep(interval)


if __name__ == "__main__":
    interval = int(os.getenv("N", 5))  # Default to 5 seconds if N is not set
    hello_world_flow(interval)
