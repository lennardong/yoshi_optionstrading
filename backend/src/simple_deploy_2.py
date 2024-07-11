import os
from time import sleep

import httpx
from prefect import flow, task

# Set the Prefect API URL to your remote server
os.environ["PREFECT_API_URL"] = "http://localhost:8080/api"


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


if __name__ == "__main__":
    run_optimization_flow.serve(
        name="optimize_params",
        cron="* * * * *",
        tags=["testing", "aleph"],
        version="v1.0",
    )
