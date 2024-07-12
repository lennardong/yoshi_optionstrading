from datetime import datetime

from prefect import flow, task


@task
def print_current_time():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current time: {current_time}")
    return current_time


@flow(name="simple_flow")
def simple_flow():
    print_current_time()


if __name__ == "__main__":
    simple_flow()
