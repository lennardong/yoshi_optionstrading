import os

import httpx
from prefect import flow

# Set the Prefect API URL to your remote server
os.environ["PREFECT_API_URL"] = "http://localhost:8080/api"


@flow(log_prints=True)
def get_repo_info(repo_name: str = "PrefectHQ/prefect"):
    # url = f"https://api.github.com/repos/{repo_name}"
    # response = httpx.get(url)
    # response.raise_for_status()
    # repo = response.json()
    print(f"{repo_name} repository statistics 🤓:")


if __name__ == "__main__":
    get_repo_info.serve(
        name="my-first-deployment",
        cron="* * * * *",
        tags=["testing", "tutorial"],
        description="Given a GitHub repository, logs repository statistics for that repo.",
        version="tutorial/deployments",
    )
