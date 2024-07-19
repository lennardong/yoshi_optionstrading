"""
# TODO - implement AWS S3 storage for simpler code storage. Do NOT use fileblocks, they are going to be deprecated.

"""

from prefect import flow
from prefect.deployments import Deployment
from prefect_aws import AwsCredentials

AwsCredentials(
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
    aws_session_token=None,  # Optional, set to None for MinIO
    region_name=None,  # Optional, set to None for MinIO
    endpoint_url="http://127.0.0.1:9010",  # Your MinIO endpoint
).save("minio-credentials")


@flow
def example_flow(name: str = "World"):
    print(f"Hello, {name}!")


def deploy_flow():
    deployment = Deployment.build_from_flow(
        flow=example_flow,
        name="example-deployment",
        work_queue_name="default",
        storage=None,  # Storage is handled by prefect.yaml
    )
    deployment.apply()


if __name__ == "__main__":
    deploy_flow()
