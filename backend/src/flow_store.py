import os

import requests
from prefect import flow
from prefect.filesystems import LocalFileSystem

# Define a local file system storage block
storage = LocalFileSystem(basepath="/tmp/flows")

# Define the Prefect server URL (using HTTP as per your note)
PREFECT_FLOWS_URL = "http://localhost:8080/flows"


@flow
def my_flow():
    print("Hello, Prefect!")


# Ensure the directory exists
os.makedirs(storage.basepath, exist_ok=True)

# Save the flow locally
flow_name = "my_flow"
flow_path = os.path.join(storage.basepath, f"{flow_name}.py")

with open(flow_path, "w") as f:
    f.write(my_flow.serialize())


def upload_to_caddy(file_path):
    url = f"{PREFECT_FLOWS_URL}/{os.path.basename(file_path)}"
    try:
        with open(file_path, "rb") as file:
            response = requests.post(url, files={"file": file})
        if response.status_code == 200:
            print("File uploaded successfully")
        else:
            print(f"Failed to upload file. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error occurred while uploading: {str(e)}")
    except IOError as e:
        print(f"Error occurred while reading the file: {str(e)}")


# Call the upload function
upload_to_caddy(flow_path)
