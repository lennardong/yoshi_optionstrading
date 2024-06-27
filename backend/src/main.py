# from config import settings
from fastapi import FastAPI

import mlflow, 
# from prefect import Client

app = FastAPI()

# mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
# prefect_client = Client(api_server=settings.PREFECT_API_URL)


@app.get("/")
def read_root():
    return {"message": "Welcome to Aleph API"}


@app.get("/data")
def get_data():
    # This is where you'd interact with MLflow or Prefect
    return {"data": "Some data from the backend"}


print("hello world")
