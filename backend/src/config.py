from pydantic import BaseSettings

class Settings(BaseSettings):
    MLFLOW_TRACKING_URI: str
    PREFECT_API_URL: str

settings = Settings()
