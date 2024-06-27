#!/bin/bash

# Create directory structure
mkdir -p backend/src frontend/src mlflow prefect .devcontainer

# Backend files
cat > backend/Dockerfile << EOL
FROM python:3.9-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY src/ ./src/

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

cat > backend/pyproject.toml << EOL
[tool.poetry]
name = "aleph-backend"
version = "0.1.0"
description = "Aleph backend application"

[tool.poetry.dependencies]
python = "^3.9"
fastapi = "^0.68.0"
uvicorn = "^0.15.0"
mlflow = "^1.20.2"
prefect = "^0.15.0"

[tool.poetry.dev-dependencies]
pytest = "^6.2.5"
EOL

cat > backend/src/main.py << EOL
from fastapi import FastAPI
import mlflow
from prefect import Client
from config import settings

app = FastAPI()

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
prefect_client = Client(api_server=settings.PREFECT_API_URL)

@app.get("/")
def read_root():
    return {"message": "Welcome to Aleph API"}

@app.get("/data")
def get_data():
    # This is where you'd interact with MLflow or Prefect
    return {"data": "Some data from the backend"}
EOL

cat > backend/src/config.py << EOL
from pydantic import BaseSettings

class Settings(BaseSettings):
    MLFLOW_TRACKING_URI: str
    PREFECT_API_URL: str

settings = Settings()
EOL

# Frontend files
cat > frontend/Dockerfile << EOL
FROM python:3.9-slim

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY src/ ./src/

CMD ["python", "src/app.py"]
EOL

cat > frontend/pyproject.toml << EOL
[tool.poetry]
name = "aleph-frontend"
version = "0.1.0"
description = "Aleph frontend application"

[tool.poetry.dependencies]
python = "^3.9"
dash = "^2.0.0"
requests = "^2.26.0"

[tool.poetry.dev-dependencies]
pytest = "^6.2.5"
EOL

cat > frontend/src/app.py << EOL
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import requests

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Welcome to Aleph'),
    html.Div(id='data-display'),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('data-display', 'children'),
              Input('interval-component', 'n_intervals'))
def update_data(n):
    response = requests.get('http://backend:8000/data')
    return f"Data from backend: {response.json()['data']}"

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
EOL

# MLflow Dockerfile
cat > mlflow/Dockerfile << EOL
FROM python:3.9-slim

RUN pip install mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0"]
EOL

# Prefect Dockerfile
cat > prefect/Dockerfile << EOL
FROM python:3.9-slim

RUN pip install prefect

EXPOSE 4200

CMD ["prefect", "server", "start"]
EOL

# Docker Compose files
cat > docker-compose.yml << EOL
version: '3.8'

services:
  backend:
    build: ./backend
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - PREFECT_API_URL=http://prefect:4200/api
    depends_on:
      - mlflow
      - prefect

  frontend:
    build: ./frontend
    ports:
      - "8050:8050"
    depends_on:
      - backend

  mlflow:
    build: ./mlflow
    volumes:
      - mlflow_data:/mlflow

  prefect:
    build: ./prefect
    volumes:
      - prefect_data:/root/.prefect

volumes:
  mlflow_data:
  prefect_data:
EOL

cat > docker-compose.dev.yml << EOL
version: '3.8'

services:
  backend:
    volumes:
      - ./backend:/app
    command: poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"

  frontend:
    volumes:
      - ./frontend:/app
    command: poetry run python src/app.py
    environment:
      - BACKEND_URL=http://backend:8000

  mlflow:
    ports:
      - "5000:5000"

  prefect:
    ports:
      - "4200:4200"
EOL

# DevContainer configuration
cat > .devcontainer/devcontainer.json << EOL
{
    "name": "Aleph Development",
    "dockerComposeFile": [
        "../docker-compose.yml",
        "../docker-compose.dev.yml"
    ],
    "service": "backend",
    "workspaceFolder": "/app",
    "settings": {
        "terminal.integrated.shell.linux": "/bin/bash",
        "python.pythonPath": "/usr/local/bin/python"
    },
    "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance"
    ],
    "forwardPorts": [8000, 8050, 5000, 4200],
    "postCreateCommand": "poetry install"
}
EOL

# Environment variables file
cat > .env << EOL
MLFLOW_TRACKING_URI=http://mlflow:5000
PREFECT_API_URL=http://prefect:4200/api
EOL

echo "Project structure and files have been created successfully!"
