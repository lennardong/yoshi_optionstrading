FROM python:3.8

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false     && poetry install --no-dev --no-interaction --no-ansi

COPY src/ ./src/

CMD ["python", "src/app.py"]
