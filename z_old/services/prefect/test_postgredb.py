import asyncio
import os

from sqlalchemy.ext.asyncio import create_async_engine


async def test_connection():
    database_url = os.getenv("PREFECT_SERVER_DATABASE_URL")
    engine = create_async_engine(database_url, echo=True)

    try:
        async with engine.connect() as connection:
            print("Connected to PostgreSQL")
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")


asyncio.run(test_connection())
