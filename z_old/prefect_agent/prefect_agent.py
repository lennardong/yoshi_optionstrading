from prefect import config
from prefect.agent.local import LocalAgent

if __name__ == "__main__":
    agent = LocalAgent()
    agent.start()
