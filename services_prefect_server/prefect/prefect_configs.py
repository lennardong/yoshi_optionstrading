from prefect import config
from prefect.run_configs import DockerRun


def set_prefect_configs():
    # Setting server configurations
    config.server.host = "${PREFECT_SERVER_API_HOST}"
    config.server.port = "${PREFECT_SERVER_API_PORT}"

    # Setting database configurations
    config.database.connection_url = "${PREFECT_SERVER_DATABASE_URL}"

    # Setting API configurations
    config.api.url = "${PREFECT_API_URL}"

    # Setting telemetry configurations
    config.telemetry.enabled = False

    # Setting logging configurations
    config.logging.level = "${PREFECT_LOGGING_LEVEL}"

    # Setting flow configurations
    config.flows.checkpointing = True

    # Setting task default configurations
    config.tasks.defaults.retry_delay_seconds = 60
    config.tasks.defaults.retries = 3

    # Setting storage configurations
    config.storage.default_storage = "local"

    # Setting engine configurations
    # config.engine.flow_runner.default_policy = "cancel"


def print_prefect_configs():
    print("Server Configurations:")
    print(f"Host: {config.server.host}")
    print(f"Port: {config.server.port}")

    print("\nDatabase Configurations:")
    print(f"Connection URL: {config.database.connection_url}")

    print("\nAPI Configurations:")
    print(f"URL: {config.api.url}")

    print("\nTelemetry Configurations:")
    print(f"Enabled: {config.telemetry.enabled}")

    print("\nLogging Configurations:")
    print(f"Level: {config.logging.level}")

    print("\nFlow Configurations:")
    print(f"Checkpointing: {config.flows.checkpointing}")

    print("\nTask Default Configurations:")
    print(f"Retry Delay Seconds: {config.tasks.defaults.retry_delay_seconds}")
    print(f"Retries: {config.tasks.defaults.retries}")

    print("\nStorage Configurations:")
    print(f"Default Storage: {config.storage.default_storage}")

    print("\nEngine Configurations:")
    print(f"Flow Runner Default Policy: {config.engine.flow_runner.default_policy}")


if __name__ == "__main__":
    set_prefect_configs()
    print_prefect_configs()
