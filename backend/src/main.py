# from config import settings
import logging
from datetime import datetime, timedelta
from time import sleep

from data import DataSchema, load_df
from model import (
    mlflow_get_prod_model,
    mlflow_optimize_model,
    mlflow_promote_model,
    mlflow_register_model,
    mlflow_train_model,
    monitor_model,
    run_predictions_on_df,
    setup_mlflow,
)
from orchaestrate import PrefectConfig, create_flow_deployment, run_deployments
from prefect import flow, task

# Global variables
STOCK_SYMBOL = "AAPL"
N_TRADING_DAYS = 30
ALPHA_VANTAGE_API_KEY = "7H8XHMRSGISFBKK7"
MODEL_NAME = "LAUNCH_MODEL"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup MLflow
setup_mlflow()

# Initialize Prefect
prefect_config = PrefectConfig()


@task
def load_and_process_data():
    df = load_df(STOCK_SYMBOL, datetime.now(), N_TRADING_DAYS, ALPHA_VANTAGE_API_KEY)
    model = mlflow_get_prod_model(MODEL_NAME)
    df = run_predictions_on_df(df, model)
    return df, model


@flow
def daily_flow():
    df, model = load_and_process_data()
    df = monitor_model(df, datetime.now(), evaluation_period=5)
    new_model = mlflow_train_model(
        df, model, end_date=datetime.now(), training_window=5
    )
    mlflow_register_model(new_model, MODEL_NAME)
    mlflow_promote_model(MODEL_NAME)
    return df


@flow
def optimization_flow():
    df, _ = load_and_process_data()
    model = mlflow_optimize_model(df, datetime.now(), n_trials=100)
    mlflow_register_model(model, MODEL_NAME)
    mlflow_promote_model(MODEL_NAME)


@flow
def monitoring_flow():
    df, model = load_and_process_data()
    monitoring_results = monitor_model(df, datetime.now(), evaluation_period=5)
    if (
        monitoring_results[DataSchema.monitor_data_drift]
        or monitoring_results[DataSchema.monitor_target_drift]
    ):
        optimization_flow()


if __name__ == "__main__":
    deployments = [
        create_flow_deployment(
            daily_flow, "daily_flow", tags=["daily", "training"], cron="0 0 * * *"
        ),
        create_flow_deployment(
            optimization_flow,
            "weekly_flow",
            tags=["weekly", "optimization"],
            cron="0 0 * * 0",
        ),
        create_flow_deployment(
            monitoring_flow, "monitoring_flow", tags=["monitoring"], cron="0 */6 * * *"
        ),
    ]
    run_deployments(deployments)

    # Keep the script running
    while True:
        logger.info("Main process is running. Press Ctrl+C to exit.")
        try:
            # Sleep for a day to reduce CPU usage
            sleep(86400)
        except KeyboardInterrupt:
            logger.info("Received exit signal. Shutting down.")
            break
