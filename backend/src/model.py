import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Union

# Mlflow
import mlflow
import numpy as np
import optuna
import pandas as pd

# Internal
from data import DataSchema, load_df

# Mlflow
from mlflow.tracking import MlflowClient

# Scipy
from scipy.stats import ks_2samp, wasserstein_distance

# SKlearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

####################


@dataclass
class OptionsModel:
    selected_features: list = None
    n_lagged_days: int = None
    model: LinearRegression = None
    time_optimized: datetime = None
    time_trained: datetime = None

    def to_json(self):
        # Convert the dataclass to a dictionary and then to a JSON string
        return json.dumps(asdict(self), default=str)

    @staticmethod
    def from_json(json_str):
        # Convert the JSON string back to a dictionary
        data = json.loads(json_str)
        # Convert datetime fields back to datetime objects
        if "time_optimized" in data and data["time_optimized"]:
            data["time_optimized"] = datetime.fromisoformat(data["time_optimized"])
        if "time_trained" in data and data["time_trained"]:
            data["time_trained"] = datetime.fromisoformat(data["time_trained"])
        return OptionsModel(**data)


class ModelStage(Enum):
    STAGING = "staging"
    PROD = "prod"
    ARCHIVED = "archived"


####################


def monitor_model(
    df: pd.DataFrame,
    evaluation_date: datetime,
    evaluation_period: int = 30,
) -> pd.DataFrame:
    """
    Monitor model performance and detect potential drift using sklearn and statistical tests.

    Args:
        df (pd.DataFrame): DataFrame with predictions already made.
        evaluation_date (datetime): The date for which to generate statistics.
        evaluation_period (int): Number of days to use as reference period.

    Returns:
        pd.DataFrame: DataFrame with monitoring metrics added.
    """
    # Convert evaluation_date to pd.Timestamp
    evaluation_date = pd.Timestamp(evaluation_date)
    start_date = evaluation_date - timedelta(days=evaluation_period)

    reference_data = df.loc[:start_date].tail(evaluation_period)
    current_data = df.loc[evaluation_date:evaluation_date]

    print(f"Reference Data: \n{reference_data}")
    print(f"Current Data: \n{current_data}")

    # Ensure there is data to evaluate
    if len(reference_data) < evaluation_period or current_data.empty:
        monitoring_metrics = {
            DataSchema.monitor_rmse: np.nan,
            DataSchema.monitor_mape: np.nan,
            DataSchema.monitor_ks_statistic: np.nan,
            DataSchema.monitor_wasserstein_distance: np.nan,
            DataSchema.monitor_data_drift: np.nan,
            DataSchema.monitor_target_drift: np.nan,
        }
    else:
        # Extract true values and predictions
        y_true_ref = reference_data[DataSchema.data_delta]
        y_pred_ref = reference_data[DataSchema.pred_delta]
        y_true_cur = current_data[DataSchema.data_delta]
        y_pred_cur = current_data[DataSchema.pred_delta]

        # Check for NaN values
        if (
            y_true_ref.isna().any()
            or y_pred_ref.isna().any()
            or y_true_cur.isna().any()
            or y_pred_cur.isna().any()
        ):
            monitoring_metrics = {
                DataSchema.monitor_rmse: np.nan,
                DataSchema.monitor_mape: np.nan,
                DataSchema.monitor_ks_statistic: np.nan,
                DataSchema.monitor_wasserstein_distance: np.nan,
                DataSchema.monitor_data_drift: np.nan,
                DataSchema.monitor_target_drift: np.nan,
            }
        else:
            # Calculate RMSE
            rmse = mean_squared_error(y_true_cur, y_pred_cur, squared=False)
            mape = mean_absolute_percentage_error(y_true_cur, y_pred_cur)
            ks_statistic, _ = ks_2samp(y_true_ref, y_true_cur)
            wasserstein_dist = wasserstein_distance(y_true_ref, y_true_cur)
            data_drift_detected = ks_statistic > 0.25  # Adjust as needed
            target_drift_detected = (
                ks_2samp(y_pred_ref, y_pred_cur)[0] > 0.2
            )  # Adjust as needed

            monitoring_metrics = {
                DataSchema.monitor_rmse: rmse,
                DataSchema.monitor_mape: mape,
                DataSchema.monitor_ks_statistic: ks_statistic,
                DataSchema.monitor_wasserstein_distance: wasserstein_dist,
                DataSchema.monitor_data_drift: data_drift_detected,
                DataSchema.monitor_target_drift: target_drift_detected,
            }

    # Add data to df row
    for metric_name, metric_value in monitoring_metrics.items():
        df.loc[evaluation_date, metric_name] = metric_value

    return df


def mlflow_promote_model(model_name: str):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    # Find the latest version tagged as 'staging'
    staging_versions = [
        v for v in versions if v.tags.get("status") == ModelStage.STAGING.value
    ]
    if not staging_versions:
        raise ValueError(
            f"No model version found with status 'staging' for model '{model_name}'"
        )
    latest_staging_version = max(staging_versions, key=lambda v: int(v.version))

    # Find all versions tagged as 'prod'
    prod_versions = [
        v for v in versions if v.tags.get("status") == ModelStage.PROD.value
    ]

    # Archive all existing prod models
    for prod_version in prod_versions:
        client.set_model_version_tag(
            name=model_name,
            version=prod_version.version,
            key="status",
            value=ModelStage.ARCHIVED.value,
        )
        print(f"Model version {prod_version.version} archived.")

    # Promote the latest staging model to prod
    client.set_model_version_tag(
        name=model_name,
        version=latest_staging_version.version,
        key="status",
        value=ModelStage.PROD.value,
    )

    print(f"Model version {latest_staging_version.version} promoted to 'prod'.")


def mlflow_get_model(model_name: str):
    client = MlflowClient()
    # Get all versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    # Filter versions with the tag 'status' set to 'prod'
    prod_versions = [v for v in versions if v.tags.get("status") == "prod"]
    if not prod_versions:
        raise ValueError(
            f"No model version found with status 'prod' for model '{model_name}'"
        )
    # Return the latest version tagged as 'prod'
    latest_prod_version = max(prod_versions, key=lambda v: int(v.version))
    return latest_prod_version


def mlflow_register_model(
    options_model: OptionsModel,
    model_name: str,
):
    with mlflow.start_run() as run:
        # Log the model
        mlflow.sklearn.log_model(options_model.model, "model")

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Set the tag for the model version
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            name=model_name,
            version=result.version,
            key="status",
            value=ModelStage.STAGING.value,
        )

        # Serialize the dataclass to JSON and set it as a tag
        options_model_json = options_model.to_json()
        client.set_model_version_tag(
            name=model_name,
            version=result.version,
            key="params",
            value=options_model_json,
        )

        return result.version


# Function to run predictions on each row of the DataFrame
def run_predictions_on_df(df: pd.DataFrame, model: OptionsModel) -> pd.DataFrame:
    for date in df.index:
        df = predict_delta(df, model, date, return_as_df=True)
    return df


def predict_delta(
    df: pd.DataFrame,
    options_model: OptionsModel,
    date_to_predict: datetime,
    return_as_df=True,
) -> Union[float, pd.DataFrame]:
    """
        Predict t
    he delta for a given date using the trained model.
    """
    # Find the closest date in the index
    date_to_predict = date_to_predict.date()
    closest_date = min(df.index, key=lambda x: abs(x.date() - date_to_predict))

    # Prepare the features for prediction
    X = df.loc[closest_date:closest_date, options_model.selected_features]

    # Check for NaN, infinity, or invalid values
    if X.isnull().values.any() or np.isinf(X.values).any():
        print(
            f"Invalid data for prediction on date {closest_date}. Leaving prediction as NaN."
        )
        predicted_delta = np.nan
    else:
        # Make the prediction
        predicted_delta = options_model.model.predict(X)[0]

    # Return
    if return_as_df:
        df.loc[closest_date, DataSchema.pred_delta] = predicted_delta
        return df
    else:
        return predicted_delta


def mlflow_log_df(df: pd.DataFrame, artifact_name: str) -> None:
    """
    # FIXME- this needs fixing. unable to write directly, requires a remote solution like mino
    Save a DataFrame as a CSV file in /tmp/ and log it as an MLflow artifact.

    Args:
        df (pd.DataFrame): The DataFrame to be saved and logged.
        artifact_name (str): The name to be used for the artifact.

    Returns:
        None
    """
    # Generate the CSV filename with timestamp
    csv_filename = f"{datetime.now().strftime('%Y%m%d')}_{artifact_name}.csv"
    csv_path = os.path.join("/tmp", csv_filename)

    # Save the CSV file in the /tmp directory
    df.to_csv(csv_path, index=False)

    # Log the dataset as an artifact
    mlflow.log_artifact(csv_path, artifact_path=csv_filename)


def mlflow_optimize_model(df, cutoff_date: datetime, n_trials=100):
    """
    Optimize a linear regression model for stock price prediction using Optuna and MLflow.

    This function uses Optuna to perform hyperparameter optimization for a linear regression model
    that predicts stock price movements. It optimizes the number of lagged days and the number of
    features to use. The optimization process is logged using MLflow.

    Args:
        df (pd.DataFrame): The input dataframe containing stock price data and features.
        cutoff_date (datetime): The cutoff date for the data to be used in the optimization.
        n_trials (int, optional): The number of optimization trials to perform. Defaults to 100.

    Returns:
        dict: A dictionary containing the best parameters found during optimization, including:
            - n_lagged_days: The optimal number of lagged days to use as features.
            - n_features: The optimal number of features to select.
            - selected_features: A list of the selected feature names.
            - best_directional_accuracy: The best directional accuracy achieved.

    The function logs the following metrics and parameters to MLflow:
        - mean_rmse: The mean root mean squared error across cross-validation folds.
        - mean_directional_accuracy: The mean directional accuracy across cross-validation folds.
        - n_lagged_days: The number of lagged days used.
        - n_features: The number of features selected.
        - selected_features: The names of the selected features.
    """

    def objective(trial):
        """
        Objective function for hyperparameter optimization using Optuna.
        """
        with mlflow.start_run(nested=True):
            # Suggest hyperparameters
            n_lagged_days = trial.suggest_int("n_lagged_days", 2, 5)
            n_features = trial.suggest_int(
                "n_features", 1, len(COLS.get_independent_vars())
            )

            # Prepare data
            X, y = _prepare_data(df, n_lagged_days=n_lagged_days, end_date=cutoff_date)

            # Separate lagged features from other features
            lagged_features = [
                f"data_delta_lag_{i}" for i in range(1, n_lagged_days + 1)
            ]
            non_lagged_features = [
                col for col in X.columns if col not in lagged_features
            ]

            # Select features
            feature_selector = RFE(
                estimator=LinearRegression(), n_features_to_select=n_features
            )
            X_selected = feature_selector.fit_transform(X[non_lagged_features], y)
            selected_features = (
                X[non_lagged_features].columns[feature_selector.support_].tolist()
            )

            # Perform time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            model = LinearRegression()
            rmse_scores = []
            directional_accuracy_scores = []

            for train_index, val_index in tscv.split(X_selected):
                X_train, X_val = X_selected[train_index], X_selected[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)

                # Calculate directional accuracy
                directional_accuracy = np.mean(np.sign(y_val) == np.sign(y_pred))
                directional_accuracy_scores.append(directional_accuracy)

            mean_rmse = np.mean(rmse_scores)
            mean_directional_accuracy = np.mean(directional_accuracy_scores)

            # Normalize RMSE and Directional Accuracy
            normalized_rmse = mean_rmse / (mean_rmse + 1)  # Normalized to [0, 1]
            normalized_directional_accuracy = (
                mean_directional_accuracy  # Already in [0, 1]
            )

            # Inverse Directional Accuracy for Minimization
            inverse_directional_accuracy = 1 - normalized_directional_accuracy

            # Combine with Weights
            rmse_weight = 0.4
            directional_weight = 0.6
            blended_score = (
                normalized_rmse * rmse_weight
                + inverse_directional_accuracy * directional_weight
            )
            # Log metrics and parameters
            mlflow.log_params(
                {
                    "n_lagged_days": n_lagged_days,
                    "n_features": n_features,
                    "selected_features": selected_features,
                }
            )
            mlflow.log_metric("mean_rmse", mean_rmse)
            mlflow.log_metric("mean_directional_accuracy", mean_directional_accuracy)
            mlflow.log_metric("blended_score", blended_score)

            # Store selected_features and metrics in trial user_attrs
            trial.set_user_attr("selected_features", selected_features)
            trial.set_user_attr("selected_lagged_days", n_lagged_days)
            trial.set_user_attr("mean_rmse", mean_rmse)
            trial.set_user_attr("mean_directional_accuracy", mean_directional_accuracy)

            # Return a combination of RMSE and directional accuracy as the objective
            return blended_score

    # Start a parent run to group all experiments
    parent_run_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        # Log Dataset
        mlflow_log_df(df, "dataset")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        # Log the best model parameters and metrics in the parent run
        best_params = study.best_params
        best_params["selected_features"] = study.best_trial.user_attrs[
            "selected_features"
        ]
        best_params["selected_lagged_days"] = study.best_trial.user_attrs[
            "selected_lagged_days"
        ]
        best_params["best_blended_score"] = study.best_value
        best_params["best_rmse"] = study.best_trial.user_attrs["mean_rmse"]
        best_params["best_directional_accuracy"] = study.best_trial.user_attrs[
            "mean_directional_accuracy"
        ]
        mlflow.log_params(best_params)
        mlflow.log_metric("best_blended_score", best_params["best_blended_score"])
        mlflow.log_metric("best_rmse", best_params["best_rmse"])
        mlflow.log_metric(
            "best_directional_accuracy", best_params["best_directional_accuracy"]
        )

        # Train the best model
        options_model = OptionsModel(
            selected_features=best_params["selected_features"],
            n_lagged_days=best_params["selected_lagged_days"],
            time_optimized=datetime.now(),
        )
        best_model = mlflow_train_model(
            df, options_model, end_date=datetime.today(), training_window=5
        )

        # Create OptionsModel
        options_model.model = best_model
        options_model.time_trained = datetime.now()

        return options_model


def mlflow_train_model(
    df, options_model: OptionsModel, *, end_date: datetime, training_window: int
):

    # Helper Function

    def _train_and_log_model(X, y, selected_features):
        X_selected = X[selected_features]
        model = LinearRegression()
        model.fit(X_selected, y)

        # Log model parameters
        mlflow.log_params(
            {
                "model_type": "LinearRegression",
                "n_features": len(selected_features),
                "selected_features": selected_features,
            }
        )

        # # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Log feature importances
        feature_importance = dict(zip(selected_features, model.coef_))
        mlflow.log_params({"importance_" + k: v for k, v in feature_importance.items()})

        # Log metrics
        train_rmse = np.sqrt(mean_squared_error(y, model.predict(X_selected)))
        mlflow.log_metric("train_rmse", train_rmse)

        return model

    # Prepare for training
    X, y = _prepare_data(
        df=df,
        n_lagged_days=options_model.n_lagged_days,
        end_date=end_date,
        training_period=training_window,
    )
    selected_features = options_model.selected_features

    # Check if an MLflow run is already active
    active_run = mlflow.active_run()
    if active_run:
        # Start a nested run if an active run exists
        with mlflow.start_run(nested=True):
            return _train_and_log_model(X, y, selected_features)
    else:
        # Start a new run if no active run exists
        with mlflow.start_run():
            return _train_and_log_model(X, y, selected_features)


def mlflow_get_prod_model(name: str) -> OptionsModel:
    """
    Find the highest version with tag prod, retreive it, unserialize the metadata and return an OptionsModel
    """
    client = MlflowClient()

    # Get all versions of the model
    versions = client.search_model_versions(f"name='{name}'")

    # Filter versions with the tag 'status' set to 'prod'
    prod_versions = [
        v for v in versions if v.tags.get("status") == ModelStage.PROD.value
    ]

    if not prod_versions:
        raise ValueError(
            f"No model version found with status 'prod' for model '{name}'"
        )

    # Get the latest prod version
    latest_prod_version = max(prod_versions, key=lambda v: int(v.version))

    # Retrieve the model metadata
    model_metadata = client.get_model_version(name, latest_prod_version.version)

    # Unserialize the metadata
    options_model_json = model_metadata.tags.get("params")
    if not options_model_json:
        raise ValueError(
            f"No params found for model '{name}' version {latest_prod_version.version}"
        )

    # Convert JSON to OptionsModel
    options_model = OptionsModel.from_json(options_model_json)

    return options_model


# Utility Functions


def _prepare_data(df, n_lagged_days=5, end_date=datetime.now(), training_period=30):
    """
    Prepare data for model training.
    :param df: DataFrame containing features and target
    :param n_lagged_days: Number of lagged days to use
    :param end_date: The end date for the data selection
    :param training_period: Number of rows to use for training
    :return: X, y
    """
    df = df.copy()
    target_col = DataSchema.get_dependant_var()
    indep_col = DataSchema.get_independent_vars()
    lag_cols = [f"{target_col}_lag_{i}" for i in range(1, n_lagged_days + 1)]
    feature_cols = lag_cols + indep_col

    # Find the index of the end_date
    end_idx = df.index.get_indexer([end_date], method="nearest")[0]

    # Calculate the start index
    start_idx = max(0, end_idx - training_period + 1)

    X = df[feature_cols].iloc[start_idx : end_idx + 1]
    y = df[target_col].iloc[start_idx : end_idx + 1]

    return X, y


def setup_mlflow():
    # Set environment variables for MLflow
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:8081"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MLFLOW_ARTIFACT_ROOT"] = "s3://mlflow"
    os.environ["POSTGRES_USER"] = "mlflow"
    os.environ["POSTGRES_PASSWORD"] = "mlflowpassword"
    os.environ["POSTGRES_DB"] = "mlflow"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Create an experiment with the specified artifact location
    # NOTE - this needs to be configured, it not it will default to a local artifact store. https://mlflow.org/docs/latest/tracking/artifacts-stores.html
    experiment_name = "default_experiment"
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")
    artifact_location = f"{artifact_root}/{experiment_name}"

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location
        )

    # Set the experiment for subsequent runs
    mlflow.set_experiment(experiment_name)


####################

if __name__ == "__main__":

    ALPHA_VANTAGE_API_KEY = "7H8XHMRSGISFBKK7"
    STOCK_SYMBOL = "AAPL"
    N_TRADING_DAYS = 30
    SIMULATION_DAYS = 10
    COLS = DataSchema()

    setup_mlflow()
    df = load_df(STOCK_SYMBOL, datetime.now(), N_TRADING_DAYS, ALPHA_VANTAGE_API_KEY)
    print(df)

    X, y = _prepare_data(df)
    print("#### PREPARE DATA")
    print(f"X: {X.shape}, y: {y.shape}")
    print(X.head())
    print(y.head())

    print("\n#### OPTIMIZE PARAMS DATA")
    model_obj: OptionsModel = mlflow_optimize_model(df, datetime.now(), n_trials=10)
    print(f"Optinmized Model: {model_obj}")

    print("\n#### PRED CLOSE")
    df = predict_delta(df, model_obj, datetime.now() - timedelta(days=1))
    print(f"Prediction: {df[COLS.pred_delta]}")

    print("\n#### PRED CLOSE, all")
    df = run_predictions_on_df(df, model_obj)
    print(f"Prediction: {df[COLS.pred_delta]}")

    # Usage example:
    print("\n#### MONITOR")
    # df_monitor = monitor_model(
    #     df,
    #     evaluation_date=datetime.now() - timedelta(days=1),
    #     evaluation_period=5,
    # )
    for index, row in df.iterrows():
        df_monitor = monitor_model(
            df,
            evaluation_date=index,
            evaluation_period=5,
        )
    print(df_monitor[DataSchema.get_monitor_vars()])

    # Register the model
    print("\n#### REGISTER AND PROMOTE MODEL")
    MODELNAME = "options_model"
    version = mlflow_register_model(model_obj, MODELNAME)
    print(f"Registered model version: {version}")
    mlflow_promote_model(MODELNAME)

    print("\n#### GET PROD MODEL")
    model = mlflow_get_prod_model(MODELNAME)
    print(f"Prod model: {model}")
