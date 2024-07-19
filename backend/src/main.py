# from config import settings
import logging

from data import DataSchema, load_df
from fastapi import FastAPI
from model import *
from monitor import backtest_model_performance, monitor_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# from prefect import Client

app = FastAPI()

# mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
# prefect_client = Client(api_server=settings.PREFECT_API_URL)


@app.get("/")
def read_root():
    return {"message": "Welcome to Aleph API"}


@app.get("/data")
def run_model_optimization(df, target_col="close", n_trials=50):
    """
    Runs model optimization to find the best hyperparameters for the model.

    Args:
        df (pandas.DataFrame): The input data frame containing the stock data.
        target_col (str, optional): The name of the target column to predict. Defaults to "close".
        n_trials (int, optional): The number of optimization trials to run. Defaults to 50.

    Returns:
        dict: A dictionary containing the best hyperparameters found during the optimization.
    """
    logging.info("Starting model optimization")
    X, y = _prepare_data(df, target_col)
    best_params = optimize_model_params(X, y, n_trials)
    logging.info(f"Model optimization completed. Best parameters: {best_params}")
    return best_params


def run_model_training(df, best_params, target_col="close"):
    """
    Trains a machine learning model using the provided data and best hyperparameters.

    Args:
        df (pandas.DataFrame): The input data frame containing the stock data.
        best_params (dict): A dictionary containing the best hyperparameters found during the optimization.
        target_col (str, optional): The name of the target column to predict. Defaults to "close".

    Returns:
        A trained machine learning model.
    """
    logging.info("Starting model retraining")
    X, y = _prepare_data(df, target_col, n_lagged_days=best_params["n_lagged_days"])
    model = train_model(X, y, best_params["selected_features"])

    # Quick validation
    X_selected = X[best_params["selected_features"]]
    y_pred = model.predict(X_selected)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    logging.info(f"Model retraining completed. Validation RMSE: {rmse:.4f}")
    return model


@app.get("/prediction")
def run_prediction(model, df, prediction_date, best_params):
    """
    Runs a prediction using the trained machine learning model and the provided data.

    Args:
        model (sklearn.base.BaseEstimator): The trained machine learning model.
        df (pandas.DataFrame): The input data frame containing the stock data.
        prediction_date (pandas.Timestamp): The date for which to make the prediction.
        best_params (dict): A dictionary containing the best hyperparameters found during the optimization.

    Returns:
        tuple:
            - predicted_close (float): The predicted closing price for the given date.
            - actual_close (float): The actual closing price for the given date, if available.
            - prediction_error (float): The absolute difference between the predicted and actual closing price, if available.
    """
    logging.info(f"Predicting closing price for {prediction_date}")

    if prediction_date not in df.index:
        logging.error(f"Prediction date {prediction_date} not found in the data")
        return None, None, None

    X = df.loc[prediction_date, best_params["selected_features"]].to_frame().T
    predicted_close = model.predict(X)[0]

    actual_close = (
        df.loc[prediction_date, "close"] if prediction_date in df.index else None
    )
    abs_error = (
        abs(predicted_close - actual_close) if actual_close is not None else None
    )

    logging.info(f"Prediction completed. Predicted close: {predicted_close:.2f}")
    if actual_close is not None:
        logging.info(
            f"Actual close: {actual_close:.2f}, Prediction error: {abs_error:.2f}"
        )

    return predicted_close, actual_close, abs_error


def run_prediction_as_df(model, df, best_params, prediction_date=datetime.now()):
    logging.info(f"Predicting closing price for {prediction_date}")

    if prediction_date not in df.index:
        logging.error(f"Prediction date {prediction_date} not found in the data")
        return None

    X = df.loc[prediction_date, best_params["selected_features"]].to_frame().T
    predicted_close = model.predict(X)[0]

    actual_close = df.loc[prediction_date, "close"]
    predicted_delta = predicted_close - df.loc[prediction_date, "open"]
    actual_delta = actual_close - df.loc[prediction_date, "open"]

    prediction_df = pd.DataFrame(
        {
            "prediction": [predicted_close],
            "target": [actual_delta],
            "open": [df.loc[prediction_date, "open"]],
            "close": [actual_close],
            "predicted_delta": [predicted_delta],
            "actual_delta": [actual_delta],
        },
        index=[prediction_date],
    )

    logging.info(f"Prediction completed. Predicted delta: {predicted_delta:.2f}")
    logging.info(f"Actual delta: {actual_delta:.2f}")

    return prediction_df


def simulate_trading(data, model, best_params, SIMULATION_DAYS):
    """
    Given a dataframe, runs backtested simluation N days into the past from the last date in the dataframe.

    """
    predictions = []
    actuals = []
    dates = []

    for i in range(len(data) - SIMULATION_DAYS, len(data)):
        current_date = data.index[i]
        predicted_close, actual_close, prediction_error = run_prediction(
            model, data, current_date, best_params
        )

        predictions.append(predicted_close)
        actuals.append(actual_close)
        dates.append(current_date)

    predictions_df = pd.DataFrame(
        {
            "date": dates,
            "prediction": predictions,
            "actual": actuals,
            "open": data.iloc[-SIMULATION_DAYS:]["open"].values,
            "close": data.iloc[-SIMULATION_DAYS:]["close"].values,
        }
    )
    predictions_df.set_index("date", inplace=True)

    predictions_df["actual_delta"] = predictions_df["close"] - predictions_df["open"]
    predictions_df["predicted_delta"] = (
        predictions_df["prediction"] - predictions_df["open"]
    )

    return predictions_df


###################

if __name__ == "__main__":

    ALPHA_VANTAGE_API_KEY = "7H8XHMRSGISFBKK7"
    STOCK_SYMBOL = "AAPL"
    N_TRADING_DAYS = 30
    SIMULATION_DAYS = 10
    COLS = DataSchema()

    df = load_df(STOCK_SYMBOL, datetime.now(), N_TRADING_DAYS, ALPHA_VANTAGE_API_KEY)
    print(df)

    # Optimize model
    best_params = run_model_optimization(df)

    # Train model on fresh data
    model_obj = run_model_training(df, best_params)

    predictions_df = simulate_trading(df, model_obj, best_params, SIMULATION_DAYS)

    # Run backtesting
    backtest_results = backtest_model_performance(predictions_df, window_size=10)

    print(f"Backtesting completed for {len(backtest_results)} periods")
    for i, result in enumerate(backtest_results):
        print(
            f"Period {i+1}: MSE = {result['summary']['mse']:.4f}, "
            f"MAE = {result['summary']['mae']:.4f}, "
            f"Accuracy = {result['summary']['accuracy']:.4f}"
        )

    # # Predict for the most recent date
    # last_date = df_lagged.index[-1]
    # predicted_close, actual_close, prediction_error = run_prediction(
    #     model, df_lagged, last_date, best_params
    # )

    # if predicted_close is not None:
    #     print(f"Prediction for {last_date.date()}:")
    #     print(f"Predicted closing price: {predicted_close:.2f}")
    #     if actual_close is not None:
    #         print(f"Actual closing price: {actual_close:.2f}")
    #         print(f"Prediction error: {prediction_error:.2f}")

    # # After simulating trading, perform backtesting
    # backtest_results = backtest_model_performance(predictions_df, window_size=10)

    # print(f"Backtesting completed for {len(backtest_results)} periods")
    # for i, result in enumerate(backtest_results):
    #     print(
    #         f"Period {i+1}: MSE = {result['summary']['mse']:.4f}, "
    #         f"MAE = {result['summary']['mae']:.4f}, "
    #         f"Accuracy = {result['summary']['accuracy']:.4f}"
    #     )
