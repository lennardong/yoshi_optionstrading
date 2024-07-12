import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def create_lagged_features(df, target_col, max_lag=10):
    """
    Create lagged features from the target column.

    :param df: DataFrame containing the time series data
    :param target_col: Name of the target column
    :param max_lag: Maximum number of lagged days
    :return: DataFrame with lagged features
    """
    lagged_df = df.copy()
    for lag in range(1, max_lag + 1):
        lagged_df[f"{target_col}_lag_{lag}"] = lagged_df[target_col].shift(lag)

    lagged_df = lagged_df.dropna()
    return lagged_df


def prepare_data(df, target_col="close", n_lagged_days=5):
    """
    Prepare data for model training.

    :param df: DataFrame containing features and target
    :param target_col: Name of the target column
    :param n_lagged_days: Number of lagged days to use
    :return: X, y
    """
    lag_cols = [f"{target_col}_lag_{i}" for i in range(1, n_lagged_days + 1)]
    feature_cols = lag_cols + ["RSI", "SMA_10", "SMA_20", "Upper_BB", "Lower_BB"]
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def train_model(X, y, selected_features):
    X_selected = X[selected_features]
    model = LinearRegression()
    model.fit(X_selected, y)
    return model  # We don't need to return RFE object anymore


def evaluate_model(model, X, y):
    """
    Evaluate the model on test data.

    :param model: Trained model
    :param X: Test features
    :param y: Test target
    :return: RMSE, R2 score, and directional accuracy
    """
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # Calculate directional accuracy
    y_true = y.values
    direction_actual = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    directional_accuracy = np.mean(direction_actual == direction_pred)

    return rmse, r2, directional_accuracy


def optimize_model(X, y, n_trials=100):
    """
    Optimize model hyperparameters using Optuna.

    :param X: Feature matrix
    :param y: Target vector
    :param n_trials: Number of optimization trials
    :return: Best parameters and selected features
    """

    def optimize(trial):
        n_lagged_days = trial.suggest_int("n_lagged_days", 2, 5)
        n_features = trial.suggest_int("n_features", 1, X.shape[1])

        # Select columns based on n_lagged_days
        lag_cols = [f"close_lag_{i}" for i in range(1, n_lagged_days + 1)]
        other_cols = [col for col in X.columns if not col.startswith("close_lag_")]
        selected_cols = lag_cols + other_cols

        X_selected = X[selected_cols]

        # Perform feature selection
        rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
        rfe.fit(X_selected, y)

        selected_features = [
            feature
            for feature, selected in zip(X_selected.columns, rfe.support_)
            if selected
        ]

        # Train model with selected features
        X_rfe = rfe.transform(X_selected)
        model = LinearRegression()

        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        rmse_scores = []

        for train_index, val_index in tscv.split(X_rfe):
            X_train, X_val = X_rfe[train_index], X_rfe[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)

        mean_rmse = np.mean(rmse_scores)

        trial.set_user_attr("selected_features", selected_features)
        return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(optimize, n_trials=n_trials)

    best_params = study.best_params
    best_params["selected_features"] = study.best_trial.user_attrs["selected_features"]
    best_params["n_features"] = len(best_params["selected_features"])

    return best_params


def predict_closing_price(model, df, prediction_date, selected_features):
    if prediction_date not in df.index:
        raise ValueError(f"Prediction date {prediction_date} not found in the data")

    prediction_row = df.loc[prediction_date]
    X = prediction_row[selected_features].to_frame().T
    prediction = model.predict(X)
    return prediction[0]


def mlflow_optimize_model(X, y, n_trials=100):
    def objective(trial):
        with mlflow.start_run(nested=True):
            n_lagged_days = trial.suggest_int("n_lagged_days", 2, 5)
            n_features = trial.suggest_int("n_features", 1, X.shape[1])

            lag_cols = [f"close_lag_{i}" for i in range(1, n_lagged_days + 1)]
            other_cols = [col for col in X.columns if not col.startswith("close_lag_")]
            selected_cols = lag_cols + other_cols

            X_selected = X[selected_cols]

            rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
            rfe.fit(X_selected, y)

            selected_features = [
                feature
                for feature, selected in zip(X_selected.columns, rfe.support_)
                if selected
            ]

            X_rfe = rfe.transform(X_selected)
            model = LinearRegression()

            tscv = TimeSeriesSplit(n_splits=5)
            rmse_scores = []

            for train_index, val_index in tscv.split(X_rfe):
                X_train, X_val = X_rfe[train_index], X_rfe[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)

            mean_rmse = np.mean(rmse_scores)

            mlflow.log_params(
                {
                    "n_lagged_days": n_lagged_days,
                    "n_features": n_features,
                    "selected_features": selected_features,
                }
            )
            mlflow.log_metric("mean_rmse", mean_rmse)

            # Store selected_features in trial user_attrs
            trial.set_user_attr("selected_features", selected_features)

            return mean_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["selected_features"] = study.best_trial.user_attrs["selected_features"]
    best_params["best_rmse"] = study.best_value

    return best_params


# def mlflow_optimize_model(X, y, n_trials=100):
#     def objective(trial):
#         with mlflow.start_run(nested=True):
#             n_lagged_days = trial.suggest_int("n_lagged_days", 2, 5)
#             n_features = trial.suggest_int("n_features", 1, X.shape[1])

#             lag_cols = [f"close_lag_{i}" for i in range(1, n_lagged_days + 1)]
#             other_cols = [col for col in X.columns if not col.startswith("close_lag_")]
#             selected_cols = lag_cols + other_cols

#             X_selected = X[selected_cols]

#             rfe = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
#             rfe.fit(X_selected, y)

#             selected_features = [
#                 feature
#                 for feature, selected in zip(X_selected.columns, rfe.support_)
#                 if selected
#             ]

#             X_rfe = rfe.transform(X_selected)
#             model = LinearRegression()

#             tscv = TimeSeriesSplit(n_splits=5)
#             rmse_scores = []

#             for train_index, val_index in tscv.split(X_rfe):
#                 X_train, X_val = X_rfe[train_index], X_rfe[val_index]
#                 y_train, y_val = y.iloc[train_index], y.iloc[val_index]

#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_val)
#                 rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#                 rmse_scores.append(rmse)

#             mean_rmse = np.mean(rmse_scores)

#             mlflow.log_params(
#                 {
#                     "n_lagged_days": n_lagged_days,
#                     "n_features": n_features,
#                     "selected_features": selected_features,
#                 }
#             )
#             mlflow.log_metric("mean_rmse", mean_rmse)

#             return mean_rmse

#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=n_trials)

#     best_params = study.best_params
#     best_params["selected_features"] = study.best_trial.user_attrs["selected_features"]
#     best_params["best_rmse"] = study.best_value

#     return best_params


def mlflow_train_model(X, y, selected_features):
    with mlflow.start_run():
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

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Log feature importances
        feature_importance = dict(zip(selected_features, model.coef_))
        mlflow.log_params({"importance_" + k: v for k, v in feature_importance.items()})

        # Log metrics
        train_rmse = np.sqrt(mean_squared_error(y, model.predict(X_selected)))
        mlflow.log_metric("train_rmse", train_rmse)

        return model


if __name__ == "__main__":
    from datetime import datetime

    import main as main
    import mlflow_utils as mlflow_utils

    mlflow_utils.setup_mlflow()

    ALPHA_VANTAGE_API_KEY = "7H8XHMRSGISFBKK7"
    STOCK_SYMBOL = "AAPL"
    N_TRADING_DAYS = 30

    # Fetch data
    df = main.get_data(
        STOCK_SYMBOL, datetime.now(), N_TRADING_DAYS, ALPHA_VANTAGE_API_KEY
    )

    # Create lagged features
    df_lagged = create_lagged_features(df, "close")

    # Prepare data for optimization
    X, y = prepare_data(df_lagged)

    # Optimize model
    best_params = optimize_model(X, y)
    best_params = mlflow_optimize_model(X, y)
    print(f"Best parameters: {best_params}")

    # Train model with best parameters
    model = train_model(X, y, best_params["selected_features"])

    # Evaluate model
    X_selected = X[best_params["selected_features"]]
    rmse, r2, directional_accuracy = evaluate_model(model, X_selected, y)
    print(
        f"Model Performance - RMSE: {rmse:.4f}, R2: {r2:.4f}, Directional Accuracy: {directional_accuracy:.4f}"
    )

    # Get the most recent date in the dataset
    last_date = df_lagged.index[-1]

    # If the last date is today, use it; otherwise, use the last available date
    today = datetime.now().date()
    prediction_date = last_date if last_date.date() == today else last_date

    # Predict closing price for the prediction date
    predicted_close = predict_closing_price(
        model, df_lagged, prediction_date, best_params["selected_features"]
    )

    print(f"\n######\n# PREDICTION FOR {prediction_date.date()}:")
    print(f"Opening price: {df_lagged.loc[prediction_date, 'open']:.2f}")
    print(f"Predicted closing price: {predicted_close:.2f}")

    # If we're predicting for a past date, we can compare with the actual closing price
    if prediction_date.date() != today:
        actual_close = df_lagged.loc[prediction_date, "close"]
        print(f"Actual closing price: {actual_close:.2f}")
        print(f"Prediction error: {abs(predicted_close - actual_close):.2f}")
