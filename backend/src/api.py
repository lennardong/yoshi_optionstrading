from datetime import datetime

from data import DataSchema, load_df
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from main import ALPHA_VANTAGE_API_KEY, MODEL_NAME, N_TRADING_DAYS, STOCK_SYMBOL
from model import mlflow_get_prod_model, run_predictions_on_df
from prefect import flow

app = FastAPI()


@flow
def prediction_flow():
    model = mlflow_get_prod_model(MODEL_NAME)
    df = load_df(STOCK_SYMBOL, datetime.now(), N_TRADING_DAYS, ALPHA_VANTAGE_API_KEY)
    df = run_predictions_on_df(df, model)
    return df.iloc[-1][DataSchema.pred_delta]


@app.get("/predict")
async def predict():
    try:
        latest_prediction = prediction_flow()
        return JSONResponse(content={"prediction": float(latest_prediction)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
