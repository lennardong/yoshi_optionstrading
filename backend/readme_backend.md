
# Model Training and Lifecycle Strategy

## Overview
Use a prediction model to forecast the closing price of AAPL and make informed decisions to purchase call or put options with a 0-day expiration (same-day expiration).

Our strategy for predicting daily stock prices involves daily model retraining with continuous monitoring for model parameter optimization. This approach ensures the model always uses the most recent market data while maintaining a mechanism to trigger full re-optimization when necessary.

## Model
- Timeseries prediction using a timeseries model built with a simple lagged linear regression model.

### Model Variables
The following are the dependant and independant variables used in the model. For simplicity, we assume the stock is AAPL.
- Closing prices for the past 2-5 days (to be optimized)
- Opening price of the current day
- 3-5 technical indicators derived from stock data, `compute_features` which will output a dict of the following: 
    - Relative Strength Index (RSI) with a 14-day period
    - Simple Moving Average (SMA) with a 10-day period
    - Bollinger Bands with a 20-day period and 2 standard deviations

Explanation of the indicators:
1. Relative Strength Index (RSI):
   - RSI is a momentum oscillator that measures the speed and change of price movements.
   - It oscillates between 0 and 100, with values above 70 typically considered overbought and below 30 oversold.
   - In our daily trading context, it can help identify potential short-term reversals or continuations of trends.
   - The 14-day period is a standard setting that balances sensitivity and reliability.

2. Simple Moving Average (SMA):
   - SMA calculates the average price over a specified number of periods.
   - It helps smooth out price data to identify trends.
   - A 10-day SMA provides a short-term trend indicator suitable for daily trading.
   - It can be used to identify potential support/resistance levels and trend directions.

3. Bollinger Bands:
   - Bollinger Bands consist of a middle band (typically a 20-day SMA) and an upper and lower band (typically 2 standard deviations above and below the middle band).
   - They provide information about price volatility and potential overbought/oversold conditions.
   - In the context of daily trading and 0-day options:
     * The width of the bands can indicate potential for significant price moves.
     * Price touching or exceeding the bands can signal potential reversals or continuations, which is crucial for short-term option strategies.
   - The 20-day period and 2 standard deviations are standard settings that work well for short-term analysis.

## Model Experimentation and Optimization

We use Optuna for hyperparameter optimization. The following parameters will be tuned:
1. Number of lagged days (range: 2 to 5)
2. Feature selection using Recursive Feature Elimination (RFE)

### Optimization Process

1. Use a rolling window approach:
   - Start with the most recent 30 trading days
   - For each day in the last 10 trading days:
     a. Train on the previous 10 trading days
     b. Make a prediction for the next day
     c. Calculate the prediction error
   - Use the average RMSE and directional accuracy over these 10 predictions as the optimization targets for Optuna

2. Optuna will suggest different combinations of:
   - Number of lagged days (range: 2 to 5)
   - Features to include (selected by RFE)
   - Model-specific hyperparameters

3. For each Optuna trial:
   - Perform the rolling window validation
   - Return the average RMSE and directional accuracy to Optuna

4. Use Optuna's multi-objective optimization to balance RMSE and directional accuracy

5. After a predefined number of trials or time limit, Optuna will provide the Pareto front of best parameter combinations

6. Select the final model from the Pareto front based on a predefined trade-off between RMSE and directional accuracy

7. Train the final model using the selected parameters on the most recent 10 trading days

### Daily Retraining

After initial optimization:
1. Retrain the model daily using the most recent 10 trading days
2. Use this updated model for the next day's prediction

### Monitoring and Re-optimization

1. Track daily prediction error (RMSE) and directional accuracy
2. If performance degrades (e.g., 3-day moving average of RMSE increases by 20% or directional accuracy drops below 55%), trigger a full re-optimization

## Lifecycle Management 

### Technologies
- Orchestration: Prefect
- Model Experiments: MLFlow
- Model Optimization: Optuna
- Model Building: Scikit-learn, TensorFlow (for LSTM)
- Data Processing: Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn

### Periodic Retraining
Model will undergo periodic retraining every Sunday at 11:00 PM EST to be ready for Monday trading.  Retraining will use the same model experimentation process to get the best model. It will then retrain the model with the best model parameters.

### Monitoring 
Monitor model performance daily and retrain when significant performance degradation is detected. When performance degrades below a 0.6 R-squared score or when the model's profit factor drops below 1.2, it will trigger a retraining cycle.

#### Types of Monitoring
- Performance Monitoring: Continuously monitor the model's performance metrics (e.g., R-squared, RMSE, profit factor, Sharpe ratio) to detect significant drops.
- Statistical Tests: Use Kolmogorov-Smirnov test to detect changes in data distribution between training and prediction data.
- Feature Drift: Monitor changes in the distribution of input features over time.

### Daily Trading Routine
- Pre-market (8:30 AM EST): Fetch the model variables using `fetch_predictors`
- Prediction (9:15 AM EST): Use the trained model to predict the closing price and confidence interval.
- Decision Making (9:25 AM EST):
    - If the predicted closing price is higher than the opening price and the lower bound of the confidence interval is above the opening price, purchase call options with 0-day expiration.
    - If the predicted closing price is lower than the opening price and the upper bound of the confidence interval is below the opening price, purchase put options with 0-day expiration.
    - The size of the position should be proportional to the model's confidence (width of the confidence interval).
- Risk Management:
    - Set a stop-loss at 2% below the option purchase price.
    - Take profit at 5% above the option purchase price or at 3:45 PM EST, whichever comes first.
- End-of-Day Analysis (4:30 PM EST):
    - Compare predicted vs actual closing prices.
    - Calculate daily profit/loss.
    - Update performance metrics.
    - Log all trades and outcomes for further analysis.
