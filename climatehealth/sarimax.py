import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.seasonal import seasonal_decompose
# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset


def analyze_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Plot the time series data
    df['DengueCases'].plot(figsize=(12, 6), title='Monthly Dengue Cases')
    plt.show()
    # Decompose the time series to analyze trend, seasonality, and residuals
    result = seasonal_decompose(df['DengueCases'], model='additive')
    result.plot()
    plt.show()
    # Check autocorrelation and partial autocorrelation to determine SARIMA parameters
    plot_acf(df['DengueCases'], lags=36)
    plt.show()
    plot_pacf(df['DengueCases'], lags=36)
    plt.show()
    # Choose SARIMA parameters based on the ACF and PACF plots
    # Replace the values below with your chosen parameters
    order = (1, 1, 1)  # (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # (P, D, Q, S)

    # Exogenous variables
    exog_variables = df[['Rainfall', 'Temperature']]
    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    exog_train, exog_test = exog_variables.iloc[:train_size], exog_variables.iloc[train_size:]
    # Fit the SARIMAX model
    results = train_model(train, exog_train, order, seasonal_order)
    # Predict on the test set
    rmse_val = evaluate_model(exog_test, results, test, train)

    # Forecast future values
    # forecast_steps = 12  # Change this value based on how many months ahead you want to forecast
    # exog_forecast = exog_variables.iloc[-1:].append(
    #     pd.DataFrame(index=pd.date_range(start=df.index[-1] + pd.DateOffse0t(1), periods=forecast_steps)))
    # forecast = results.get_forecast(steps=forecast_steps, exog=exog_forecast)
    # forecast_mean = forecast.predicted_mean
    # # Plot the forecasted values
    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index, df['DengueCases'], label='Historical Data')
    # plt.plot(forecast_mean.index, forecast_mean, label='SARIMAX Forecast', color='red')
    # plt.title(f'SARIMAX Forecast for {forecast_steps} Months with Exogenous Variables')
    # plt.legend()
    # plt.show()
    return rmse_val


def evaluate_model(exog_test, results, test, train):
    predictions = results.get_forecast(steps=len(test), exog=exog_test)
    predicted_mean = predictions.predicted_mean
    # Plot the actual vs. predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['DengueCases'], label='Train')
    plt.plot(test.index, test['DengueCases'], label='Test')
    plt.plot(test.index, predicted_mean, label='SARIMAX Forecast', color='red')
    plt.title('SARIMAX Forecast for Monthly Dengue Cases with Exogenous Variables')
    plt.legend()
    plt.show()
    # Evaluate the model performance
    mse = ((predicted_mean - test['DengueCases']) ** 2).mean()
    rmse_val = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): {rmse_val}')
    return rmse_val


def train_model(train, exog_train, order, seasonal_order):
    model = SARIMAX(train['DengueCases'], exog=exog_train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False)
    results = model.fit()
    return results

# analyze_data()