import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.seasonal import seasonal_decompose

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
df = pd.read_csv('your_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


def get_order(dengue_cases):
    global dengue_cases, order, seasonal_order
    # Plot the time series data

    dengue_cases.plot(figsize=(12, 6), title='Monthly Dengue Cases')
    plt.show()
    # Decompose the time series to analyze trend, seasonality, and residuals
    result = seasonal_decompose(dengue_cases, model='additive')
    result.plot()
    plt.show()
    # Check autocorrelation and partial autocorrelation to determine SARIMA parameters
    plot_acf(dengue_cases, lags=36)
    plt.show()
    plot_pacf(dengue_cases, lags=36)
    plt.show()
    # Choose SARIMA parameters based on the ACF and PACF plots
    # Replace the values below with your chosen parameters
order = (1, 1, 1)  # (p, d, q)
seasonal_order = (1, 1, 1, 12)  # (P, D, Q, S)


order, seasonal_order = get_order(df['DengueCases'])

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Fit the SARIMA model
model = SARIMAX(train['DengueCases'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
results = model.fit()

# Predict on the test set
predictions = results.get_forecast(steps=len(test))
predicted_mean = predictions.predicted_mean

# Plot the actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['DengueCases'], label='Train')
plt.plot(test.index, test['DengueCases'], label='Test')
plt.plot(test.index, predicted_mean, label='SARIMA Forecast', color='red')
plt.title('SARIMA Forecast for Monthly Dengue Cases')
plt.legend()
plt.show()

# Evaluate the model performance
mse = ((predicted_mean - test['DengueCases']) ** 2).mean()
rmse_val = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse_val}')

# Forecast future values
forecast_steps = 12  # Change this value based on how many months ahead you want to forecast
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# Plot the forecasted values
plt.figure(figsize=(12, 6))
plt.plot(df.index, dengue_cases, label='Historical Data')
plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color='red')
plt.title(f'SARIMA Forecast for {forecast_steps} Months')
plt.legend()
plt.show()