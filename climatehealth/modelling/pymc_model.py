import pandas as pd
import numpy as np
np.bool = bool
import matplotlib.pyplot as plt
import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
#df = pd.read_csv('your_dataset.csv')

def new_pymc_sarima(df):
    # Load your dataset
    # Replace 'your_dataset.csv' with the actual file path or URL of your dataset
    # df = pd.read_csv('your_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Exogenous variables
    exog_variables = df[['Rainfall', 'Temperature']]

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    exog_train, exog_test = exog_variables.iloc[:train_size], exog_variables.iloc[train_size:]

    # Bayesian SARIMAX model
    with pm.Model() as sarimax_model:
        # Priors
        mu = pm.Normal('mu', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=10)

        # SARIMAX coefficients
        ar = pm.Normal('ar', mu=0, sigma=1, shape=1)  # AR order 1
        ma = pm.Normal('ma', mu=0, sigma=1, shape=1)  # MA order 1

        # Exogenous variable coefficients
        beta_rainfall = pm.Normal('beta_rainfall', mu=0, sigma=1)
        beta_temperature = pm.Normal('beta_temperature', mu=0, sigma=1)

        # Seasonal effect
        seasonal_effect = pm.Normal('seasonal_effect', mu=0, sigma=1, shape=12)

        # Model equation
        y_est = (
                mu +
                ar * pm.AR('ar_vals', 1, ar_order=2, steps=3) +
                ma * pm.AR('ma_vals', 1, ar_order=2, steps=3) +
                beta_rainfall * exog_variables['Rainfall'] +
                beta_temperature * exog_variables['Temperature'] +
                seasonal_effect[df.index.month.to_numpy() - 1]
        )

        # Likelihood
        likelihood = pm.Normal('likelihood', mu=y_est, sigma=sigma, observed=train['DengueCases'])

    # Sampling
    with sarimax_model:
        trace = pm.sample(1000, tune=1000, cores=1)

    # Posterior predictive checks
    ppc = pm.sample_posterior_predictive(trace, samples=1000, model=sarimax_model)

    # Plot the posterior predictive checks
    # az.plot_ppc(az.from_pymc5(posterior_predictive=ppc, model=sarimax_model), mean=False)
    # plt.show()
    return ppc


def pymc_sarima(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Exogenous variables
    exog_variables = df[['Rainfall', 'Temperature']]
    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    exog_train, exog_test = exog_variables.iloc[:train_size], exog_variables.iloc[train_size:]
    # Bayesian SARIMAX model
    with pm.Model() as sarimax_model:
        # Priors
        mu = pm.Normal('mu', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=10)

        # SARIMAX coefficients
        ar = pm.Normal('ar', mu=0, sd=1, shape=1)  # AR order 1
        ma = pm.Normal('ma', mu=0, sd=1, shape=1)  # MA order 1

        # Exogenous variable coefficients
        beta_rainfall = pm.Normal('beta_rainfall', mu=0, sd=1)
        beta_temperature = pm.Normal('beta_temperature', mu=0, sd=1)

        # Seasonal effect
        seasonal_effect = pm.Normal('seasonal_effect', mu=0, sd=1, shape=12)

        # Model equation
        y_est = (
                mu +
                ar * pm.AR('ar_vals', 1) +
                ma * pm.AR('ma_vals', 1) +
                beta_rainfall * exog_variables['Rainfall'] +
                beta_temperature * exog_variables['Temperature'] +
                seasonal_effect[df.index.month - 1]
        )

        # Likelihood
        likelihood = pm.Normal('likelihood', mu=y_est, sd=sigma, observed=train['DengueCases'])
    # Sampling
    with sarimax_model:
        trace = pm.sample(1000, tune=1000, cores=1)
    # Posterior predictive checks
    ppc = pm.sample_posterior_predictive(trace, samples=1000, model=sarimax_model)
    # Plot the posterior predictive checks
    # az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=sarimax_model), mean=False)
    plt.show()
    return trace

# pymc_sarima()