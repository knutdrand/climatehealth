import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dates():
    return pd.date_range('20130101', periods=12*7, freq='M')


@pytest.fixture
def dengue_cases(dates):
    return np.arange(len(dates))


@pytest.fixture
def rain_fall(dates):
    return np.arange(len(dates)) * 100


@pytest.fixture
def temperature(dates):
    return np.arange(len(dates)) * 10


@pytest.fixture
def df(dates, dengue_cases, rain_fall, temperature):
    return pd.DataFrame({'Date': dates,
                         'DengueCases': dengue_cases,
                         'Rainfall': rain_fall,
                         'Temperature': temperature})
