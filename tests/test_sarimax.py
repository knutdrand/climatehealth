from climatehealth.sarimax import analyze_data
from climatehealth.pymc_model import new_pymc_sarima
from .datasets import *

@pytest.mark.parametrize('model_func', [new_pymc_sarima])#, analyze_data])
def test_analyze_data(df, model_func):
    print(f'------------{model_func(df)}')

