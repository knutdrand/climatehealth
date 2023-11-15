from climatehealth.sarimax import analyze_data
from .datasets import *

@pytest.mark.parametrize('model_func', [analyze_data])
def test_analyze_data(df, model_func):
    print(f'------------{model_func(df)}')

