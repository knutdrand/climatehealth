import ee
import pytest

from climatehealth.ee_wrapper import EEWrapper

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


@pytest.fixture
def ee_dataset():
    ic = ee.ImageCollection(
        'ECMWF/ERA5_LAND/MONTHLY_AGGR')
    return ic
    # return EEWrapper(ic)


def test_compute():
    ee_dataset = EEWrapper(ic)

    assert False
