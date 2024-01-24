import os
import pickle

import ee
import pandas as pd

from .laos_example import get_dataset, get_city_name, get_point_dict
from ..ee_wrapper import get_image_collection, EEWrapper
from ..health_and_climate_dataset import extract_same_range_from_climate_data


def main(filename='example_data/10yeardengudata.csv'):
    data = get_dataset(filename)
    print(data)
    if os.path.exists('point_dict.pickle'):
        with open('point_dict.pickle', 'rb') as f:
            point_dict = pickle.load(f)
    else:
        point_dict = get_point_dict(data)
        with open('point_dict.pickle', 'wb') as f:
            pickle.dump(point_dict, f)
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    ic = get_image_collection(period='DAILY')
    ee_dataset = EEWrapper(ic)
    range_data = extract_same_range_from_climate_data(data, ee_dataset)
    d = range_data.total_precipitation_sum
    d2 = range_data.temperature_2m

    for name, point in list(point_dict.items())[1:2]:
        if point is None:
            continue
        ee_point = ee.Geometry.Point(point.longitude, point.latitude)
        values = d[ee_point].compute()
        temperature = d2[ee_point].compute()
        new_data_frame = pd.DataFrame({'Date': data['periodname'],
                                       'Rainfall': values,
                                       'DengueCases': data[name],
                                       'Temperature': temperature})

        new_data_frame.to_csv(f'{name}_daily.csv')
