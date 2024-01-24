import os
import pickle
from datetime import date
import ee
import pandas as pd
import numpy as np
from ..geo_coding import get_location
from ..ee_wrapper import EEWrapper, get_image_collection
from ..health_and_climate_dataset import extract_same_range_from_climate_data
from ..modelling.dengue_sir_model import analyze_data

months = ['January', 'February', 'March', 'April', 'May',
          'June', 'July', 'August', 'September', 'October', 'November',
          'December']

lookup = dict(zip(months, range(12)))


def get_city_name(location):
    return location.split(maxsplit=6)[-1]


def get_date(month, year):
    return date(year, lookup[month] + 1, 1)


def get_dataset(filename):
    filepath_or_buffer = '../example_data/10yeardengudata.csv'
    data = pd.read_csv(filename, sep='\t', header=1)
    years = np.array([int(s.strip().split()[-1]) for s in data['periodname']])
    data['periodname'] = [get_date(s.strip().split()[0], int(s.strip().split()[-1])) for s in data['periodname']]
    data = data.sort_values(by='periodname')
    data = data.iloc[:-2]  # remove november, december 2023
    for columnname in data.columns[1:]:
        column = data[columnname]
        data[columnname] = column.replace(np.nan, 0)

    return data


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

    name = list(point_dict)[1]
    if False and os.path.exists(f'{name}.csv'):
        new_data_frame = pd.read_csv(f'{name}.csv')
        analyze_data(new_data_frame, exog_names=['Rainfall'])
        return

    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    ic = get_image_collection()
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

        new_data_frame.to_csv(f'{name}.csv')
        analyze_data(new_data_frame, exog_names=['Rainfall'])


def get_point_dict(data):
    city_names = [get_city_name(c) for c in data.columns[1:]]
    locations = [get_location(name) for name in city_names]
    point_dict = {name: location for name, location in zip(data.columns, locations)}
    return point_dict


