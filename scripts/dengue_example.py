from datetime import date

import ee
import pandas as pd
import numpy as np

from climatehealth.ee_wrapper import EEWrapper
from climatehealth.health_and_climate_dataset import extract_same_range_from_climate_data
# from climatehealth.modelling.sarimax import analyze_data
from climatehealth.modelling.particles_wrapper import analyze_data
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']

lookup = dict(zip(months, range(12)))

def get_city_name(location):
    return location.split(maxsplit=6)[-1]


def get_date(month, year):
    return date(year, lookup[month]+1, 1)


def get_dataset(filename):
    filepath_or_buffer = '../example_data/10yeardengudata.csv'
    data = pd.read_csv(filename, sep='\t', header=1)
    years = np.array([int(s.strip().split()[-1]) for s in data['periodname']])
    data['periodname'] = [get_date(s.strip().split()[0], int(s.strip().split()[-1])) for s in data['periodname']]
    # month = [lookup[s.strip().split()[0]] for s in data['periodname']]
    # data['periodname'] = (years * 12 + month)
    data = data.sort_values(by='periodname')
    data = data.iloc[:-2] # remove november, december 2023
    for columnname in data.columns[1:]:
        column = data[columnname]
        data[columnname] = column.replace(np.nan, 0)

    return data


def main():
    data = get_dataset('../example_data/10yeardengudata.csv')
    print(data)

    # data = pd.melt(data, id_vars='periodname', var_name='location')
    # data['year'] = [str(year) for year in data['periodname'] // 12]
    # data['month'] = data['periodname'] % 12
    # print(data)
    # print(min(data['periodname']), max(data['periodname']))
    # city_names = [get_city_name(c) for c in data.columns[1:]]
    # locations = [get_location(name) for name in city_names]
    # point_dict = {name: location for name, location in zip(data.columns, locations)}
    # pickle point dict
    import pickle
    # with open('point_dict.pickle', 'wb') as f:
    #    pickle.dump(point_dict, f)

    # read point_dict
    with open('point_dict.pickle', 'rb') as f:
        point_dict = pickle.load(f)

    # point_dict = eval(open('point_dict.py').read())
    print(point_dict)
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    ic = ee.ImageCollection(
        'ECMWF/ERA5_LAND/MONTHLY_AGGR')  # .filterDate('2022-01-01', '2023-01-01').select('total_precipitation_sum')

    ee_dataset = EEWrapper(ic)
    range_data = extract_same_range_from_climate_data(data, ee_dataset)
    d = range_data.total_precipitation_sum
    d2 = range_data.temperature_2m
    d3 = range_data.temperature_2m_max
    for name, point in list(point_dict.items())[1:2]:
        if point is None:
            continue
        ee_point = ee.Geometry.Point(point.longitude, point.latitude)
        print(name, ee_point)
        values = d[ee_point].compute()
        temperature = d2[ee_point].compute()
        new_data_frame = pd.DataFrame({'Date': data['periodname'],
                                       'Rainfall': values,
                                       'DengueCases': data[name],
                                       'Temperature': temperature})
        # data['precipitation'] = values
        print(name)
        new_data_frame.to_csv(f'{name}.csv')
        analyze_data(new_data_frame, exog_names=['Rainfall'])


    # with open('point_dict.py', 'w') as f:
    #     f.write(repr(point_dict))
    # for name, location in point_dict.items():
    #     print(location)
    #     print(data[name])

if __name__ == '__main__':
    main()