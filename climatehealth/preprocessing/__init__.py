from datetime import date
import numpy as np
import pandas as pd

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']

lookup = dict(zip(months, range(12)))


def get_city_name(location):
    return location.split(maxsplit=6)[-1]


def get_date(month, year):
    return date(year, lookup[month]+1, 1)


def get_dataset(filename):
    data = pd.read_csv(filename, sep='\t', header=1)
    data['periodname'] = [get_date(s.strip().split()[0], int(s.strip().split()[-1])) for s in data['periodname']]
    data = data.sort_values(by='periodname')
    data = data.iloc[:-2] # remove november, december 2023
    for column_name in data.columns[1:]:
        column = data[column_name]
        data[column_name] = column.replace(np.nan, 0)

    return data


processors  = {'10yeardengudata.csv': get_dataset}