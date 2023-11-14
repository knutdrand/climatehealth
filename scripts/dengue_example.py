from datetime import date
import pandas as pd
import numpy as np

from climatehealth.geo_coding import get_location

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
          'December']
lookup = dict(zip(months, range(12)))


def get_city_name(location):
    return location.split(maxsplit=6)[-1]

def get_date(month, year):
    return date(year, lookup[month]+1, 1)


data = pd.read_csv('../example_data/10yeardengudata.csv', sep='\t', header=1)
years = np.array([int(s.strip().split()[-1]) for s in data['periodname']])


data['periodname'] = [get_date(s.strip().split()[0], int(s.strip().split()[-1])) for s in data['periodname']]
#month = [lookup[s.strip().split()[0]] for s in data['periodname']]
# data['periodname'] = (years * 12 + month)
data = data.sort_values(by='periodname')
print(data.periodname)
#data = pd.melt(data, id_vars='periodname', var_name='location')
#data['year'] = [str(year) for year in data['periodname'] // 12]
#data['month'] = data['periodname'] % 12
#print(data)
#city_names = [get_city_name(c) for c in data.columns[1:]]
#locations = [get_location(name) for name in city_names]
#point_dict = {name: location for name, location in zip(data.columns, locations)}
# with open('point_dict.py', 'w') as f:
#     f.write(repr(point_dict))
# for name, location in point_dict.items():
#     print(location)
#     print(data[name])