import pandas as pd


def get_week(t):
    week, c, year = t.split()
    assert week[0] == 'W'
    week = int(week[1:])-1
    assert c == f'{year}Sun'
    year = int(year)

    return year*53+(week-1)


file_path = '/home/knut/Data/climatehealth/laopr.csv'
f = pd.read_csv(file_path, header=1)

f['Period'] = [get_week(t) for t in f['Period']]
df = f.pivot(index=['Period'], columns=['Data'], values='Value')
df = df.sort_index()

rainfall = df['Climate-Rainfall']
temperature = df['Climate-Temperature avg']
dengue = df['NCLE: 8. Acute watery diarrhea cases']
