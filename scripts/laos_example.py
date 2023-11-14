import ee
import xarray

# ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
# import ee

# Authenticate to the Earth Engine servers
# ee.Authenticate()
ee.Initialize()

# Define the point location
point = ee.Geometry.Point(102.250915, 19.227447)

# Daily precipitation for one year in meters (multiply with 1000 to get mm)
precipitation = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                 .select('total_precipitation_sum')
                 .filterDate('2022-01-01', '2023-01-01')
                 .map(lambda image: ee.Feature(
                     None,
                     image.reduceRegion(ee.Reducer.mean(), point, 1)
                 )))

print('precipitation', precipitation.getInfo())

# Daily temperature for one year in kelvin (subtract 273.15 to get °C)
temperature = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
               .select('temperature_2m')
               .filterDate('2022-01-01', '2023-01-01')
               .map(lambda image: ee.Feature(
                   None,
                   image.reduceRegion(ee.Reducer.mean(), point, 1)
               )))

print('temperature', temperature.getInfo())
'''
// Dataset:
// https: // developers.google.com / earth - engine / datasets / catalog / ECMWF_ERA5_LAND_DAILY_AGGR

// Kasi
Hospital in Laos: https: // maps.app.goo.gl / vWc1MkifMJPKqoB26
var
point = ee.Geometry.Point(102.250915, 19.227447)

// Daily
precipitation
for one year in meters(multiply with 1000 to get mm)
var
precipitation = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
.select('total_precipitation_sum')
.filterDate('2022-01-01', '2023-01-01')
.map(function(image)
{
return ee.Feature(
    null,
    image.reduceRegion(ee.Reducer.mean(), point, 1)
)
})

print('precipitation', precipitation)

// Daily
temperature
for one year in kelvin(subtract 273.15 to get °C)
var temperature = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
.select('temperature_2m')
.filterDate('2022-01-01', '2023-01-01')
.map(function(image) {
return ee.Feature(
    null,
    image.reduceRegion(ee.Reducer.mean(), point, 1)
)
})

print('temperature', temperature)
'''