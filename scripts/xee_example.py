import ee
import xarray

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
ds = xarray.open_dataset('ee://ECMWF/ERA5_LAND/HOURLY', engine='ee')
ds = xarray.open_dataset('ee://ECMWF/ERA5_LAND/HOURLY', engine='ee',
                         crs='EPSG:4326', scale=0.25)

ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate('1992-10-05', '1993-03-31')
ds = xarray.open_dataset(ic, engine='ee', crs='EPSG:4326', scale=0.25)

print(ds)