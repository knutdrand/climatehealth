import ee
import numpy as np

from .indexing_interface import SpatioTemporalIndexable, TimePoint


class EEWrapper(SpatioTemporalIndexable):
    def __init__(self, ee_object, name=None):
        self._ee_object = ee_object
        self._name = name

    def _get_temporal_item(self, item: TimePoint):
        if isinstance(item, slice):
            value = self._ee_object.filterDate(item.start, item.stop)
            return self._wrap(value)
        else:
            raise ValueError(f'Cannot index {self.__class__.__name__} with {item}')

    def _get_spatial_item(self, item):
        # if isinstance(item, ee.Geometry.Point):
        if isinstance(item, list):
            #use image.RedcueRegions instead
            # FeatureCollection from a list of features.
            # list_of_features = [
            #     ee.Feature(ee.Geometry.Point(-62.54, -27.32), {'key': 'val1'}),
            #     ee.Feature(ee.Geometry.Point(-69.18, -10.64), {'key': 'val2'}),
            #     ee.Feature(ee.Geometry.Point(-45.98, -18.09), {'key': 'val3'})
            # ]
            #list_of_features_fc = ee.FeatureCollection(list_of_features)
            pass
        value = self._ee_object.map(lambda image: ee.Feature(None, image.reduceRegion(ee.Reducer.mean(), item, 1)))
        return self._wrap(value)

    def _wrap(self, value):
        return self.__class__(value, name=self._name)

    def __getattr__(self, name):
        # Also get two names
        return self.__class__(self._ee_object.select(name), name=name)

    def __repr__(self):
        return repr(self._ee_object)

    def __str__(self):
        return str(self._ee_object)

    def compute(self):
        assert self._name is not None
        features = self._ee_object.getInfo()['features']
        return np.array([f['properties'][self._name] for f in features])


def get_image_collection(period='MONTHLY', dataset='ERA5'):
    dataset_lookup = {'ERA5': 'ECMWF/ERA5_LAND'}
    name = f'{dataset_lookup[dataset]}/{period}_AGGR'
    ic = ee.ImageCollection(
        name)  # .filterDate('2022-01-01', '2023-01-01').select('total_precipitation_sum')
    return ic
