import typing
from datetime import date


class SpatialLocation:
    pass


class TimePoint:
    @classmethod
    def accepts_string(cls, text):
        try:
            date.fromisoformat('2019-12-04')
            return True
        except ValueError:
            return False

# data['2022-01-01':'2023-01-01', point].total_precipitation_sum

class SpatioTemporalIndexable(typing.Protocol):
    def _get_temporal_item(self, item: TimePoint):
        return NotImplemented

    def _is_temporal(self, item):
        return isinstance(item, TimePoint) or isinstance(item, str) or isinstance(item, slice)# TimePoint.accepts_string(item)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            cur = self
            for i in item:
                cur = cur[i]
            return cur
        if self._is_temporal(item):
            return self._get_temporal_item(item)
        else:
            return self._get_spatial_item(item)


