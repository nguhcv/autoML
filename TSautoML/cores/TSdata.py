'''
Description:

This file contains the main object (data structure) in our framework:
    class: TSData: input of a ML pipeline
'''

import logging
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _show_error(info: str) -> ValueError:
    logging.error(info)
    return ValueError(info)

class TSData:
    """The main object of the autoML:
        - store a time series dataset.
        - input of ML pipeline
    Arguments:
    - value: A pandas.Series or pandas.DataFrame storing the series value(s)
        (default None).
     - time: a `pandas.Series` or `pandas.DatetimeIndex` storing the time
        values (default None)

    Attributes:
      time: A `pandas.Series` object storing the time values of the time
        series.
      value: A `pandas.Series` (if univariate) or `pandas.DataFrame` (if
        multivariate) object storing the values of each field in the time
        series.
      min: A float or `pandas.Series` representing the min value(s) of the
        time series.
      max: A float or `pandas.Series` representing the max value(s) of the
        time series.
    """

    _time: pd.Series
    _value: Union[pd.Series, pd.DataFrame]
    _min: float = np.nan
    _max: float = np.nan

    def __init__(
        self,
        time: Union[pd.Series, pd.DatetimeIndex, None],
        value: Union[pd.Series, pd.DataFrame],

    ) -> None:
        self.time = time
        self.value = value
        # self._calc_min_max_values()

    # @property
    # def time(self) -> pd.Series:
    #     return self._time

    # @property
    # def value(self) -> Union[pd.Series, pd.DataFrame]:
    #     return self._value

    @property
    def min(self) -> Union[pd.Series, float]:
        return self._min

    @property
    def max(self) -> Union[pd.Series, float]:
        return self._max

    def __len__(self) -> int:
        return len(self.value)

    def is_empty(self) -> bool:
        return self.value.empty and self.time.empty

    def _calc_min_max_values(self) -> None:
        # Get maximum and minimum values
        if not self.value.empty:
            if isinstance(self.value, pd.Series):
                self._min = np.nanmin(self.value)
                self._max = np.nanmax(self.value)
            else:
                self._min = self.value.min(skipna=True)
                self._max = self.value.max(skipna=True)
        else:
            self._min = np.nan
            self._max = np.nan



    def is_value_missing(self) -> bool:
        if self.value.isnull().sum().sum()>0:
            return True
        else:
            return False


    def is_univariate(self) -> bool:
        return len(self.value.shape) == 1

    def to_dataframe(self, standard_time_col_name: bool = False) -> pd.DataFrame:
        pass


    def to_array(self) -> np.ndarray:
        return self.to_dataframe().to_numpy()

    def plot(self):

        data = self.value
        print(len(data.shape))
        if len(data.shape) ==1:
            plt.plot(data)
            plt.show()
        elif len(data.shape)==2:
            for k in range (data.shape[0]):
                plt.plot(data[k])
            plt.show()

    def train_test_split(self, ratio=0.8):
        print(self.value.shape)
        if self.is_univariate():
            train_length = int(self.value.shape[0]* ratio)
            self.train_set = self.value[0:train_length]
            self.test_set = self.value[train_length:]





        pass


if __name__ == '__main__':
    source = 'C:/Project/autoML/auto_ML/dataset/air_passengers.csv'
    file = open(source, 'rb')
    air_passengers_df = pd.read_csv(file, header=0, names=["time", "passengers"])

    air_passengers_ts = TSData(time=air_passengers_df['time'], value=air_passengers_df['passengers'])
    air_passengers_ts.plot()
