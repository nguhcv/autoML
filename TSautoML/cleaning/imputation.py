from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData
from TSautoML.cores.TSdata import TSData
from kats.models.model import Model
from kats.utils.parameter_tuning_utils import get_default_arima_parameter_search_space
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
import statsmodels.api as sm


class TS_SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, metric='mean'):
        assert metric in ['mean', 'median'], 'Unrecognized value for metric, should be mean/median'
        self.metric = metric
        self.mean_value = None
        self.median_value = None

    def fit(self, X, y=None):

        #check missing value in each column of pandas frame
        if isinstance(X, pd.Series):
            if self.metric =='mean':
                self.mean_value = X.mean(skipna=True)
                # X.value.fillna(value=mean_value, inplace=True)
            elif self.metric =='median':
                self.median_value = X.median(skipna=True)
                # X.value.fillna(value=median_value, inplace=True)


        elif isinstance(X.value, pd.DataFrame):
            for col_number in X.columns:
                col = X.value[col_number]
                print(col.values)

        return self

    def transform(self, X, y=None):

        if isinstance(X, pd.Series):
            if self.mean_value is None and self.median_value is None:
                raise ValueError('fit function should be called first')
            else:
                if self.metric =='mean':
                    X.fillna(value=self.mean_value, inplace=True)
                elif self.metric =='median':
                    X.fillna(value=self.median_value, inplace=True)
        return X

if __name__ == '__main__':
    source = 'C:/Project/autoML/auto_ML/dataset/air_passengers.csv'
    file = open(source, 'rb')
    air_passengers_df = pd.read_csv(file, header=0, names=["time", "passengers"])

    # convert to TimeSeriesData object
    # air_passengers_ts = TimeSeriesData(air_passengers_df)
    air_passengers_ts = TSData(time=air_passengers_df['time'], value=air_passengers_df['passengers'])

    a = TS_SimpleImputer()
    a.fit(X=air_passengers_ts)
    a.transform(X=air_passengers_ts)