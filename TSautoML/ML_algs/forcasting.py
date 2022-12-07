from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from TSautoML.cores.TSdata import TSData
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

class Arima(BaseEstimator,RegressorMixin):
    def __init__(self, order=(1,1,1) ):
        self.order = order

    def fit(self,X,y=None,sample_weight=None ):
        if not isinstance(X, pd.Series):
            msg = (
                "Only support univariate time series, but got "
                f"{type(X)}."
            )
            raise ValueError(msg)
        else:
            self.__traindata = X
            arima = ARIMA(self.__traindata, order=(self.order))
            self.model = arima.fit()
            return self

    def predict(self,X,steps=30):
        model = self.model
        if model is None:
            raise ValueError("Call fit() before predict().")
        else:
            l_trainset= len(self.__traindata.to_numpy())
            l_testset = len(X.to_numpy())
            self.fcst = model.predict(start= l_trainset, end=l_trainset+l_testset-1,dynamic=True)
            return self.fcst

    def score(self, X, y, sample_weight=None):
        if self.fcst is None:
            raise ValueError("Call predict() before score().")
        else:
            return np.mean(np.abs(self.fcst.to_numpy()-X.to_numpy()))


class Sarima(BaseEstimator,RegressorMixin):
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order=order
        self.seasonal_order = seasonal_order

    def fit(self,X,y=None,sample_weight=None ):
        if not isinstance(X, pd.Series):
            msg = (
                "Only support univariate time series, but got "
                f"{type(X)}."
            )
            raise ValueError(msg)
        self.__traindata = X
        sarima = sm.tsa.statespace.SARIMAX(self.__traindata, order=self.order, seasonal_order=self.seasonal_order)
        self.model = sarima.fit()
        return self

    def predict(self,X):
        model = self.model
        if model is None:
            raise ValueError("Call fit() before predict().")
        else:
            l_trainset= len(self.__traindata.to_numpy())
            l_testset = len(X.to_numpy())
            self.fcst = model.predict(start= l_trainset, end=l_trainset+l_testset-1,dynamic=True)
            return self.fcst

    def score(self, X, y, sample_weight=None):
        if self.fcst is None:
            raise ValueError("Call predict() before score().")
        else:
            return np.mean(np.abs(self.fcst.to_numpy()-X.to_numpy()))

    def plot(self,X):
        fcts = self.fcst
        if fcts is None:
            raise  ValueError("call predict before plot")
        else:
            plt.plot(fcts)
            plt.plot(X)
            plt.show()


class AR(BaseEstimator,RegressorMixin):
    def __init__(self, trend:str='n', seasonal:bool=True, period:int=12, lags:int=20):
        self.trend = trend
        self.seasonal=seasonal
        self.period=period
        self.lags = lags

    def fit(self,X,y=None,sample_weight=None ):
        if not isinstance(X, pd.Series):
            msg = (
                "Only support univariate time series, but got "
                f"{type(X)}."
            )
            raise ValueError(msg)
        else:
            self.__traindata = X
            AR = AutoReg(self.__traindata, lags=self.lags,trend=self.trend,seasonal=self.seasonal,period=self.period)
            self.model = AR.fit()
            return self

    def predict(self,X):
        model = self.model
        if model is None:
            raise ValueError("Call fit() before predict().")
        else:
            l_trainset= len(self.__traindata.to_numpy())
            l_testset = len(X.to_numpy())
            self.fcst = model.predict(start= l_trainset, end=l_trainset+l_testset-1,dynamic=True)
            return self.fcst

    def score(self, X, y, sample_weight=None):
        if self.fcst is None:
            raise ValueError("Call predict() before score().")
        else:
            return np.mean(np.abs(self.fcst.to_numpy()-X.to_numpy()))



if __name__ == '__main__':
    a = Arima(order=(1,1,1))
    print(a.get_params())

    b = Sarima(order =(1,1,1),seasonal_order=(1,1,1,12))
    print(b.get_params())

    source = 'C:/Project/autoML/auto_ML/dataset/air_passengers.csv'
    file = open(source, 'rb')
    air_passengers_df = pd.read_csv(file, header=0, names=["time", "passengers"])

    # convert to TimeSeriesData object
    # air_passengers_ts = TimeSeriesData(air_passengers_df)
    air_passengers_ts = TSData(time=air_passengers_df['time'],value=air_passengers_df['passengers'])

    a.fit(air_passengers_ts)
    op = a.predict(X=air_passengers_ts)
    #
    # print(op)
    print(op)
    plt.plot(air_passengers_ts.value)
    plt.plot(op)
    plt.show()

    b.fit(air_passengers_ts)
    op2 = b.predict(X=air_passengers_ts)
    print(op2)
    plt.plot(air_passengers_ts.value)
    plt.plot(op2)
    plt.show()



    # params={'p':[1,2,3], 'd':[1,2,3], 'q':[1,2,3]}
    # gs =GridSearchCV(Arima(), param_grid=params, cv=4)





