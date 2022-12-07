from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TSautoML.cores.TSdata import TSData
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm


class Decomposition(BaseEstimator,RegressorMixin):
    def __init__(self, decomp:str='additive', iqr_mult:float=3.0):

        if decomp in ['additive', 'multiplicative']:
            self.decomp = decomp
        else:
            raise ValueError('decomp should be: additive or multiplicative')
        self.iqr_mult = iqr_mult

    def _clean_ts(self):
        result = seasonal_decompose(
            x= self.__traindata,model=self.decomp
        )
        rem = result.resid
        detrend = self.__traindata - result.resid
        strength = float(1 - np.nanvar(rem) / np.nanvar(detrend))
        if strength >= 0.6:
            self.__traindata = self.__traindata- result.seasonal

        # using IQR as threshold
        resid = self.__traindata - result.trend
        resid_q = np.nanpercentile(resid, [25, 75])
        iqr = resid_q[1] - resid_q[0]
        limits = resid_q + (self.iqr_mult * iqr * np.array([-1, 1]))
        # calculate scores
        output_scores = list((resid - limits[0]) / (limits[1] - limits[0]))
        outliers = resid[(resid > limits[1]) | (resid < limits[0])]
        outliers_index = list(outliers.index)

        return outliers_index, output_scores
        pass


    def fit(self,X,y=None,sample_weight=None ):
        if not isinstance(X, pd.Series):

            msg = (
                "Only support univariate time series, but got "
                f"{type(X)}."
            )
            raise ValueError(msg)
        else:
            self.__traindata = X
            self._clean_ts()


            

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
            # self.fcst = model.predict(start= l_trainset, end=l_trainset+l_testset-1,dynamic=True)
            self.fcst = model.predict(start=0, end=l_trainset-1, dynamic=True)
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





if __name__ == '__main__':

    pass






