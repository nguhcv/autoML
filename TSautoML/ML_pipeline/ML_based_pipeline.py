import abc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from TSautoML.cores.TSdata import TSData
#model

from TSautoML.ML_pipeline.auto_grid_forecasting import auto_grid_forcasting



class ML_basedPipeline(abc.ABC):
    @abc.abstractmethod
    def __init__(self,):

        pass

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def plot_results(self):
        pass

    @abc.abstractmethod
    def best_scores(self):
        pass





class MLbased_Forecasting_pipeline(ML_basedPipeline):
    def __init__(self,
                 dataset:TSData,
                 train_test_ratio:float,
                 save_path:str,
                 metric:str,
                 n_saved_models:int=3,
                 tuning:str='grid'
                 ):
        super(MLbased_Forecasting_pipeline, self).__init__()
        self.dataset=dataset
        self.train_test_ratio = train_test_ratio
        self.save_path = save_path
        self.metric = metric
        self.n_saved_model = n_saved_models
        self.tuning = tuning

        pass

    def run(self):
        print('run here')
        if self.tuning=='grid':
            self.grid = auto_grid_forcasting(dataset=self.dataset,
                                 train_test_ratio=self.train_test_ratio,
                                 save_path=self.save_path,
                                 metric=self.metric,
                                 n_saved_models=self.n_saved_model)


    def plot_results(self):
        self.grid.plot()
        pass

    def best_scores(self):
        return self.grid.best_score()
        pass


