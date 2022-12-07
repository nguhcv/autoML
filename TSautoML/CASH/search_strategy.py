from sklearn.model_selection import ParameterGrid
from TSautoML.cores.TSdata import TSData
from sklearn.pipeline import Pipeline
from TSautoML.eval_metrics.forcasting_metrics import *
from TSautoML.eval_metrics.anomaly_metrics import F1_score
import joblib
import matplotlib.pyplot as plt

'''implement a gid search strategy for ML pipeline'''

class BaseSearch():
    def __init__(self):
        pass

    def param_combination(self):
        pass

    def __implement(self):
        pass

    def fit(self):
        pass

    def __scoring(self, test, predicted):
        pass

    def best_score(self):
        pass
    def plot(self):
        pass


class CompleteSearch(BaseSearch):
    def __init__(self,dataset:TSData,
                 pipeline:Pipeline,
                 optimization_grids,
                 save_path,
                 save_file_name,
                 train_test_ratio=0.8,
                 metric:str = 'mae',
                 n_saved_models:int=6,
                 labels=None
                 ):
        super(CompleteSearch, self).__init__()

        self.pipeline = pipeline
        self.optimization_grid = optimization_grids
        self.dataset = dataset
        self.save_path = save_path
        self.train_set_ratio = train_test_ratio
        self.n_saved_models = n_saved_models
        self.save_file_name = save_file_name
        self.labels = labels

        if metric =='me' or metric =='mae' or metric =='mse' or metric=='f1':
            self.metric = metric
        else:
            raise ValueError('please select one of these metrics: me, mae, mse, f1')


    def __param_combination(self):
        grid_combine = []
        for grid in self.optimization_grid:
            li = list(ParameterGrid(grid))
            grid_combine.append(li)
        return grid_combine
        pass

    def __implement(self):

        # build combination param grids
        self.grid_combination = self.__param_combination()

        #train_test_split
        self.dataset.train_test_split(ratio=self.train_set_ratio)

        #fit pipeline for each case
        self.b_scores=[]
        for i in range (len(self.grid_combination)):
            for combination in self.grid_combination[i]:
                self.pipeline.set_params(**combination)
                self.pipeline.fit(X=self.dataset.train_set,y=None)
                predicted = self.pipeline.predict(X=self.dataset.test_set)
                score =self.__scoring(test=self.dataset.test_set, predicted= predicted)

                if len(self.b_scores)<self.n_saved_models:
                    self.b_scores.append([score, combination, self.pipeline, predicted])

                else:
                    self.b_scores = sorted(self.b_scores, key=lambda x: x[0])
                    # max_score = max(self.b_scores, key=lambda x:x[0])
                    # idx = self.b_scores.index(max_score)
                    if self.b_scores[-1][0] > score:
                        self.b_scores[-1][0] = score
                        self.b_scores[-1][1] = combination
                        self.b_scores[-1][2] = self.pipeline
                        self.b_scores[-1][3] = predicted


        joblib.dump(self.b_scores, self.save_path + self.save_file_name)


    def fit(self):
        self.__implement()

    def __scoring(self, test, predicted, label=None):
        if self.metric =='me':
            return mean_error(actual=test, predicted=predicted)
            pass
        elif self.metric=='mse':
            return mean_squared_error(actual=test, predicted=predicted)
            pass
        elif self.metric=='mae':
            return mean_absolute_error(actual=test, predicted=predicted)
            pass

        elif self.metric=='f1':
            return F1_score(th=0.5, actual=test, predicted=predicted,ground_true=self.labels)


    def best_score(self):
        self.b_scores = sorted(self.b_scores, key=lambda x: x[0])
        for element in self.b_scores:
            print(element[0], element[1])
        return self.b_scores

    def plot(self):
        plt.plot(self.b_scores[0][-1], label='Best Model')
        plt.plot(self.dataset.test_set, label='Test set')
        plt.legend()
        plt.show()

        pass

