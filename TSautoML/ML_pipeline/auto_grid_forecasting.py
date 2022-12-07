from TSautoML.cleaning.imputation import TS_SimpleImputer
from sklearn.pipeline import Pipeline
from TSautoML.ML_algs.forcasting import Sarima,Arima
from TSautoML.cores.TSdata import TSData
from TSautoML.CASH.search_strategy import CompleteSearch


def auto_grid_forcasting(dataset:TSData,
                          train_test_ratio:float,
                          save_path:str,
                          metric:str,
                          n_saved_models:int):
    '''define pipeline structure'''
    pipeline = Pipeline(steps=[("cleaning", TS_SimpleImputer()), ("classifier", Arima())])

    '''build hyperparameter grid'''
    optimization_grid = []

    optimization_grid.append({
        'cleaning__metric': ['mean', 'median'],
        'classifier__order': [(1, 1, 1), (1, 1, 2), (1, 2, 3), (1, 3, 3), (2, 2, 3), (2,3,3)],
        'classifier__seasonal_order': [(1, 1, 1, 12), (1, 1, 1, 24), (1, 1, 1, 60), (1, 1, 1, 30),
                                       (1, 2, 3, 30), (1,2,3,12),(1,2,3,24),(1,2,3,60),
                                       (2,2,3,12), (2,2,3,24), (2,2,3,30), (2,2,3,60)],
        'classifier': [Sarima()]

    })

    optimization_grid.append({
        'cleaning__metric': ['mean', 'median'],
        'classifier__order': [(1, 1, 1), (1, 1, 3), (1, 1, 2), (1, 2, 3), (2,2,3), (3,2,3), (3,2,1)],
        'classifier': [Arima()]
    })

    # build search strategy
    grid = CompleteSearch(dataset=dataset,
                          train_test_ratio=train_test_ratio,
                          pipeline= pipeline,
                          optimization_grids=optimization_grid,
                          save_path=save_path,
                          metric=metric,
                          n_saved_models=n_saved_models)
    grid.fit()
    best_score = grid.best_score()
    print(best_score)
    return grid


    pass



