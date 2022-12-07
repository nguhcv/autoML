
import pandas as pd
import json
from TSautoML.cleaning.imputation import TS_SimpleImputer
from sklearn.pipeline import Pipeline
from TSautoML.ML_algs.forcasting import Sarima,Arima
from TSautoML.cores.TSdata import TSData
from TSautoML.CASH.search_strategy import CompleteSearch
import inspect

if __name__ == '__main__':
    # data= {
    #       "dataset":
    #         {
    #             "type": "univariate",
    #             "path": "C:/Project/autoML/auto_ML/dataset/air_passengers.csv",
    #             "column_name": ["time", "value"],
    #             "input_column": "value"
    #         },
    #
    #     "preprocessing": {
    #         "handling_missing": ["mean", "median"],
    #         "normalization": ["l1", "l2", "max"],
    #     },
    #
    #     "model": "sarima",
    #     "h_param": {
    #         "order": [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 3, 1), (1, 3, 2), (1, 3, 3),
    #                   (2, 1, 1), (2, 1, 2)],
    #         "seasonal_order": [(1, 1, 1,12), (1, 1, 1,24), (1, 1, 1,30), (1, 1, 1,60)]
    #
    #     },
    #     "metric": ["mae"]
    # }

    with open('fc_test.json', 'r') as file:
        data = json.load(file)

    print(data["dataset"]["path"])
    file = open(data['dataset']['path'], 'rb')
    air_passengers_df = pd.read_csv(file, header=0, names=data["dataset"]["column_name"])

    '''TSData- main data structure in our framework'''
    air_passengers_ts = TSData(time=air_passengers_df[data['dataset']['column_name'][0]], value=air_passengers_df[data['dataset']['column_name'][1]])

    steps = []
    paramsgrid={}

    if "preprocessing" in data:
        if len(data["preprocessing"])> 0:
            for i in range (len(data["preprocessing"])):
                tp = list(data["preprocessing"].keys())[i]
                if tp =='handling_missing':
                    steps.append(('handling_missing', TS_SimpleImputer()))
                    values = list(data["preprocessing"].values())[i]
                    arguments = inspect.getargspec(TS_SimpleImputer)
                    for j in range (1,len(arguments[0])):
                        key = str(tp + '__' + arguments[0][j])
                        paramsgrid.update({key: values})

                # if tp=='normalization':
                #     steps.append('normal')

    if "model" in data:
        print(data['model'])
        if data['model'] =='sarima':
            steps.append(('model', Sarima()))
            if "h_param" in data:
                for i in range(len(data["h_param"])):
                    k = list(data["h_param"].keys())[i]
                    v = list(data["h_param"].values())[i]
                    key = str('model__'+ k)
                    paramsgrid.update({key: v})
                    print(paramsgrid)
        elif data['model'] =='arima':
            steps.append(('model',Arima()))

    #build pipeline
    pipeline = Pipeline(steps=steps)
    paramsgrid = [paramsgrid]

    if 'metric' in data:
        '''build search strategy'''
        grid = CompleteSearch(dataset=air_passengers_ts,
                              train_test_ratio=0.8,
                              pipeline=pipeline,
                              optimization_grids=paramsgrid,
                              save_path='/TSautoML/save_path/',
                              metric=data['metric'],
                              save_file_name='fc_test',
                              n_saved_models=3)
    else:
        grid = CompleteSearch(dataset=air_passengers_ts,
                              train_test_ratio=0.8,
                              pipeline=pipeline,
                              optimization_grids=paramsgrid,
                              save_path='/TSautoML/save_path/',
                              metric='mae',
                              save_file_name='fc_test',
                              n_saved_models=3)



    '''fit the grid search'''
    grid.fit()
    best_score = grid.best_score()
    print(best_score)

    '''visualization for the best model performance'''
    grid.plot()


