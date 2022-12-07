
import pandas as pd
import json
from TSautoML.cleaning.imputation import TS_SimpleImputer
from sklearn.pipeline import Pipeline
from TSautoML.ML_algs.ad import Sarima
from TSautoML.cores.TSdata import TSData
from TSautoML.CASH.search_strategy import CompleteSearch
import inspect

if __name__ == '__main__':


    with open('ad_test.json', 'r') as file:
        data = json.load(file)

    print(data["dataset"]["path"])
    file = open(data['dataset']['path'], 'rb')

    power_demand_df = pd.read_csv(file, header=0, names=data["dataset"]["column_name"])


    '''TSData- main data structure in our framework'''
    power_demand_ts = TSData(time=power_demand_df[data['dataset']['column_name'][0]], value=power_demand_df[data['dataset']['column_name'][1]])
    print(len(power_demand_df.values))

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
        grid = CompleteSearch(dataset=power_demand_ts,
                              train_test_ratio=0.551,
                              pipeline=pipeline,
                              optimization_grids=paramsgrid,
                              save_path='/TSautoML/save_path/',
                              metric=data['metric'],
                              save_file_name='ad_test',
                              n_saved_models=3)
    else:
        grid = CompleteSearch(dataset=power_demand_ts,
                              train_test_ratio=0.551,
                              pipeline=pipeline,
                              optimization_grids=paramsgrid,
                              save_path='/TSautoML/save_path/',
                              metric='f1',
                              save_file_name='ad_test',
                              n_saved_models=3)



    '''fit the grid search'''
    grid.fit()
    best_score = grid.best_score()
    print(best_score)

    '''visualization for the best model performance'''
    grid.plot()


