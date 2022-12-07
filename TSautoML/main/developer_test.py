from TSautoML.cleaning.imputation import TS_SimpleImputer
from sklearn.pipeline import Pipeline
from TSautoML.ML_algs.forcasting import Sarima
import pandas as pd
from TSautoML.cores.TSdata import TSData
from TSautoML.CASH.search_strategy import CompleteSearch


'''Read air-passenger dataset'''
source = 'C:/Project/autoML/auto_ML/dataset/air_passengers.csv'
file = open(source, 'rb')
air_passengers_df = pd.read_csv(file, header=0, names=["time", "passengers"])

'''TSData- main data structure in our framework'''
air_passengers_ts = TSData(time=air_passengers_df['time'],value=air_passengers_df['passengers'])

'''define pipeline structure'''
pipeline = Pipeline(steps=[("cleaning", TS_SimpleImputer()),("classifier", Sarima())])

'''build hyperparameter grid'''
optimization_grid=[]

optimization_grid.append({
    'cleaning__metric':['mean', 'median'],
    'classifier__order':[(1,1,1), (1,1,2),(1,2,3), (1,3,3), (1,2,2)],
    'classifier__seasonal_order':[(1,1,1,12), (1,1,1,24), (1,1,1,60), (1,1,1,30)],
    'classifier': [Sarima()]})

'''build search strategy'''
grid = CompleteSearch(dataset=air_passengers_ts,
                      train_test_ratio=0.8,
                      pipeline=pipeline,
                      optimization_grids=optimization_grid,
                      save_path='/TSautoML/save_path/',
                      metric='mae',
                      n_saved_models = 3)

'''fit the grid search'''
grid.fit()
best_score = grid.best_score()
print(best_score)

'''visualization for the best model performance'''
grid.plot()




#
# pipeline.fit(X=air_passengers_ts.train_set, y=None)
# pipeline.predict(X=air_passengers_ts.test_set)
# score = pipeline.score(X=air_passengers_ts.test_set,y=None)
# print(score)
# breakpoint()