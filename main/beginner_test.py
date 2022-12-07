


from TSautoML.cores.TSdata import TSData
from TSautoML.cores.autoML_setting import autoML
import pandas as pd



if __name__ == '__main__':
    '''Read air-passenger dataset'''
    source = 'C:/Project/autoML/auto_ML/dataset/air_passengers.csv'
    file = open(source, 'rb')
    air_passengers_df = pd.read_csv(file, header=0, names=["time", "passengers"])

    '''TSData- main data structure in our framework'''
    air_passengers_ts = TSData(time=air_passengers_df['time'], value=air_passengers_df['passengers'])

    '''build a autoML instance'''
    forecastingML = autoML(dataset=air_passengers_ts,
                           task='fc',
                           approach='ML',
                           metric='mae',
                           saved_path='/TSautoML/save_path/',
                           train_test_split_ratio=0.8,
                           tuning_mechanism='grid')

    '''deploy autoML from given dataset'''
    forecastingML.deploy()

    '''show the best models'''
    forecastingML.best_score()

    '''plot the best result'''
    forecastingML.plot_result()





