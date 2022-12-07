import numpy as np

EPSILON = 1e-10


# class Forcasting_Metric():
#     def __init__(self):
#
#         pass
#     def __error(self, actual, predicted):
#         return np.linalg.norm((actual - predicted),axis=0)
#
#     def mean_error(self,actual, predicted):
#         return np.mean(self.__error(actual=actual, predicted=predicted))
#     def mean_absolute_error(self, actual, predicted):
#         return np.mean(np.abs(self.__error(actual=actual, predicted=predicted)))
#
#     def mean_squared_error(self, actual, predicted):
#         return np.mean(np.square(self.__error(actual=actual, predicted=predicted)))
#
#     def root_mean_squared_error(self,actual, prediced):
#         return np.sqrt(self.mean_squared_error(actual=actual, predicted=prediced))



def __error(actual, predicted):
    # return np.linalg.norm((actual.to_numpy() - predicted.to_numpy()),axis=0)
    return actual.to_numpy() - predicted.to_numpy()

def mean_error(actual, predicted):     #me
    return np.mean(__error(actual=actual, predicted=predicted))
def mean_absolute_error(actual, predicted):   #mae
    return np.mean(np.abs(__error(actual=actual, predicted=predicted)))

def mean_squared_error(actual, predicted):    #mse
    return np.mean(np.square(__error(actual=actual, predicted=predicted)))

def root_mean_squared_error(actual, prediced): #rmse
    return np.sqrt(mean_squared_error(actual=actual, predicted=prediced))