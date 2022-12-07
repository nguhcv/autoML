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
    return abs(actual.to_numpy() - predicted.to_numpy())

def F1_score(th, actual ,ground_true, predicted):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    differences = abs(actual - predicted)  # calculate difference
    anomaly_score = differences - th  # determine the anomaly score
    anomaly_score[anomaly_score <= 0] = 0.
    anomaly_score[anomaly_score > 0] = 1.

    # calculate TP,FP,TN,FN
    for index, value in enumerate(ground_true):
        if value == 1.:
            if anomaly_score[index] == 1:
                TP += 1
            elif anomaly_score[index] == 0.:
                FP += 1
        elif value == 0.:
            if anomaly_score[index] == 0:
                TN += 1
            elif anomaly_score[index] == 1.:
                FN += 1

    Precision = TP /(TP +FP)
    Recall = TP /(TP +FN)
    F_score = ( 2* Precision *Recall) / (Precision +Recall)
    return F_score

