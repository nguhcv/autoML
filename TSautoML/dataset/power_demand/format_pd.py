import pandas as pd
import matplotlib.pyplot as plt
# path = "C:/Project/autoML/auto_ML/dataset/power_demand/raw/power_data.txt"
#
# file = open(path, 'rb')
# df = pd.read_fwf(file)
# df['time'] = df.index
# df.rename(columns={'950': 'value'}, inplace=True)
# col_list = list(df.columns)
# x, y = col_list.index('value'), col_list.index('time')
# col_list[y], col_list[x] = col_list[x], col_list[y]
# df = df[col_list]
#
# print(df.head())
# df.to_csv('converted.csv', sep=',', index=False)

# breakpoint()
import numpy as np

path = "/TSautoML/dataset/power_demand/labeled/"
trainfile = open(path + '/train/power_data.pkl', 'rb')
testfile = open(path + '/test/power_data.pkl', 'rb')
tr_data = pd.DataFrame(pd.read_pickle(trainfile))
train_data = tr_data[[0]].to_numpy()
train_data = train_data.T
train_label = tr_data[[1]].to_numpy()
train_label = train_label.T
# train_label = np.reshape(train_label, newshape=(train_label.shape[0],))

te_data = pd.DataFrame(pd.read_pickle(testfile))
test_data = te_data[[0]].to_numpy()
test_data = test_data.T
test_label = te_data[[1]].to_numpy()
test_label = test_label.T
# test_label = np.reshape(test_label, newshape=(test_label.shape[0],))


ful_sereis = np.concatenate((train_data[0], test_data[0]))
full_labels = np.concatenate((train_label[0], test_label[0]))


df = pd.DataFrame(ful_sereis)
print(df.head())

df = df.reset_index()
df = df.rename(columns={0: "value", "index": "time"})
df['label'] = full_labels

print(df.head())
df.to_csv('full_series.csv', sep=',', index=False)



# breakpoint()
#
# plt.plot(ful_sereis)
# plt.show()
#
# plt.plot(ful_sereis[18144: 20200])
# plt.show()