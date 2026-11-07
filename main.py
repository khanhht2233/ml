import pandas as pd


data = pd.read_csv('diabetes.csv')
# print(data.info())
# print(data.describe())

result = data.corr()
print(data.value_counts())


