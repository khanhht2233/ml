import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from PIL.ImageOps import fit
from fontTools.misc import transform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
import numpy as np

data = pd.read_csv('diabetes.csv')
# print(data.info())
# print(data.describe())

# result = data.corr()
# print(data["Outcome"].value_counts())

#
# plt.figure(figsize=(8,8))
# sn.histplot(data["Outcome"])
# plt.title("diabetes distribution")
# plt.savefig("test.jpg")

#set feature anf target: Phan chia du lieu theo chieu doc
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

#split data : phan chia du lieu theo chieu ngang
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
# print(len(x_train), len(y_train))
# print(len(x_test), len(y_test))
#
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# result = scaler.fit_transform(x_train[["Pregnancies"]])
#
# # fit_transform
# # fit
# # transform
#
# for i, j in zip(x_train["Pregnancies"].values, result):
#     print("Before {} After {}".format(i, j))


cls = SVC()
cls.fit(x_train, y_train)

y_predict = cls.predict(x_test)
# for i, j in zip(y_test, y_predict):
#     print("Actual {} Predicted {}".format(i, j))

print(classification_report(y_test, y_predict))

cm = np.array(confusion_matrix(y_test, y_predict, labels = [0, 1]))
confusion = pd.DataFrame(cm, index=['Not Diabetic', 'Diabetic'], columns=['Not Diabetic', 'Diabetic'])
sn.heatmap(confusion, annot=True, fmt='g')
plt.savefig('diabetes_prediction.jpg')