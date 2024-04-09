# 使用one class svm进行估计
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

train_data = np.genfromtxt('data/Train_data.csv', delimiter=',')
test_data1 = np.genfromtxt('data/Test_data1.csv', delimiter=',')
test_data2 = np.genfromtxt('data/Test_data2.csv', delimiter=',')

clf = svm.OneClassSVM(nu=0.1,kernel='rbf')
clf.fit(train_data)
pred = clf.predict(test_data1)

# scores = clf.decision_function(train_data)
# print(max(scores))
# print(min(scores))
# threshold = scores.mean() + scores.std()
# print(threshold)
# scores1 = clf.decision_function(test_data1)
# scores2 = clf.decision_function(test_data2)


plt.bar(range(1,4001),pred)
plt.title("test_data1 analyze")
plt.xlabel("samples")
plt.ylabel("scores")
# plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
# plt.legend()
plt.show()
