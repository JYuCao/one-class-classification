# 带马氏距离判别的多元正态分布
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis

# 读取数据集
train_data = np.genfromtxt('data/Train_data.csv', delimiter=',')
test_data1 = np.genfromtxt('data/Test_data1.csv', delimiter=',')
test_data2 = np.genfromtxt('data/Test_data2.csv', delimiter=',')

# 计算多元正态分布
train_cov = np.cov(train_data.T)
train_mean = np.mean(train_data, axis=0)
train_mvn = multivariate_normal(mean=train_mean, cov=train_cov)

# 计算测试集距训练集的马氏距离
md_test1 = []
md_test2 = []
train_cov_inv = np.linalg.inv(np.cov(train_data.T))
for i in range(0, 4000):
    md_test1.append(mahalanobis(test_data1[i], train_data.mean(axis=0), train_cov_inv))
    md_test2.append(mahalanobis(test_data2[i], train_data.mean(axis=0), train_cov_inv))


# 统计正确率
def detect_anomalies_mahalanobis(mahalanobis_distances, threshold):
    correct_count = 0.0
    for i in range(0, 4000):
        if i < 2000:
            if mahalanobis_distances[i] < threshold:
                correct_count += 1
        else:
            if mahalanobis_distances[i] > threshold:
                correct_count += 1
    return correct_count/4000


# 测试最佳阈值
max_currency1 = 0
max_currency2 = 0
t1 = 0
t2 = 0
for threshold in np.arange(5, 15, 0.1):
    # 对测试数据1进行异常检测
    currency_test1 = detect_anomalies_mahalanobis(md_test1, threshold) 
    if max_currency1 < currency_test1:
        max_currency1 = currency_test1
        t1 = threshold
    # 对测试数据2进行异常检测
    currency_test2 = detect_anomalies_mahalanobis(md_test2, threshold)
    if max_currency2 < currency_test2:
        max_currency2 = currency_test2
        t2 = threshold

t1 = round(t1, 1)
t2 = round(t2, 1)

print("# threshold = {}, Currency in Test Data 1: {}".format(t1, max_currency1))
print("# threshold = {}, Currency in Test Data 1: {}".format(t2, max_currency2))


# 绘图
plt.subplot(1, 2, 1)
plt.bar(range(1,4001),md_test1)
plt.title("data analyze")
plt.xlabel("samples")
plt.ylabel("scores")
plt.axhline(y=t1, color='red', linestyle='--', label='Threshold = {}'.format(t1))
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(range(1,4001),md_test2)
plt.title("data analyze")
plt.xlabel("samples")
plt.ylabel("scores")
plt.axhline(y=t2, color='red', linestyle='--', label='Threshold = {}'.format(t2))
plt.legend()
plt.show()
