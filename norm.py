# 使用多元正态分布和概率密度进行估计
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# 读取数据集
train_data = np.genfromtxt('data/Train_data.csv', delimiter=',')
test_data1 = np.genfromtxt('data/Test_data1.csv', delimiter=',')
test_data2 = np.genfromtxt('data/Test_data2.csv', delimiter=',')

# 计算多元正态分布
train_cov = np.cov(train_data.T)
train_mean = np.mean(train_data, axis=0)
train_mvn = multivariate_normal(mean=train_mean, cov=train_cov)
print(train_mean.shape)

# 统计正确率
def detect_anomalies(probabilities, threshold):
    currect = 0.0
    for i in range(0, 4000):
        if i < 2000:
            if probabilities[i] >= threshold:
                currect += 1
        else:
            if probabilities[i] < threshold:
                currect += 1
    return currect/4000

pdf_value = train_mvn.pdf(train_data)
pdf_value1 = train_mvn.pdf(test_data1)
pdf_value2 = train_mvn.pdf(test_data2)

threshold = sorted(pdf_value)[0]

# 对测试数据1进行异常检测
currency_test1 = detect_anomalies(pdf_value1, threshold)
print("Currency in Test Data 1:", currency_test1)

# 对测试数据2进行异常检测
currency_test2 = detect_anomalies(pdf_value2, threshold)
print("Currency in Test Data 2:", currency_test2)

# 绘图
plt.bar(range(1,4001),pdf_value1)
plt.title("data analyze")
plt.xlabel("samples")
plt.ylabel("scores")
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()
