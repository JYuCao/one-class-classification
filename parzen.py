from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

# 读取数据集
train_data = np.genfromtxt('data/Train_data.csv', delimiter=',')
test_data1 = np.genfromtxt('data/Test_data1.csv', delimiter=',')
test_data2 = np.genfromtxt('data/Test_data2.csv', delimiter=',')

# 使用Parzen窗法进行拟合
bandwidth = 1.0
parzen_estimator = KernelDensity(bandwidth=bandwidth)
parzen_estimator.fit(train_data)

log_density = parzen_estimator.score_samples(train_data)
log_density1 = parzen_estimator.score_samples(test_data1)
log_density2 = parzen_estimator.score_samples(test_data2)

# 使用拟合好的Parzen窗法来评估测试样本的密度
threshold = np.percentile(sorted(log_density), 0)

# 计算异常值的函数
def detect_anomalies(log_density, threshold):
    currect = 0.0
    for i in range(0, 4000):
        if i < 2000:
            # 高于控制线的正常值计数
            if log_density[i] >= threshold:
                currect += 1
        else:
            # 低于控制线的异常值计数
            if log_density[i] < threshold:
                currect += 1
    return currect/4000

# 得到最终正确率
currency_test1 = detect_anomalies(log_density1, threshold)
print("Currency in Test Data 1:", currency_test1)

currency_test2 = detect_anomalies(log_density2, threshold)
print("Currency in Test Data 2:", currency_test2)

# 绘图
plt.subplot(1, 2, 1)
plt.bar(range(1,4001),log_density1)
plt.title("data analyze")
plt.xlabel("samples")
plt.ylabel("scores")
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(range(1,4001),log_density2)
plt.title("data analyze")
plt.xlabel("samples")
plt.ylabel("scores")
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.legend()

plt.show()
