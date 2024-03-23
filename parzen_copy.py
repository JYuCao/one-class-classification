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

X = np.random.normal(loc=0, scale=1, size=(4000, 31))
print(X.shape)
log_density = parzen_estimator.score_samples(train_data)
log_density1 = parzen_estimator.score_samples(X)

threshold = np.percentile(sorted(log_density), 0)

# 绘图
plt.bar(range(1,4001),log_density1)
plt.title("data analyze")
plt.xlabel("samples")
plt.ylabel("scores")
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.legend()

plt.show()
