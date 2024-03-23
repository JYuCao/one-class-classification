import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 加载数据
train_data = np.genfromtxt('data/Train_data.csv', delimiter=',')
test_data1 = np.genfromtxt('data/Test_data1.csv', delimiter=',')
test_data2 = np.genfromtxt('data/Test_data2.csv', delimiter=',')

# 训练GMM模型
gmm = GaussianMixture(n_components=2)  # 假设有两个组件
gmm.fit(train_data)

# 利用GMM进行异常检测
def detect_anomalies(data, gmm_model):
    log_probs = gmm_model.score_samples(data)
    currect = 0.0
    for i in range(0, 4000):
        if i < 2000:
            if log_probs[i] >= threshold:
                currect += 1
        else:
            if log_probs[i] < threshold:
                currect += 1
    return currect/4000

# 定义阈值（可以根据实际情况调整）
train_scores = gmm.score_samples(train_data)
threshold = train_scores.mean() - 3 * train_scores.std()

# 对测试数据1进行异常检测
currency_test1 = detect_anomalies(test_data1, gmm)
print("Currency in Test Data 1:", currency_test1)

# 对测试数据2进行异常检测
currency_test2 = detect_anomalies(test_data2, gmm)
print("Currency in Test Data 2:", currency_test2)

plt.bar(range(1,4001),gmm.score_samples(test_data1))
plt.title("data analyze")
plt.xlabel("samples")
plt.ylabel("scores")
plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()
