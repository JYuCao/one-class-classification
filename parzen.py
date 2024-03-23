from sklearn.neighbors import KernelDensity
import numpy as np

train_data = np.genfromtxt('data/Train_data.csv', delimiter=',')
test_data1 = np.genfromtxt('data/Test_data1.csv', delimiter=',')
test_data2 = np.genfromtxt('data/Test_data2.csv', delimiter=',')

bandwidth = 1.0  # 窗宽度参数
parzen_estimator = KernelDensity(bandwidth=bandwidth)
parzen_estimator.fit(train_data)

log_density = parzen_estimator.score_samples(train_data)
log_density1 = parzen_estimator.score_samples(test_data1)
log_density2 = parzen_estimator.score_samples(test_data2)

threshold = np.percentile(log_density, 5)  # 以5%分位数作为阈值

def detect_anomalies(log_density, threshold):
    currect = 0.0
    for i in range(0, 4000):
        if i < 2000:
            if log_density[i] >= threshold:
                currect += 1
        else:
            if log_density[i] < threshold:
                currect += 1
    return currect/4000

currency_test1 = detect_anomalies(log_density1, threshold)
print("Currency in Test Data 1:", currency_test1)

currency_test2 = detect_anomalies(log_density2, threshold)
print("Currency in Test Data 2:", currency_test2)
