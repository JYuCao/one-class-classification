import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

def scatter(transformed_data):
    # 绘制散点图,仅二维用
    plt.scatter(transformed_data[2000:, 0], transformed_data[2000:, 1])
    plt.scatter(transformed_data[:2000, 0], transformed_data[:2000, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter Plot')
    plt.show()

def load_data():
    # 加载数据
    train_data = np.genfromtxt('data/Train_data.csv', delimiter=',')
    test_data1 = np.genfromtxt('data/Test_data1.csv', delimiter=',')
    test_data2 = np.genfromtxt('data/Test_data2.csv', delimiter=',')
    return train_data, test_data1, test_data2

def normalized(data, mean, std):
    # 数据标准化
    normalized_data = (data - mean) / std
    return normalized_data

def spe(P, lambdas, x, k):
    sp = np.dot(np.dot(P, P.T), x.T)
    sr = np.dot(np.eye(31) - np.dot(P, P.T), x.T)
    # 计算T^2 和 阈值Ta
    T_2 = np.linalg.norm(sp, axis=0)
    Ta = k * (4000 - 1) * (4000 + 1) * f.ppf(0.99, k, 4000-k) / (4000 * (4000 - k))
    # 计算Q 和 阈值Qa
    Q = np.linalg.norm(sr, axis=0)
    theta1 = np.sum(lambdas[k+1:])
    theta2 = np.sum(np.square(lambdas[k+1:]))
    theta3 = np.sum(np.power(lambdas[k+1:], 3))
    h0 = 1 - 2 * theta1 * theta3 / (3 * theta2 * theta2)
    ca = 1.6449
    Qa = theta1 * np.power(ca * h0 * np.sqrt(2 * theta2) / theta1 + 1 + theta2 * h0 * (h0 - 1)/ np.power(theta1, 2), 1 / h0)

    return T_2, Ta, Q, Qa

def T_calculate_accuracy(T_2, Ta2, Q, Qa):
    T_TP = T_FP = T_TN = T_FN = 0
    Q_TP = Q_FP = Q_TN = Q_FN = 0
    for i in range(4000):
        if i < 2000:
            if T_2[i] < Ta2:
                T_TP += 1
            else:
                T_FN += 1
            if Q[i] < Qa:
                Q_TP += 1
            else:
                Q_FN += 1
        else:
            if T_2[i] > Ta2:
                T_TN += 1
            else:
                T_FP += 1
            if Q[i] > Qa:
                Q_TN += 1
            else:
                Q_FP += 1
    T_accuracy = (T_TP + T_TN) / 4000
    T_precision = T_TP / (T_TP + T_FP)
    T_recall = T_TP / (T_TP + T_FN)
    Q_accuracy = (Q_TP + Q_TN) / 4000
    Q_precision = Q_TP / (Q_TP + Q_FP)
    Q_recall = Q_TP / (Q_TP + Q_FN)
    print(f"# T: accuracy:{T_accuracy}, precision:{T_precision}, recall:{T_recall}")
    print(f"# Q: accuracy:{Q_accuracy}, precision:{Q_precision}, recall:{Q_recall}")


if __name__ == '__main__':
    # 加载数据
    train_data, test_data1, test_data2 = load_data()

    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0, ddof=1)

    # 归一化训练数据
    normalized_data_train = normalized(train_data, mean, std)

    # 计算协方差矩阵
    standardized_covariance_matrix = np.dot(normalized_data_train.T, normalized_data_train/(4000-1))

    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(standardized_covariance_matrix)

    sorted_eigenvalues = sorted(eigenvalues, reverse=True)

    # 按照特征值降序，对特征向量进行重新排序
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 选择前k个特征向量，得到降维矩阵
    sorted_eigenvalues = eigenvalues[sorted_indices]
    k = 1
    while np.sum(sorted_eigenvalues[:k]) / np.sum(eigenvalues) < 0.8:
        k += 1
    print(f"# k = {k}")

    diagonal_matrix = np.diag(sorted_eigenvalues[:k])
    selected_eigenvectors = sorted_eigenvectors[:, :k]

    # 归一化测试数据
    normalized_data_test1 = normalized(test_data1, mean, std)
    normalized_data_test2 = normalized(test_data2, mean, std)

    # 计算T^2 和 Q
    T_21, Ta21, Q1, Qa1 = spe(selected_eigenvectors, sorted_eigenvalues, normalized_data_test1, k)
    T_22, Ta22, Q2, Qa2 = spe(selected_eigenvectors, sorted_eigenvalues, normalized_data_test2, k)

    # 计算准确率
    print("# Test Data 1:")
    T_calculate_accuracy(T_21, Ta21, Q1, Qa1)
    print("# Test Data 2:")
    T_calculate_accuracy(T_22, Ta22, Q2, Qa2)

    # 绘制图像
    plt.subplot(2,2,1)
    plt.plot(np.r_[0.:4000], T_21, label='Test Data 1')
    plt.axhline(y=Ta21, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Data Index')
    plt.ylabel('T Value')
    plt.title('T Value Plot')
    plt.legend()
    plt.subplot(2,2,2)
    plt.plot(np.r_[0.:4000], T_22, label='Test Data 2')
    plt.axhline(y=Ta22, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Data Index')
    plt.ylabel('T Value')
    plt.title('T Value Plot')
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(np.r_[0.:4000], Q1, label='Test Data 1')
    plt.axhline(y=Qa1, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Data Index')
    plt.ylabel('Q Value')
    plt.title('Q Value Plot')
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(np.r_[0.:4000], Q2, label='Test Data 2')
    plt.axhline(y=Qa2, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Data Index')
    plt.ylabel('Q Value')
    plt.title('Q Value Plot')
    plt.legend()
    plt.show()
    
