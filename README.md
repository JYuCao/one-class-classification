# 针对特定单分类数据集的训练方法
## 说明
在`./data`中有一个训练集和两个测试集，每个数据集有4000个31维样本，且均无标签。

其中，`Train_data`为训练集，训练集的4000个样本都为正常数据。`Test_data1`和`Test_data2`都为测试集，测试集的前2000个样本为正常数据，后2000个样本为异常数据。`Test_data1`的数据容易检测，`Test_data2`的数据极难检测。

要求 __只使用训练集__ 进行训练，并将拟合模型用于测试测试集中的异常数据。
## 使用
所需环境：anaconda/miniconda
1. 创建虚拟环境
```
conda create -n oc python=3.9
```
2. 进入虚拟环境
```
conda activate oc
```
3. 安装依赖
```
pip install -r requirements.txt
```
4. 运行
```
python mixture.py
python norm.py
python svm.py
python parzen.py
```
## 实践过的方法以及说明
### norm.py
利用训练集构建一个多元正态分布，并由此计算出两个测试集的概率密度函数。将概率密度函数和一个阈值进行比较，以确定异常数据。
### norm_md.py
和 norm.py 类似，唯一的区别在于确定异常数据由pdf(概率密度函数)变成了马氏距离。
### mixture.py
利用训练集构建一个高斯混合模型，用其估计测试集中的异常数据。
### svm.py
利用训练集构建一个svm模型，用其估计测试集中的异常数据。
### parzen.py
使用parzen窗法构建一个训练集的模型，用其估计测试集中的异常数据。
### 各实践的效果
parzen窗法对于两个测试集都达到了100%的分辨效果，其他方法皆对 test2 无效，对 test1 有接近 100% 的准确率。

## 部分参考
单分类算法：One Class SVM:
https://blog.csdn.net/qq_19446965/article/details/118742147

SVM的核函数如何选取？
https://www.zhihu.com/question/21883548

sklearn中文文档：
https://sklearn.apachecn.org/