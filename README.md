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

## 部分参考
单分类算法：One Class SVM:
https://blog.csdn.net/qq_19446965/article/details/118742147

SVM的核函数如何选取？
https://www.zhihu.com/question/21883548

sklearn中文文档：
https://sklearn.apachecn.org/