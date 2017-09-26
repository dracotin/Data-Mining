# coding: utf-8
# �߼��ع� �Զ���ģ
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR


filename = '../data/bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:, :8].as_matrix()
y = data.iloc[:, 8].as_matrix()


rlr = RLR(selection_threshold=0.5)  # 建立模型
rlr.fit(x, y)  # 训练
rlr.get_support()  # 获取特征筛选结果
print('通过随机逻辑回归模型筛选结束')
print('有效特征：%s' % ','.join(data.columns[rlr.get_support(indices=True)]))
x = data[data.columns[rlr.get_support(indices=True)]].as_matrix()  # 筛选好特征

lr = LR()
lr.fit(x, y)
print(u'逻辑回归模型训练结束')
print(u'模型平均正确率：%s' % lr.score(x, y))  # 本例 ：81.4%
