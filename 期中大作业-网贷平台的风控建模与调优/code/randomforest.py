import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_validate, GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as preprocessing
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
#%%
master = pd.read_csv('./master.csv',encoding='utf-8')
userupdateinfo = pd.read_csv('./userupdate_df.csv',encoding='utf-8')
#train_loginfo = pd.read_csv('./loginfo_df.csv',encoding='utf-8')
#%%
all = pd.merge(master, userupdateinfo, how='left', on='Idx')
#train_all = pd.merge(train_all, train_loginfo, how='left', on='Idx')

#%%
import warnings
warnings.filterwarnings("ignore")
all['target'].fillna(-1, inplace=True)
all['Idx'] = all['Idx'].astype(np.int64)
all['target'] = all['target'].astype(np.int64)
all = pd.get_dummies(all)

#%%
## 填充缺失值
features_to_fillna = all.columns
# 用平均值来填充
all[features_to_fillna] = all[features_to_fillna].fillna(all[features_to_fillna].mean())
## 对数值型特征进行scaling
import warnings
from sklearn.preprocessing import StandardScaler
# 创建一个标准化器
scaler = StandardScaler()
# 选择需要缩放的数值型特征列，例如选择所有数值型特征
numeric_features = all.select_dtypes(include=['number']).columns
numeric_features = [feature for feature in numeric_features if feature != 'Idx' and feature !='target']
# 对数值型特征进行缩放
all[numeric_features] = scaler.fit_transform(all[numeric_features])
warnings.filterwarnings("ignore")
#%%
# 拆分数据集回训练集和测试集
train_data = all[all['source_test'] == False]
test_data = all[all['source_test'] == True]
train_data.drop(['source_test','source_train'],axis=1,inplace=True)
test_data.drop(['source_test','source_train'],axis=1,inplace=True)

#%%
y_train=train_data['target']
test_data.drop(['target'],axis=1,inplace=True)
train_data.drop(['target'],axis=1,inplace=True)


from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")
rf0 = RandomForestClassifier(oob_score=True, random_state=42, min_samples_split=5, min_samples_leaf=2)
rf0.fit(train_data.values[:, 1:], y_train)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(train_data.values[:, 1:])[:, 1]
print('AUC Score(Train): %f' % metrics.roc_auc_score(y_train, y_predprob))

#%%
rf_pred=rf0.predict_proba(train_data.values[:, 1:])[:,1].astype(float)
result_rf = pd.DataFrame({'ID': test_data.values[:, 0], 'Prediction': rf_pred})
result_rf
result_rf.to_csv('result_rf.csv',encoding='utf-8')
#%%
import warnings

warnings.filterwarnings("ignore")
lr = LogisticRegression(tol=1e-6)
parameters = {'penalty': ('l1', 'l2'), 'C': [0.001, 0.01, 0.1, 1, 10, 100]}
clf_lr = GridSearchCV(lr, parameters, cv=3)
print('开始训练')
clf_lr.fit(train_data.values[:, 1:], y_train)
print('模型训练结束')
clf_lr
import warnings

warnings.filterwarnings("ignore")
clf_lr_accuracy = clf_lr.score(train_data.values[:, 1:], y_train)
print(clf_lr_accuracy)
clf_lr.cv_results_, clf_lr.best_params_, clf_lr.best_score_