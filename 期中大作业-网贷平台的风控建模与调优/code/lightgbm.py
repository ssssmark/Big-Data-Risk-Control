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

import lightgbm as lgb
from sklearn import metrics

def train_lightgbm(train_data, y_train, params=None, num_boost_round=100):
    if params is None:
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_leaves': 40,
            'learning_rate': 0.02,
            'feature_fraction': 0.2,
            'max_depth':11,
            'reg_alpha':0.4,
            'reg_lambda':0.4,
            'n_estimators':500,
        }

    lgb_train = lgb.Dataset(train_data, label=y_train)
    model = lgb.train(params, lgb_train, num_boost_round=num_boost_round)

    # 预测概率
    y_pred = model.predict(train_data)

    # 计算AUC
    auc_score = metrics.roc_auc_score(y_train, y_pred)

    return model, auc_score

# 调用函数进行训练
lgbmodel, auc = train_lightgbm(train_data, y_train)
print(f'AUC Score: {auc}')

#%%
lgbmodel.predict(test_data)
#%%
yprob_lgb = lgbmodel.predict(test_data)
result_lgb = pd.DataFrame({'ID': test_data.values[:, 0], 'Prediction': yprob_lgb})
result_lgb
result_lgb.to_csv('result_lgb.csv',encoding='utf-8')