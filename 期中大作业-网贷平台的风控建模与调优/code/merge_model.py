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


from sklearn.model_selection import train_test_split

# 先将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train_data, y_train, test_size=0.2, random_state=42)
#%%
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
rf = RandomForestClassifier(oob_score=True, random_state=42, min_samples_split=5, min_samples_leaf=2)
xgb = XGBClassifier(random_state=42,
                    learning_rate=0.04,
                    n_estimators=200,
                    max_depth=7,
                    min_child_weight=1.0,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
lgbm = LGBMClassifier(random_state=42,
                      objective='binary',
                      boosting_type='gbdt',
                      num_leaves= 40,
                      learning_rate= 0.02,
                      feature_fraction= 0.2,
                      max_depth=11,
                      reg_alpha=0.4,
                      reg_lambda=0.4,
                      n_estimators=500)
catboost = CatBoostClassifier(random_state=42,
                              iterations=1000,
                              learning_rate=0.04,
                              depth=7,
                              loss_function='Logloss',
                              custom_metric=['AUC'],
                              l2_leaf_reg=2,)
# 训练基本学习器
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train)
catboost.fit(X_train, y_train)

# 创建投票分类器（Voting Classifier）作为融合模型
ensemble_model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)], voting='soft')

# 训练融合模型
ensemble_model.fit(X_train, y_train)

# 使用融合模型进行预测
test_predictions = ensemble_model.predict(X_test)

# 评估融合模型的性能
accuracy = accuracy_score(y_test, test_predictions)
roc_auc = roc_auc_score(y_test, ensemble_model.predict_proba(X_test)[:, 1])

print("准确率:", accuracy)
print("AUC 分数:", roc_auc)

# 使用融合模型对测试数据进行预测
test_predictions = ensemble_model.predict(test_data)
#%%
from sklearn.ensemble import StackingClassifier
rf = RandomForestClassifier(oob_score=True, random_state=42, min_samples_split=5, min_samples_leaf=2)
xgb = XGBClassifier(random_state=42,
                    learning_rate=0.04,
                    n_estimators=200,
                    max_depth=7,
                    min_child_weight=1.0,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
lgbm = LGBMClassifier(random_state=42,
                      objective='binary',
                      boosting_type='gbdt',
                      num_leaves= 40,
                      learning_rate= 0.02,
                      feature_fraction= 0.2,
                      max_depth=11,
                      reg_alpha=0.4,
                      reg_lambda=0.4,
                      n_estimators=500)
catboost = CatBoostClassifier(random_state=42,
                              iterations=1000,
                              learning_rate=0.04,
                              depth=7,
                              loss_function='Logloss',
                              l2_leaf_reg=2,)
logistic=LogisticRegression()
# 训练基本学习器
# rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train)
catboost.fit(X_train, y_train)
logistic.fit(X_train, y_train)
# 创建底层模型的列表
base_models = [('xgb', xgb), ('lgbm', lgbm),('ctb',catboost)]

# 创建顶层模型
meta_learner = logistic

# 创建 Stacking 模型
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_learner)

# 训练 Stacking 模型
stacking_model.fit(X_train, y_train)

# 使用 Stacking 模型进行预测
stacking_predictions = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, stacking_predictions)
roc_auc = roc_auc_score(y_test, stacking_model.predict_proba(X_test)[:, 1])

print("准确率:", accuracy)
print("AUC 分数:", roc_auc)

#%%
# 使用 Stacking 模型对测试数据进行预测
merge_predictions = stacking_model.predict_proba(test_data)[:,1]
result_merge = pd.DataFrame({'ID': test_data.values[:, 0], 'Prediction': merge_predictions})
result_merge
result_merge.to_csv('result_merge.csv', encoding='utf-8')
#%%
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
rf = RandomForestClassifier(oob_score=True, random_state=42, min_samples_split=5, min_samples_leaf=2)
xgb = XGBClassifier(random_state=42,
                    learning_rate=0.04,
                    n_estimators=200,
                    max_depth=7,
                    min_child_weight=1.0,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
lgbm = LGBMClassifier(random_state=42,
                      objective='binary',
                      boosting_type='gbdt',
                      num_leaves= 40,
                      learning_rate= 0.02,
                      feature_fraction= 0.2,
                      max_depth=11,
                      reg_alpha=0.4,
                      reg_lambda=0.4,
                      n_estimators=500)
catboost = CatBoostClassifier(random_state=42,
                              iterations=1000,
                              learning_rate=0.04,
                              depth=7,
                              loss_function='Logloss',
                              l2_leaf_reg=2,)
logistic=LogisticRegression()

# 训练基本学习器
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train)
catboost.fit(X_train, y_train)

# 使用基本学习器进行预测
xgb_predictions = xgb.predict(X_test)
lgbm_predictions = lgbm.predict(X_test)
catboost_predictions = catboost.predict(X_test)
rf_predictions = rf.predict(X_test)

# 创建 Blending 的训练集
blending_train_data = pd.DataFrame({'XGB': xgb_predictions, 'LGBM': lgbm_predictions, 'CatBoost': catboost_predictions, 'RF': rf_predictions})

# 创建顶层模型
meta_learner = catboost

# 训练顶层模型
meta_learner.fit(blending_train_data, y_test)

# 使用顶层模型进行预测
blending_predictions = meta_learner.predict(blending_train_data)

# 评估 Blending 模型的性能
accuracy = accuracy_score(y_test, blending_predictions)
roc_auc = roc_auc_score(y_test, meta_learner.predict_proba(blending_train_data)[:, 1])

print("Blending 模型准确率:", accuracy)
print("Blending 模型 AUC 分数:", roc_auc)
