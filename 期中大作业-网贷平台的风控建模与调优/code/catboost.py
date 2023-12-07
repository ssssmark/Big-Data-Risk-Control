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


from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def train_and_evaluate_catboost(train_data, y_train):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(train_data.values[:, 1:], y_train, test_size=0.2, random_state=42)

    # 创建 CatBoost 分类器模型
    catboost_model = CatBoostClassifier(iterations=1000,
                                        learning_rate=0.04,
                                        depth=7,
                                        loss_function='Logloss',
                                        custom_metric=['AUC'],
                                        random_state=42,
                                        l2_leaf_reg=2,

                                        )

    # 创建 CatBoost 数据池
    train_pool = Pool(X_train, label=y_train)
    test_pool = Pool(X_test, label=y_test)

    # 训练 CatBoost 模型
    catboost_model.fit(train_pool, eval_set=test_pool,plot=True)

    # 进行预测
    y_pred = catboost_model.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, catboost_model.predict_proba(X_test)[:, 1])

    print("准确率:", accuracy)
    print("AUC 分数:", roc_auc)

    # 获取特征重要性
    feature_importance = catboost_model.get_feature_importance(data=train_pool, type='LossFunctionChange')

    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance)
    plt.yticks(range(len(feature_importance)), train_data.columns[1:])
    plt.xlabel('Feature Importance')
    plt.title('CatBoost Feature Importance')
    plt.show()
    return catboost_model
ctb_model = train_and_evaluate_catboost(train_data, y_train)

ctb_pred =ctb_model.predict_proba(test_data.values[:, 1:])[:, 1].astype(float)
result_ctb = pd.DataFrame({'ID': test_data.values[:, 0], 'Prediction': ctb_pred})
result_ctb
result_ctb.to_csv('result_ctb.csv', encoding='utf-8')