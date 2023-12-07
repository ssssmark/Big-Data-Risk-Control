import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import re
#%%
trainpath = '../data/train'
train_master = pd.read_csv(os.path.join(trainpath, 'Master_Training_Set.csv'), encoding='gbk')
train_userupdateinfo = pd.read_csv(os.path.join(trainpath, 'Userupdate_Info_Training_Set.csv'), encoding='gbk')
train_loginfo = pd.read_csv(os.path.join(trainpath, 'LogInfo_Training_Set.csv'), encoding='gbk')
train_master['source'] = 'train'
train_userupdateinfo['source'] = 'train'
train_loginfo['source'] = 'train'

testpath ='../data/test'
test_master = pd.read_csv(os.path.join(testpath, 'Master_Test_Set.csv'), encoding='gbk')
test_userupdateinfo = pd.read_csv(os.path.join(testpath, 'Userupdate_Info_Test_Set.csv'), encoding='gbk')
test_loginfo = pd.read_csv(os.path.join(testpath, 'LogInfo_Test_Set.csv'), encoding='gbk')
test_master['source'] = 'test'
test_userupdateinfo['source'] = 'test'
test_loginfo['source'] = 'test'
#%%
master =pd.concat([train_master, test_master], ignore_index=True)
userupdateinfo = pd.concat([train_userupdateinfo,test_userupdateinfo], ignore_index=True)
loginfo =pd.concat([train_loginfo,test_loginfo],ignore_index=True)


myfont = FontProperties(fname=r"./SIMHEI.TTF",size=12)
master.drop(['WeblogInfo_1' ,'WeblogInfo_3'],axis=1,inplace=True)
#%%
## 处理UserInfo_12缺失
print(master['UserInfo_12'].unique())
#fig = plt.figure()
#fig.set(alpha=0.2)
target_UserInfo_12_not = master.target[master.UserInfo_12.isnull()].value_counts()
target_UserInfo_12_ = master.target[master.UserInfo_12.notnull()].value_counts()
df_UserInfo_12 = pd.DataFrame({'missing': target_UserInfo_12_not, 'not_missing': target_UserInfo_12_})
df_UserInfo_12
df_UserInfo_12.plot(kind='bar', stacked=True)
plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
plt.xlabel(u'有无', fontproperties=myfont)
plt.ylabel(u'违约情况', fontproperties=myfont)
plt.show()
master.loc[(master.UserInfo_12.isnull(), 'UserInfo_12')] = 2.0
#train_master['UserInfo_11'].fillna(2.0)
#train_master['UserInfo_12'] =train_master['UserInfo_12'].astype(np.int32)
print(master['UserInfo_12'].dtypes)
print(master['UserInfo_12'].unique())
#%%
## 处理UserInfo_11缺失
master['UserInfo_11'].unique()
#fig = plt.figure()
#fig.set(alpha=0.2)
target_UserInfo_11_not = master.target[master.UserInfo_11.isnull()].value_counts()
target_UserInfo_11_ = master.target[master.UserInfo_11.notnull()].value_counts()
df_UserInfo_11 = pd.DataFrame({'no_have': target_UserInfo_11_not, 'have': target_UserInfo_11_})
df_UserInfo_11.plot(kind='bar', stacked=True)
plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
plt.xlabel(u'有无', fontproperties=myfont)
plt.ylabel(u'违约情况', fontproperties=myfont)
plt.show()
#train_master['UserInfo_11'] =train_master['UserInfo_11'].astype(str)
master.loc[(master.UserInfo_11.isnull(), 'UserInfo_11')] = 2.0
master['UserInfo_11'].unique()
#%%
## 处理UserInfo_13缺失
master['UserInfo_13'].unique()
#fig = plt.figure()
#fig.set(alpha=0.2)
target_UserInfo_13_not = master.target[master.UserInfo_13.isnull()].value_counts()
target_UserInfo_13_ = master.target[master.UserInfo_13.notnull()].value_counts()
df_UserInfo_13 = pd.DataFrame({'no_have': target_UserInfo_13_not, 'have': target_UserInfo_13_})
df_UserInfo_13
df_UserInfo_13.plot(kind='bar', stacked=True)
plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
plt.xlabel(u'有无', fontproperties=myfont)
plt.ylabel(u'违约情况', fontproperties=myfont)
plt.show()
#train_master['UserInfo_13'] =train_master['UserInfo_13'].astype(str)
master.loc[(master.UserInfo_13.isnull(), 'UserInfo_13')] = 2.0
master['UserInfo_13'].unique()
#%%
## 处理WeblogInfo_20 缺失
print(master['WeblogInfo_20'].unique())
#fig = plt.figure()
#fig.set(alpha=0.2)
target_WeblogInfo_20_not = master.target[master.WeblogInfo_20.isnull()].value_counts()
target_WeblogInfo_20_ = master.target[master.WeblogInfo_20.notnull()].value_counts()
df_WeblogInfo_20 = pd.DataFrame({'no_have': target_WeblogInfo_20_not, 'have': target_WeblogInfo_20_})
df_WeblogInfo_20
df_WeblogInfo_20.plot(kind='bar', stacked=True)
plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
plt.xlabel(u'有无', fontproperties=myfont)
plt.ylabel(u'违约情况', fontproperties=myfont)
plt.show()
#train_master['WeblogInfo_20'] =train_master['WeblogInfo_20'].astype(str)
master.loc[(master.WeblogInfo_20.isnull(), 'WeblogInfo_20')] = u'不详'
master['WeblogInfo_20'].unique()
#%%
print(master['WeblogInfo_19'].unique())
#fig = plt.figure()
#fig.set(alpha=0.2)
target_WeblogInfo_19_not = master.target[master.WeblogInfo_19.isnull()].value_counts()
target_WeblogInfo_19_ = master.target[master.WeblogInfo_19.notnull()].value_counts()
df_WeblogInfo_19 = pd.DataFrame({'no_have': target_WeblogInfo_19_not, 'have': target_WeblogInfo_19_})
df_WeblogInfo_19

df_WeblogInfo_19.plot(kind='bar', stacked=True)
plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
plt.xlabel(u'有无', fontproperties=myfont)
plt.ylabel(u'违约情况', fontproperties=myfont)
plt.show()

#train_master['WeblogInfo_19'] =train_master['WeblogInfo_19'].astype(str)
master.loc[(master.WeblogInfo_19.isnull(), 'WeblogInfo_19')] = u'不详'
master['WeblogInfo_19'].unique()
#%%
## 处理WeblogInfo_21 缺失
print(master['WeblogInfo_21'].unique())
#fig = plt.figure()
#fig.set(alpha=0.2)
target_WeblogInfo_21_not = master.target[master.WeblogInfo_21.isnull()].value_counts()
target_WeblogInfo_21_ = master.target[master.WeblogInfo_21.notnull()].value_counts()
df_WeblogInfo_21 = pd.DataFrame({'no_have': target_WeblogInfo_21_not, 'have': target_WeblogInfo_21_})
df_WeblogInfo_21

df_WeblogInfo_21.plot(kind='bar', stacked=True)
plt.title(u'有无这个特征对结果的影响', fontproperties=myfont)
plt.xlabel(u'有无', fontproperties=myfont)
plt.ylabel(u'违约情况', fontproperties=myfont)
plt.show()
#train_master['WeblogInfo_21'] =train_master['WeblogInfo_21'].astype(str)
master.loc[(master.WeblogInfo_21.isnull(), 'WeblogInfo_21')] = '0'
master['WeblogInfo_21'].unique()
#%%
## 其余缺失值很少的就用均值或众数填充
len(master['UserInfo_2'].value_counts())  ## 城市地理位置
len(master['UserInfo_4'].value_counts())  ## 城市地理位置
len(master['UserInfo_8'].value_counts())  ## 城市地理位置
len(master['UserInfo_9'].unique())  ## 城市地理位置
len(master['UserInfo_20'].value_counts())  ## 城市地理位置
len(master['UserInfo_7'].unique())  ## 省份地理位置
len(master['UserInfo_19'].unique())
## 省份地理位置
# 如果选择以0填充，下述部分就维持现状，如果选择中位数/众数填充，就把下述的部分注释掉
master.loc[(master.UserInfo_2.isnull(), 'UserInfo_2')] = '0'
master.loc[(master.UserInfo_4.isnull(), 'UserInfo_4')] = '0'
master.loc[(master.UserInfo_8.isnull(), 'UserInfo_8')] = '0'
master.loc[(master.UserInfo_9.isnull(), 'UserInfo_9')] = '0'
master.loc[(master.UserInfo_20.isnull(), 'UserInfo_20')] = '0'
master.loc[(master.UserInfo_7.isnull(), 'UserInfo_7')] = '0'
master.loc[(master.UserInfo_19.isnull(), 'UserInfo_19')] = '0'
#%%
## 用众数填充缺失值
categoric_cols = ['UserInfo_1' ,'UserInfo_2' ,'UserInfo_3' ,'UserInfo_4' , 'UserInfo_5' ,'UserInfo_6','UserInfo_7','UserInfo_8','UserInfo_9','UserInfo_11','UserInfo_12','UserInfo_13','UserInfo_14','UserInfo_15','UserInfo_16','UserInfo_17','UserInfo_19','UserInfo_20','UserInfo_21','UserInfo_22','UserInfo_23','UserInfo_24','Education_Info1','Education_Info2','Education_Info3','Education_Info4','Education_Info5','Education_Info6','Education_Info7','Education_Info8','WeblogInfo_19','WeblogInfo_20','WeblogInfo_21','SocialNetwork_1','SocialNetwork_2','SocialNetwork_7','SocialNetwork_12']
# for col in categoric_cols:
#     mode_cols = master[col].mode()[0]
#     master.loc[(master[col].isnull() , col)] = mode_cols
numeric_cols = master.select_dtypes(include='number').columns
categorical_cols = master.select_dtypes(exclude='number').columns
## 用均值填充缺失值
# non_numeric_cols = ['UserInfo_2', 'UserInfo_4', 'UserInfo_7', 'UserInfo_8','UserInfo_9','UserInfo_19','UserInfo_20','UserInfo_22','UserInfo_23','UserInfo_24','Education_Info2','Education_Info3','Education_Info4','Education_Info6','Education_Info7','Education_Info8','WeblogInfo_19','WeblogInfo_20','WeblogInfo_21']
# numeric_cols=[i for i in categoric_cols if i not in non_numeric_cols]
for col in master.columns:
    if col not in categoric_cols and col !=u'Idx' and col !=u'target'and col !='source' and col !='ListingInfo':
        mean_cols = master[col].mean()
        master.loc[(master[col].isnull() , col)] = mean_cols

#%%
## 剔除标准差几乎为零的特征项
feature_std = master[numeric_cols].std().sort_values(ascending=True)
print(feature_std.head(20))
columns_to_drop = feature_std[(feature_std < 0.1) & (feature_std.index != 'target')].index
master.drop(columns_to_drop, axis=1, inplace=True)
numeric_cols=[i for i in numeric_cols if i not in columns_to_drop]
master['Idx'] = master['Idx'].astype(np.int32)
