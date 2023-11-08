import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import re

def cleandata():
    trainpath = './data/train'
    train_master = pd.read_csv(os.path.join(trainpath, 'Master_Training_Set.csv'), encoding='gbk')
    train_userupdateinfo = pd.read_csv(os.path.join(trainpath, 'Userupdate_Info_Training_Set.csv'), encoding='gbk')
    train_loginfo = pd.read_csv(os.path.join(trainpath, 'LogInfo_Training_Set.csv'), encoding='gbk')
    train_master['source'] = 'train'
    train_userupdateinfo['source'] = 'train'
    train_loginfo['source'] = 'train'

    testpath ='./data/test'
    test_master = pd.read_csv(os.path.join(testpath, 'Master_Test_Set.csv'), encoding='gbk')
    test_userupdateinfo = pd.read_csv(os.path.join(testpath, 'Userupdate_Info_Test_Set.csv'), encoding='gbk')
    test_loginfo = pd.read_csv(os.path.join(testpath, 'LogInfo_Test_Set.csv'), encoding='gbk')
    test_master['source'] = 'test'
    test_userupdateinfo['source'] = 'test'
    test_loginfo['source'] = 'test'

    master = pd.merge(train_master,test_master)
    userupdateinfo = pd.merge(train_userupdateinfo,test_userupdateinfo)
    loginfo =pd.merge(train_loginfo,test_loginfo)

    myfont = FontProperties(fname=r"./SIMHEI.TTF",size=12)
    master.drop(['WeblogInfo_1' ,'WeblogInfo_3'],axis=1,inplace=True)
    master.loc[(master.UserInfo_12.isnull() , 'UserInfo_12')] = 2.0
    master.loc[(master.UserInfo_11.isnull() , 'UserInfo_11')] = 2.0
    master.loc[(master.UserInfo_13.isnull() , 'UserInfo_13')] = 2.0
    master.loc[(master.WeblogInfo_20.isnull() , 'WeblogInfo_20')] = u'不详'
    master.loc[(master.WeblogInfo_19.isnull() , 'WeblogInfo_19')] = u'不详'
    master.loc[(master.WeblogInfo_21.isnull() , 'WeblogInfo_21')] = '0'

    ## 用众数填充缺失值
    categoric_cols = ['UserInfo_1' ,'UserInfo_2' ,'UserInfo_3' ,'UserInfo_4' , 'UserInfo_5' ,'UserInfo_6','UserInfo_7','UserInfo_8','UserInfo_9','UserInfo_11','UserInfo_12','UserInfo_13','UserInfo_14','UserInfo_15','UserInfo_16','UserInfo_17','UserInfo_19','UserInfo_20','UserInfo_21','UserInfo_22','UserInfo_23','UserInfo_24','Education_Info1','Education_Info2','Education_Info3','Education_Info4','Education_Info5','Education_Info6','Education_Info7','Education_Info8','WeblogInfo_19','WeblogInfo_20','WeblogInfo_21','SocialNetwork_1','SocialNetwork_2','SocialNetwork_7','SocialNetwork_12']
    # for col in categoric_cols:
    #     mode_cols = master[col].mode()[0]
    #     master.loc[(master[col].isnull() , col)] = mode_cols 
    
    ## 用均值填充缺失值 
    numeric_cols = []
    for col in master.columns:
        if col not in categoric_cols and col !=u'Idx' and col !=u'target' and col !='ListingInfo':
            mean_cols = master[col].mean()
            master.loc[(master[col].isnull() , col)] = mean_cols
    # 去掉空格
    master['UserInfo_9'] = train_master['UserInfo_9'].apply(lambda x: x.strip())
    ## 去掉大小写
    userupdateinfo['UserupdateInfo1'] =userupdateinfo['UserupdateInfo1'].apply(lambda x:x.lower())
    ## 将UserInfo_8中城市名归一化
    def encodingstr(s):
        regex = re.compile(r'.+市')
        if regex.search(s):
            s = s[:-1]
            return s
        else:
            return s
    master['UserInfo_8'] =master['UserInfo_8'].apply(lambda x: encodingstr(x))
    userupdateinfo.to_csv('./userupdateinfo.csv',index=False,encoding='utf-8')

cleandata()