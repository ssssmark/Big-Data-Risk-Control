from collections import defaultdict
import datetime as dt
import pandas as pd

##  userupdateinfo表
userupdate_info_number = defaultdict(list)  ### 用户信息更新的次数
userupdate_info_category = defaultdict(set)  ###用户信息更新的种类数
userupdate_info_times = defaultdict(list)  ### 用户分几次更新了
userupdate_info_date = defaultdict(list)  #### 用户借款成交与信息更新时间跨度
with open('./userupdateinfo.csv', 'r') as f:
    f.readline()
    for line in f.readlines():
        cols = line.strip().split(",")  ### cols 是list结果
        userupdate_info_date[cols[0]].append(cols[1])
        userupdate_info_number[cols[0]].append(cols[2])
        userupdate_info_category[cols[0]].add(cols[2])
        userupdate_info_times[cols[0]].append(cols[3])
    print(u'提取信息完成')



#%%
userupdate_info_number_ = defaultdict(int)  ### 用户信息更新的次数
userupdate_info_category_ = defaultdict(int)  ### 用户信息更新的种类数
userupdate_info_times_ = defaultdict(int)  ### 用户分几次更新了
userupdate_info_date_ = defaultdict(int)  #### 用户借款成交与信息更新时间跨度
for key in userupdate_info_date.keys():
    userupdate_info_times_[key] = len(set(userupdate_info_times[key]))
    delta_date = dt.datetime.strptime(userupdate_info_date[key][0], '%Y/%m/%d') - dt.datetime.strptime(
        list(set(userupdate_info_times[key]))[0], '%Y/%m/%d')
    userupdate_info_date_[key] = abs(delta_date.days)
    userupdate_info_number_[key] = len(userupdate_info_number[key])
    userupdate_info_category_[key] = len(userupdate_info_category[key])

print('信息处理完成')

#%%
## 建立一个DataFrame
Idx_ = list(userupdate_info_date_.keys())  #### list
numbers_ = list(userupdate_info_number_.values())
categorys_ = list(userupdate_info_category_.values())
times_ = list(userupdate_info_times_.values())
dates_ = list(userupdate_info_date_.values())
userupdate_df = pd.DataFrame(
    {'Idx': Idx_, 'numbers': numbers_, 'categorys': categorys_, 'times': times_, 'dates': dates_})
userupdate_df.head()
userupdate_df.to_csv('./userupdate_df.csv', index=False, encoding='utf-8')