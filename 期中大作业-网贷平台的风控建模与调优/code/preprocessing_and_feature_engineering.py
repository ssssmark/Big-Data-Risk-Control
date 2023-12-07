import eda_and_data_cleaning
import re
import pandas as pd
master=eda_and_data_cleaning.master
userupdateinfo=eda_and_data_cleaning.userupdateinfo
numeric_cols = master.select_dtypes(include='number').columns
# 去掉空格
master['UserInfo_9'] = master['UserInfo_9'].apply(lambda x: x.strip())
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
def encodingregion(s):
    special = ['内蒙古自治区', '宁夏回族自治区', '广西壮族自治区', '新疆维吾尔自治区']
    ret = ['内蒙古', '宁夏', '广西', '新疆']
    for i in range(len(special)):
        if s == special[i]:
            return ret[i]
    regex1 = re.compile(r'.+省')
    regex2 = re.compile(r'.+市')
    if regex1.search(s):
        s = s[:-1]
        return s
    elif regex2.search(s):
        s = s[:-1]
        return s
    else:
        return s

print(master['UserInfo_2'].nunique())
print(master['UserInfo_4'].nunique())
print(master['UserInfo_7'].nunique())
print(master['UserInfo_19'].nunique())
print(master['UserInfo_8'].nunique())
print(master['UserInfo_20'].nunique())

master['UserInfo_2'] = master['UserInfo_2'].apply(lambda x: encodingstr(x))
master['UserInfo_4'] = master['UserInfo_4'].apply(lambda x: encodingstr(x))
master['UserInfo_7'] = master['UserInfo_7'].apply(lambda x: encodingregion(x))
master['UserInfo_19'] = master['UserInfo_19'].apply(lambda x: encodingregion(x))
master['UserInfo_8'] = master['UserInfo_8'].apply(lambda x: encodingstr(x))
master['UserInfo_20'] = master['UserInfo_20'].apply(lambda x: encodingstr(x))
userupdateinfo.to_csv('./userupdateinfo.csv', index=False, encoding='utf-8')

print(master.columns)

#%%
std_threshold = 3  # 阈值为均值的3倍标准差
for col in numeric_cols:
    if col != 'Idx' and col != 'target' and col != 'source' and col != 'ListingInfo':
        mean = master[col].mean()
        std = master[col].std()
        # 识别离群点
        outliers = master[abs(master[col] - mean) > std_threshold * std]
        # 用均值替换离群点
        master.loc[abs(master[col] - mean) > std_threshold * std, col] = mean

#%%
userupdateinfo.to_csv('./userupdateinfo.csv',index=False,encoding='utf-8')
#%%
## 借款日期离散化
# 把月、日、单独拎出来，放到3列中
master['month'] = pd.DatetimeIndex(master.ListingInfo).month
master['day']  = pd.DatetimeIndex(master.ListingInfo).day
master['day'].head()
master.drop(['ListingInfo'],axis=1,inplace=True)
master['target'] = master['target'].astype(str)
master.to_csv('./master.csv',index=False,encoding='utf-8')
#%%
# import matplotlib.pyplot as plt
#
# for col in categoric_cols:
#     plt.hist(master[col], bins=20)
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.show()
#%%
def get_city_tier(city):
    first_tier = ['北京', '上海', '广州', '深圳']
    new_first_tier = ['成都', '重庆', '杭州', '武汉', '西安', '郑州', '青岛', '长沙', '天津', '苏州', '南京', '东莞',
                      '沈阳', '合肥', '佛山']
    second_tier = ['昆明', '福州', '无锡', '厦门', '哈尔滨', '长春', '南昌', '济南', '宁波', '大连', '贵阳', '温州',
                   '石家庄', '泉州', '南宁', '金华', '常州', '珠海', '惠州', '嘉兴', '南通', '中山', '保定', '兰州',
                   '台州', '徐州', '太原', '绍兴', '烟台', '廊坊']
    third_tier = ['海口', '汕头', '潍坛', '扬州', '洛阳', '乌鲁木齐', '临沂', '唐山', '镇江', '盐城', '湖州', '赣州',
                  '漳州', '揭阳', '江门', '桂林', '邯郸', '泰州', '济宁', '呼和浩特', '咸阳', '芜湖', '三亚', '阜阳',
                  '淮安', '遵义', '银川', '衡阳', '上饶', '柳州', '淄博', '莆田', '绵阳', '湛江', '商丘', '宜昌',
                  '沧州', '连云港', '南阳', '蚌埠', '驻马店', '滁州', '邢台', '潮州', '秦皇岛', '肇庆', '荆州', '周口',
                  '马鞍山', '清远', '宿州', '威海', '九江', '新乡', '信阳', '襄阳', '岳阳', '安庆', '菏泽', '宜春',
                  '黄冈', '泰安', '宿迁', '株洲', '宁德', '鞍山', '南充', '六安', '大庆', '舟山']
    fourth_tier = ['常德', '渭南', '孝感', '丽水', '运城', '德州', '张家口', '鄂尔多斯', '阳江', '泸州', '丹东', '曲靖',
                   '乐山', '许昌', '湘潭', '晋中', '安阳', '齐齐哈尔', '北海', '宝鸡', '抚州', '景德镇', '延安', '三明',
                   '抚顺', '亳州', '日照', '西宁', '衢州', '拉萨', '淮北', '焦作', '平顶山', '滨州', '吉安', '濮阳',
                   '眉山', '池州', '荆门', '铜仁', '长治', '衡水', '铜陵', '承德', '达州', '邵阳', '德阳', '龙岩',
                   '南平', '淮南', '黄石', '营口', '东营', '吉林', '韶关', '枣庄', '包头', '怀化', '宣城', '临汾',
                   '聊城', '梅州', '盘锦', '锦州', '榆林', '玉林', '十堰', '汕尾', '咸宁', '宜宾', '永州', '益阳',
                   '黔南州', '黔东南', '恩施', '红河', '大理', '大同', '鄂州', '忻州', '吕梁', '黄山', '开封', '郴州',
                   '茂名', '漯河', '葫芦岛', '河源', '娄底', '延边']

    if city in first_tier:
        return 1
    elif city in new_first_tier:
        return 2
    elif city in second_tier:
        return 3
    elif city in third_tier:
        return 4
    elif city in fourth_tier:
        return 5
    elif city == '不详':
        return 6
    return None


city_col = ['UserInfo_2', 'UserInfo_4', 'UserInfo_8', 'UserInfo_20']
for col in city_col:
    master[col] = master[col].apply(get_city_tier)
    mode_value = master[col].mode()[0]
    master[col].fillna(mode_value, inplace=True)
    print(master[col].head(5))
#%%
# 是否为**省
my_alpha = 0.05

province_col = ['UserInfo_7', 'UserInfo_19']
province_info = pd.DataFrame(columns=['Province', 'y_sum', 'num', 'grade'])
master['target'] = pd.to_numeric(master['target'], errors='coerce')
co = master['UserInfo_7']
sum_y = master.groupby(co)['target'].sum().reset_index()

#%%
num = master.groupby(co).size().reset_index()
province_info['Province'] = master.groupby(co).groups.keys()
province_info['y_sum'] = sum_y['target'].astype(int)
province_info['num'] = num[0].astype(int)
province_info['grade'] = (province_info['y_sum'] / province_info['num'] * 1000).astype(int)
# province_info['grade'] = province_info['y_sum'] / province_info['num']
filtered_province_info = province_info[province_info['grade'] > 20]
print(filtered_province_info)
province_names = filtered_province_info['Province']
print(province_names)
print(master.shape)
# 遍历每个省份，如果 'UserInfo_7' 列的值等于该省份名称，对应列设置为1，否则为0
for col in province_col:
    col = master[col]
    for province in province_names:
        column_name = f'is_{province}'  # 根据省份名称创建列名
        master[column_name] = (col == province).astype(int)

print(master.shape)