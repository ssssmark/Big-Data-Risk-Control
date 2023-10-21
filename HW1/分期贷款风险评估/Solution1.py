import numpy as np
import pandas as pd
# 读取两份EXCEL文件

class Solution():
    def cal(self,_df1, _df2):
        # 借款客户
        used = _df1.loc[_df1.if_approved & _df1.if_used, "client_no"]
        # 违约客户
        default = _df2.loc[(_df2.periods>0) & (_df2.if_repaid==0) & _df2.if_measurable, "client_no"].unique()
        # 风险敞口
        ead = _df2.loc[(_df2.periods>0) & _df2.if_measurable, "annunity"].sum()

        # 违约率
        df_rate = len(default) / len(used)

        return df_rate, ead
    def compute(self):
        df1 = pd.read_excel("data_client_2.xlsx")
        df2 = pd.read_excel("data_client_2_hist.xlsx")
        df_hight_school = df1[df1.education == "high_school_degree"]
        df_bachelor_degree = df1[df1.education == "bachelor_degree"]
        df_master_degree = df1[df1.education == "master_degree"]

        df_high = df2.loc[df2.client_no.isin(df_hight_school.client_no)]
        df_bachelor = df2.loc[df2.client_no.isin(df_bachelor_degree.client_no)]
        df_master =df2.loc[df2.client_no.isin(df_master_degree.client_no)]
        pd_1, ead_1 = self.cal(df_hight_school, df_high)
        pd_2, ead_2 = self.cal(df_bachelor_degree,df_bachelor)
        pd_3, ead_3 = self.cal(df_master_degree,df_master)
        high_school_degree = {
            'df_rate': pd_1,
            'ead': ead_1,
            'education': 'high_school_degree'
        }
        bachelor_degree = {
        'df_rate': pd_2,
        'ead': ead_2,
        'education': 'bachelor_degree'
        }
        master_degree = {
        'df_rate': pd_3,
        'ead': ead_3,
        'education': 'master_degree'
        }
        return [high_school_degree, bachelor_degree, master_degree]