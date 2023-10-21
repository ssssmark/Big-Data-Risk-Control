import numpy as np
import pandas as pd


class Solution():
    def compute(self):
        client = pd.read_excel('data_client_1.xlsx')
        client_hist=pd.read_excel('data_client_1_hist.xlsx')
        length = len(client)
        approve = client[client['if_approved'] == True]
        # 计算通过率
        aprrove_rate = len(approve) / length
        used=client[(client['if_approved']==True)&(client['if_used']==True)]
        # 转化率=获得审批的且有贷款记录的人/所有获批人数
        transfer_rate=len(used)/len(approve)
        # 计算所有授信金额
        credits_approve=0
        for i in approve['credit_approved']:
            credits_approve += i
        # 计算客户违约率：借贷天数超过90天且仍未还款的客户比例
        default_clients = client_hist.loc[client_hist.loan_repaid_date.isna() & (client_hist.loan_used_days > 90), 'client_no'].nunique()
        default_rate = default_clients / client['if_used'].sum()
        return aprrove_rate,transfer_rate,credits_approve,default_rate







