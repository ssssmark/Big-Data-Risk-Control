import pandas as pd


class Solution:
    # 计算借据违约率
    def cal_pd(self,df):
        a=len(df.loc[df.loan_order>0,'client_no'])
        b=len(df.loc[df.loan_used_days>90,'client_no'])
        return b/a
    # 计算违约损失率
    def cal_lgd(self,df):
        a=df.loc[df.loan_used_days>110,'loan_amt'].sum()
        b=df.loc[df.loan_used_days>90,'loan_amt'].sum()
        return a/b
    # 计算风险敞口
    def cal_EaD(self,df):
        return df.loan_amt.sum()
    # 计算预期损失  Expected Loss= Probability of Default * Loss Given Default * Exposure at Default
    def cal_Expect_Loss(self,df):
        return self.cal_pd(df)*self.cal_lgd(df)*self.cal_EaD(df)
    # 计算实际损失
    def cal_Real_loss(self,df):
        return df.loc[df.loan_used_days>110,'loan_amt'].sum()
    def compute(self):
        # 读取两份EXCEL文件
        client = pd.read_excel("data_client_1.xlsx")
        client_hist = pd.read_excel("data_client_1_hist.xlsx")
        client1=client.loc[(client.credit_score>=160)&(client.credit_score<=190),'client_no']
        client2=client.loc[client.credit_score>190,'client_no']
        df1=client_hist.loc[client_hist.client_no.isin(client1)]
        df2=client_hist.loc[client_hist.client_no.isin(client2)]

        # WARNING! 以下内容请修改为你处理数据的代码！
        res1 = {'pd': self.cal_pd(df1),
                'ead': self.cal_EaD(df1),
                'exact_loss': self.cal_Real_loss(df1),
                'expected_loss': self.cal_Expect_Loss(df1),
                'lgd': self.cal_lgd(df1),
                'score_range': '160~190'}
        res2 = {'pd': self.cal_pd(df2),
                'ead': self.cal_EaD(df2),
                'exact_loss': self.cal_Real_loss(df2),
                'expected_loss': self.cal_Expect_Loss(df2),
                'lgd': self.cal_lgd(df2),
                'score_range': '190+'}
        return [res1, res2]

sln = Solution()
res = sln.compute()
print(res)
