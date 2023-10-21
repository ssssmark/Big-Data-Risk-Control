import pandas as pd
import numpy as np


class Solution:
    # 计算各类还款周期的首逾率、本金损失期望和资金占用比例
    def cal(self, df2, term_num):
        _df2 = df2.copy()
        # 是否违约
        _df2["if_default"] = (_df2.periods > 0) & (_df2.if_repaid == 0) & (_df2.if_measurable == 1)
        # 每笔还款含本金
        _df2["principle"] = _df2.installment_loan_amount / _df2.periods
        # 每笔还款含利息
        _df2["interest"] = _df2.annunity - _df2.principle

        # 初始化
        df = np.zeros(term_num, dtype=float)
        ead = np.zeros(term_num, dtype=float)
        expected_loss = np.zeros(term_num, dtype=float)
        occupation = 0

        # 遍历各还款期数
        for i in range(1, term_num + 1):
            df_term = _df2.loc[(_df2.term == i) & (_df2.if_measurable == 1)]
            df[i - 1] = df_term.if_default.sum() / len(df_term)
            ead[i - 1] = df_term.principle.sum()
            lgd = 0.4
            expected_loss[i - 1] = df[i - 1] * ead[i - 1] * lgd
            occupation+=i/term_num

        # 首逾率
        fpd = len(_df2.loc[(_df2.if_default == 1) & (_df2.term == 1) & (_df2.periods == term_num)]) / len(_df2.loc[(_df2.term == 1) & (_df2.periods == term_num)])
        # 本金损失期望
        loss = expected_loss.sum()
        # 资金占用比例
        occu = occupation/term_num

        return fpd, loss, occu

    def compute(self):
        # 读取两份EXCEL文件
        df1 = pd.read_excel("data_client_2.xlsx")
        df2 = pd.read_excel("data_client_2_hist.xlsx")
        # WARNING! 以下内容请修改为你处理数据的代码！
        fpd_3, loss_3, occu_3 = self.cal(df2.loc[df2.periods == 3], 3)
        fpd_6, loss_6, occu_6 = self.cal(df2.loc[df2.periods == 6], 6)
        fpd_9, loss_9, occu_9 = self.cal(df2.loc[df2.periods == 9], 9)
        fpd_12, loss_12, occu_12 = self.cal(df2.loc[df2.periods == 12], 12)
        period_3 = {
            'expected_loss': loss_3,
            'occupation': occu_3,
            'fpd': fpd_3,
            'periods': '3'
        }
        period_6 = {
            'expected_loss': loss_6,
            'occupation': occu_6,
            'fpd': fpd_6,
            'periods': '6'
        }
        period_9 = {
            'expected_loss': loss_9,
            'occupation': occu_9,
            'fpd': fpd_9,
            'periods': '9'
        }
        period_12 = {
            'expected_loss': loss_12,
            'occupation': occu_12,
            'fpd': fpd_12,
            'periods': '12'
        }
        return [period_3, period_6, period_9, period_12]
sln = Solution()
res = sln.compute()
print(res)