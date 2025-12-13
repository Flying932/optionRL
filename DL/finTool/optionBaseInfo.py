"""
    本代码实现期权相关的基本的信息处理
"""

import pandas as pd

# 期权基本信息类
class optionBaseInfo:
    def __init__(self, stockCodeList, filepath: str=f'./miniQMT/datasets/optionInfo'):
        # 数据路径
        self.filepath = filepath

        # 已经被读取的ETF期权list, 里面存的是ETF代码, 形如510050.SH
        self.read_code_list = []

        # 期权信息, 从ETF代码映射到期权列表字典
        self.optioncode_info = {}

        # 标的 -> 期权代码列表
        self.target_to_list = {}
        for stock in stockCodeList:
            self.target_to_list[stock] = []

        for stockCode in stockCodeList:
            self.read(stockCode)
        

    # 读取ETF期权
    def read(self, stockCode: str='510050'):
        if stockCode in self.read_code_list:
            return
        
        self.read_code_list.append(stockCode)

        df = pd.read_excel(f'{self.filepath}/{stockCode}期权合约数据.xlsx')
        for row in df.itertuples(index=False, name=None):
            optionCode, openDate, expireDate, op_type, mul, strike, margin = row

            optionCode = str(optionCode)
            openDate = str(openDate)
            expireDate = str(expireDate)
            
            self.optioncode_info[optionCode] = {
                '发行日': openDate,
                '到期日': expireDate,
                '期权类型': op_type,
                '标的': stockCode,
                '合约乘数': mul,
                '行权价': strike,
                '保证金': margin
            }
            
            # 标的代码到期权列表的映射
            self.target_to_list[stockCode].append(optionCode)
    def get_option_list(self, targetCode: str):
        if targetCode in self.target_to_list:
            return self.target_to_list[targetCode]

        return []

    # 期权信息获取
    def get_openDate(self, optionCode: str):
        if optionCode in self.optioncode_info:
            return self.optioncode_info[optionCode]['发行日']
            
    def get_expireDate(self, optionCode: str):
        if optionCode in self.optioncode_info:
            return self.optioncode_info[optionCode]['到期日']
            
    def get_optionType(self, optionCode: str):
        if optionCode in self.optioncode_info:
            return self.optioncode_info[optionCode]['期权类型']
    
    def get_stockCode(self, optionCode: str):
        if optionCode in self.optioncode_info:
            return self.optioncode_info[optionCode]['标的']

    def get_multiplier(self, optionCode: str):
        if optionCode in self.optioncode_info:
            return self.optioncode_info[optionCode]['合约乘数']  

    def get_strikePrice(self, optionCode: str):
        if optionCode in self.optioncode_info:
            return self.optioncode_info[optionCode]['行权价']  
    
    def get_margin(self, optionCode: str):
        if optionCode in self.optioncode_info:
            return self.optioncode_info[optionCode]['保证金']  

    # (标的, 到期月, 期权类型) -> 期权代码列表
    def find_options_by_stock_and_expiry(self, stockCode: str, expire: str, opt_type: str | None = None) -> list[str]:
        # 确保该标的已读取
        if stockCode not in self.read_code_list:
            self.read(stockCode)

        # 规范到期日字符串：保留前8位数字
        s = str(expire).strip()
        if len(s) >= 6:
            s = s[:6]
        if not (len(s) == 6 and s.isdigit()):
            # 到期日格式不对，返回空
            return []

        # 过滤
        if opt_type is None:
            return [code for code, info in self.optioncode_info.items()
                    if info.get('标的') == stockCode and info.get('到期日')[0: 6] == s]
        else:
            return [code for code, info in self.optioncode_info.items()
                    if info.get('标的') == stockCode
                    and info.get('到期日')[0: 6] == s
                    and str(info.get('期权类型')).upper() == str(opt_type).upper()]
