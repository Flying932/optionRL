import pyautogui as pg
import time
import pandas as pd
import os
import glob
from windowAccount import windowAccount

# --- 核心工具函数：安全找图 ---
def safe_locate(image_path, confidence=0.9, intent="未知步骤"):
    """
    尝试定位图片。
    如果找不到，直接 raise Exception，带上具体是哪一步失败的信息。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"本地图片文件不存在: {image_path}")

    try:
        pos = pg.locateCenterOnScreen(image_path, confidence=confidence)
        
        # 针对某些旧版本 pyautogui 返回 None 而不报错的情况做兼容
        if pos is None:
            raise pg.ImageNotFoundException
            
        return pos
        
    except pg.ImageNotFoundException:
        # 这里抛出的异常会被最外层的 try...except 捕获
        # 我们把 'intent' (意图) 加进去，这样你就知道是哪一步挂了
        img_name = os.path.basename(image_path)
        error_msg = f"定位失败 -> 步骤: [{intent}] | 图片: {img_name}"
        raise Exception(error_msg)


def get_excel_files(file_path):
    files_xlsx = glob.glob(os.path.join(file_path, "*.xlsx"))
    files_xls = glob.glob(os.path.join(file_path, "*.xls"))
    files = files_xlsx + files_xls
    files = [
        f for f in files
        if not os.path.basename(f).startswith(".")
           and os.path.isfile(f)
    ]
    return files


def gen_option_data(optionCode: str, k: str='30', market: str='sh'):
    print(f"--------开始执行数据下载, optionCode = {optionCode}---------")
    time.sleep(3)

    # 1. 输入框
    search_img = f'./miniQMT/DL/finTool/pics/search_button.png'
    # 如果找不到，这里直接报错，跳到外层 except
    pos = safe_locate(search_img, confidence=0.9, intent="寻找输入框") 
    pg.click(pos)
    pg.write(optionCode, interval=0.05)
    
    # 2. 搜索按钮
    find_img = f'./miniQMT/DL/finTool/pics/find.png'
    pos = safe_locate(find_img, confidence=0.9, intent="点击搜索按钮")
    pg.click(pos)
    
    time.sleep(8)

    # 3. 市场选择
    if market == 'sh':
        market_img = f'./miniQMT/DL/finTool/pics/sh.png'
        market_name = "上海市场(sh)"
    else:
        market_img = f'./miniQMT/DL/finTool/pics/sz.png'
        market_name = "深圳市场(sz)"
    
    pos = safe_locate(market_img, confidence=0.85, intent=f"选择{market_name}")
    pg.click(pos)

    time.sleep(3)

    # 4. 时间周期
    time_interval = f'./miniQMT/DL/finTool/pics/{k}.png'
    pos = safe_locate(time_interval, confidence=0.9, intent=f"选择时间周期 {k}")
    pg.click(pos)
    
    time.sleep(1)

    # 5. 导出数据菜单
    gen_data = f'./miniQMT/DL/finTool/pics/gendata2.png'
    pos = safe_locate(gen_data, confidence=0.80, intent="点击导出数据菜单")
    pg.click(pos)
    
    time.sleep(0.5)

    # 6. 确认导出
    gen = f'./miniQMT/DL/finTool/pics/gen.png'
    pos = safe_locate(gen, confidence=0.9, intent="点击确认导出")
    pg.click(pos)
    
    time.sleep(1.5)  

    # 7. 检查是否有“无数据”弹窗 
    # (特殊情况：这个找不到是好事，不能用 safe_locate 报错)
    no_data = f'./miniQMT/DL/finTool/pics/no_data.png'
    try:
        # 这里单独 try，因为找不到是正常的
        pos = pg.locateCenterOnScreen(no_data, confidence=0.9)
    except pg.ImageNotFoundException:
        pos = None
    
    if pos:
        print(f"[Warning] 期权{optionCode}没有数据")
        # 即使没数据，流程可能也需要继续走去点“保存/取消”或者关闭窗口
        # 如果没数据会弹窗阻挡后续操作，这里可能需要额外的处理逻辑

    # 8. 保存/取消按钮
    # save_and_cancel = f'./miniQMT/DL/finTool/pics/save_K.png'
    # save_and_cancel = f'./miniQMT/DL/finTool/pics/save_and_cancel.png'
    # pos = safe_locate(save_and_cancel, confidence=0.85, intent="点击保存位置")
    
    # x, y = pos
    # new_x = x - (1795 - 1739)
    # new_y = y - 2
    # # new_x = x
    # # new_y = y
    # pg.click(new_x, new_y)
    pg.press('enter')
    
    time.sleep(2.0)
    
    # 关闭窗口 (Ctrl+W)
    pg.hotkey('ctrl', 'w')
    time.sleep(0.5)
    pg.hotkey('ctrl', 'w')


def gen_data_from_choice(option_list: list, market: str='sh', start_idx: int=0):
    len_df = len(option_list)
    for index, optionCode in enumerate(option_list):
        optionCode = str(optionCode).strip() 

        if index < start_idx:
            continue
        
        try:
            gen_option_data(optionCode, '30', market)
            print(f"[Info] 处理{optionCode}成功, 进度: {index+1} / {len_df}")
        
        except Exception as e:
            # --- 这里的 e 包含了 safe_locate 抛出的具体位置信息 ---
            print(f"[Info] 处理{optionCode}失败, 进度: {index+1} / {len_df}")
            print(f"   >>> 错误详情: {e}") 
            # 这里的 e 会显示类似： "定位失败 -> 步骤: [点击搜索按钮] | 图片: find.png"
            break


def get_exist_option_list(file_path: str='./miniQMT/datasets/realInfo'):
    files = get_excel_files(file_path)
    option_list = []
    for f in files:
        if not f.endswith('_30分钟线数据.xlsx'):
            continue
        first = f.find('_')
        second = f.find('_', first + 1)

        option = f[first + 1: second]
        if len(option) == 8:
            option_list.append(option)  
    
    return option_list

def get_all_option_list(benchmark: str='510050'):
    file_path = f'./miniQMT/datasets/optionInfo/{benchmark}期权合约数据.xlsx'
    df = pd.read_excel(file_path)
    target_expire = ['2024' + str(i) for i in range(10, 13)]
    
    cond_a = (df['到期日'].astype(str).str[:6].isin(target_expire))
    cond_b = (df['合约乘数'] == 10000)

    filtered_df = df[cond_a & cond_b]
    option_list = filtered_df['期权代码'].astype(str)

    return option_list.to_list()

def get_need_option_list(benchmark: str='510050'):
    all_list = get_all_option_list(benchmark)
    exist_list = get_exist_option_list()

    res = []
    for item in all_list:
        if item in exist_list:
            continue
        res.append(item)
    
    return res

def get_condition_list(target: str='510050'):
    account = windowAccount(100000, fee=1.3, period='30m', stockList=[target])

    option_list = account.get_option_list(target, '202412', 'call')
    option_list.extend(account.get_option_list(target, '202412', 'put'))
    option_list.extend(account.get_option_list(target, '202411', 'call'))
    option_list.extend(account.get_option_list(target, '202411', 'put'))
    option_list.extend(account.get_option_list(target, '202410', 'call'))
    option_list.extend(account.get_option_list(target, '202410', 'put'))
    option_list.extend(account.get_option_list(target, '202409', 'call'))
    option_list.extend(account.get_option_list(target, '202409', 'put'))
    option_list.extend(account.get_option_list(target, '202408', 'call'))
    option_list.extend(account.get_option_list(target, '202408', 'put'))
    option_list.extend(account.get_option_list(target, '202407', 'call'))
    option_list.extend(account.get_option_list(target, '202407', 'put'))
    option_list.extend(account.get_option_list(target, '202406', 'call'))
    option_list.extend(account.get_option_list(target, '202406', 'put'))

    option_list.extend(account.get_option_list(target, '202405', 'call'))
    option_list.extend(account.get_option_list(target, '202405', 'put'))
    
    option_list.extend(account.get_option_list(target, '202404', 'call'))
    option_list.extend(account.get_option_list(target, '202404', 'put'))
    
    option_list.extend(account.get_option_list(target, '202403', 'call'))
    option_list.extend(account.get_option_list(target, '202403', 'put'))
    
    option_list.extend(account.get_option_list(target, '202402', 'call'))
    option_list.extend(account.get_option_list(target, '202402', 'put'))
    
    option_list.extend(account.get_option_list(target, '202401', 'call'))
    option_list.extend(account.get_option_list(target, '202401', 'put'))
    

    return option_list

if __name__ == '__main__':
    exist_list = get_exist_option_list()
    lis = get_condition_list()

    target_lis = []
    for op in lis:
        if op in exist_list:
            continue 
        target_lis.append(op)

    print(f'total_length = {len(target_lis)}')
    gen_data_from_choice(target_lis, 'sh', start_idx=0)

    print(0 / 0)