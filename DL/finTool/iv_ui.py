import requests
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox



# --- 原始数据获取和计算函数 (保持不变) ---
# 请确保使用您本地带错误处理的版本

def direct_get_data(code: str = '159915', selday: str = '2025-11-28', today: str = '2025-12-01',
                    endDate: str = '0') -> pd.DataFrame:
    """
    从指定 API 获取期权数据。
    """
    url = "https://op.liaoy.cn/API_opiv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://op.liaoy.cn",
        "Referer": "https://op.liaoy.cn/iv"
    }

    data = {
        "Bd": code,
        "endDate": endDate,
        "selday": selday,
        "today": today
    }

    resp = None
    try:
        resp = requests.post(url, headers=headers, data=data, timeout=15)
        resp.raise_for_status()

        api_data = resp.json()

        if "iv" not in api_data or not api_data["iv"]:
            raise ValueError("API 返回数据中 'iv' 字段为空或不存在。")

        iv = api_data["iv"]

        dfs = iv[0: 6]
        cols = ['日期', '收盘价', 'HV20', '综合IV', '当月IV', '次月IV']
        df = pd.DataFrame(dfs)

        if df.empty:
            raise ValueError("API 返回数据为空，无法创建 DataFrame。")

        df = df.T
        df.columns = cols

        for col in ['HV20', '综合IV', '当月IV', '次月IV']:
            # 确保 IV 列是数值类型
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except requests.exceptions.HTTPError as e:
        error_detail = "无详细信息"
        if resp is not None:
            error_detail = f"服务器返回错误码 {resp.status_code}。\n详细信息 (可能为空): {resp.text}"
        messagebox.showerror("网络错误", f"请求 API 失败: {e}\n{error_detail}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        messagebox.showerror("网络错误", f"请求 API 失败: {e}")
        return pd.DataFrame()
    except ValueError as e:
        messagebox.showerror("数据错误", f"处理数据失败: {e}")
        return pd.DataFrame()
    except Exception as e:
        messagebox.showerror("未知错误", f"发生未知错误: {e}")
        return pd.DataFrame()


def cal_rank(df: pd.DataFrame, mode: str = '综合IV', iv: float = None, backsize: int = 250) -> float:
    """
    计算 IV 分位数。
    # (代码不变，使用您本地带错误处理的版本)
    """
    if df.empty or mode not in df.columns:
        raise ValueError(f"DataFrame 为空或不包含列 '{mode}'。")

    actual_backsize = min(backsize, len(df) - 1)
    if actual_backsize <= 0:
        raise ValueError("回溯窗口大小设置不合理，请检查数据量。")

    df_list = df.iloc[-(actual_backsize + 1): -1]

    if iv is None:
        iv_val = df.iloc[-1][mode]
    else:
        iv_val = iv

    if pd.isna(iv_val):
        raise ValueError(f"用于对比的最新 {mode} 值是 NaN。")

    rank = (df_list[mode] < iv_val).sum() / len(df_list)

    return rank


# --- Tkinter UI 应用程序 (增加最新日期显示) ---

class IVRankApp:
    def __init__(self, master):
        self.master = master
        master.title("期权 IV 分位数计算器")

        # 调整窗口大小以适应新字段
        master.geometry('450x420')
        master.resizable(False, False)

        # 定义字体
        self.FONT_INPUT = ('Helvetica', 12)
        self.FONT_LABEL = ('Helvetica', 12)
        self.FONT_RESULT = ('Helvetica', 16, 'bold')
        self.FONT_DATE = ('Helvetica', 10)

        # 默认值
        self.default_code = '510050'
        self.default_mode = '综合IV'
        self.default_backsize = 250
        self.default_today = '2025-12-01'
        self.default_selday = '2025-11-28'

        # 创建变量来存储用户输入和结果
        self.code_var = tk.StringVar(value=self.default_code)
        self.mode_var = tk.StringVar(value=self.default_mode)
        self.backsize_var = tk.StringVar(value=str(self.default_backsize))
        self.selday_var = tk.StringVar(value=self.default_selday)
        self.today_var = tk.StringVar(value=self.default_today)
        self.rank_result_var = tk.StringVar(value="等待计算...")
        self.iv_result_var = tk.StringVar(value="N/A")
        # 新增日期变量
        self.date_result_var = tk.StringVar(value="N/A")

        self._create_widgets(master)

    def _create_widgets(self, master):
        # 设置主框架
        main_frame = ttk.Frame(master, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)

        # --- 输入参数区 ---
        row_idx = 0

        # ... (标的代码、模式、回溯窗口、日期输入框代码不变) ...
        # 标的代码 (Code)
        ttk.Label(main_frame, text="标的代码 (Code):", font=self.FONT_LABEL).grid(row=row_idx, column=0, sticky=tk.W,
                                                                                  pady=5)
        ttk.Entry(main_frame, textvariable=self.code_var, font=self.FONT_INPUT).grid(row=row_idx, column=1,
                                                                                     sticky=(tk.W, tk.E), pady=5)
        row_idx += 1

        # IV 模式 (Mode)
        ttk.Label(main_frame, text="IV 模式 (Mode):", font=self.FONT_LABEL).grid(row=row_idx, column=0, sticky=tk.W,
                                                                                 pady=5)
        mode_options = ['综合IV', '当月IV', '次月IV', 'HV20']
        ttk.Combobox(main_frame, textvariable=self.mode_var, values=mode_options, font=self.FONT_INPUT,
                     state="readonly").grid(row=row_idx, column=1, sticky=(tk.W, tk.E), pady=5)
        row_idx += 1

        # 回溯窗口 (Backsize)
        ttk.Label(main_frame, text="回溯窗口 (Backsize):", font=self.FONT_LABEL).grid(row=row_idx, column=0,
                                                                                      sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.backsize_var, font=self.FONT_INPUT).grid(row=row_idx, column=1,
                                                                                         sticky=(tk.W, tk.E), pady=5)
        row_idx += 1

        # 对比日期 (SelDay)
        ttk.Label(main_frame, text="对比日期 (SelDay):", font=self.FONT_LABEL).grid(row=row_idx, column=0, sticky=tk.W,
                                                                                    pady=5)
        ttk.Entry(main_frame, textvariable=self.selday_var, font=self.FONT_INPUT).grid(row=row_idx, column=1,
                                                                                       sticky=(tk.W, tk.E), pady=5)
        row_idx += 1

        # 最新日期 (Today)
        ttk.Label(main_frame, text="最新日期 (Today):", font=self.FONT_LABEL).grid(row=row_idx, column=0, sticky=tk.W,
                                                                                   pady=5)
        ttk.Entry(main_frame, textvariable=self.today_var, font=self.FONT_INPUT).grid(row=row_idx, column=1,
                                                                                      sticky=(tk.W, tk.E), pady=5)
        row_idx += 1

        # --- 操作按钮区 ---
        ttk.Button(main_frame, text="开始计算 Rank & 获取最新 IV", command=self.calculate_rank, style='TButton').grid(
            row=row_idx, column=0, columnspan=2, pady=15, sticky=(tk.W, tk.E))
        row_idx += 1

        # --- 结果显示区 ---

        # 最新 IV 结果 (占两行)
        ttk.Label(main_frame, text="最新 IV 值:", font=self.FONT_LABEL).grid(row=row_idx, column=0, sticky=tk.W,
                                                                             pady=(5, 0))
        # IV 值显示
        ttk.Label(main_frame, textvariable=self.iv_result_var, font=self.FONT_RESULT, foreground='darkgreen').grid(
            row=row_idx, column=1, sticky=(tk.W), padx=0, pady=(5, 0))
        row_idx += 1

        # 日期显示 (在 IV 值下面)
        ttk.Label(main_frame, text="对应日期:", font=self.FONT_LABEL).grid(row=row_idx, column=0, sticky=tk.W,
                                                                           pady=(0, 10))
        ttk.Label(main_frame, textvariable=self.date_result_var, font=self.FONT_DATE, foreground='gray').grid(
            row=row_idx, column=1, sticky=(tk.W), padx=5, pady=(0, 10))
        row_idx += 1

        # Rank 结果
        ttk.Label(main_frame, text="IV Rank (分位数):", font=self.FONT_LABEL).grid(row=row_idx, column=0, sticky=tk.W,
                                                                                   pady=5)
        ttk.Label(main_frame, textvariable=self.rank_result_var, font=self.FONT_RESULT, foreground='blue').grid(
            row=row_idx, column=1, sticky=(tk.W, tk.E), pady=5)
        row_idx += 1

    def calculate_rank(self):
        """执行数据获取和 Rank 计算的逻辑"""
        code = self.code_var.get().strip()
        mode = self.mode_var.get()
        selday = self.selday_var.get().strip()
        today = self.today_var.get().strip()

        try:
            backsize = int(self.backsize_var.get().strip())
        except ValueError:
            messagebox.showerror("输入错误", "回溯窗口 (Backsize) 必须是一个整数。")
            return

        # 1. 重置并获取数据
        self.rank_result_var.set("正在从 API 获取数据...")
        self.iv_result_var.set("N/A")
        self.date_result_var.set("N/A")  # 重置日期
        self.master.update()

        df = direct_get_data(
            code=code,
            selday=selday,
            today=today
        )

        if df.empty:
            self.rank_result_var.set("获取失败")
            return

        # 2. 获取用于计算的最新 IV 值和日期
        try:
            # 最新数据是 DataFrame 的最后一行
            latest_row = df.iloc[-1]

            latest_iv = latest_row[mode]
            latest_date = latest_row['日期']

            if pd.isna(latest_iv):
                raise ValueError(f"最新数据中 {mode} 的 IV 值为 NaN，无法计算。")

            self.iv_result_var.set(f"{latest_iv:.4f}")
            self.date_result_var.set(f"({latest_date})")  # 设置日期显示

        except Exception as e:
            messagebox.showerror("数据错误", f"无法获取最新的 {mode} IV 值或日期: {e}")
            self.rank_result_var.set("计算失败")
            return

        # 3. 计算 Rank
        try:
            rank = cal_rank(df, mode=mode, backsize=backsize, iv=latest_iv)

            # 4. 显示 Rank 结果
            rank_percent = rank * 100
            self.rank_result_var.set(f"{rank_percent:.2f} %")

            messagebox.showinfo(
                "计算成功",
                f"最新 IV 值 ({mode}, {latest_date}): {latest_iv:.4f}\n"
                f"IV Rank 计算完成：{rank_percent:.2f}%"
            )

        except ValueError as e:
            messagebox.showerror("计算错误", str(e))
            self.rank_result_var.set("计算失败")
        except Exception as e:
            messagebox.showerror("未知错误", f"计算过程中发生未知错误: {e}")
            self.rank_result_var.set("计算失败")


# --- 运行应用程序 ---
if __name__ == '__main__':
    root = tk.Tk()
    style = ttk.Style(root)
    style.configure('TButton', font=('Helvetica', 12))

    app = IVRankApp(root)
    root.mainloop()