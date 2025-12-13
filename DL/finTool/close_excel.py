import os
import time

def kill_excel_force():
    """
    使用系统命令强制结束 Excel 进程。
    /F : 强制终止进程 (Force)
    /IM : 指定映像名称 (Image Name)
    >nul 2>nul : 屏蔽命令行的输出结果，保持控制台清爽
    """
    # 执行命令，返回值 0 表示成功杀掉了进程，128 表示没找到进程
    ret = os.system("taskkill /F /IM excel.exe >nul 2>nul")
    
    # 如果你想看日志，可以把下面这两行取消注释
    # if ret == 0:
    #     print(f"[{time.strftime('%H:%M:%S')}] 已强制关闭 Excel 进程")

if __name__ == "__main__":
    print("[系统监控中] 每 0.1 秒检查并强制关闭 Excel...")
    
    try:
        while True:
            kill_excel_force()
            time.sleep(0.1)  # 保持 0.1 秒的间隔
    except KeyboardInterrupt:
        print("\n监控已停止。")