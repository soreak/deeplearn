# -*- coding: utf-8 -*-
from QtFusion.config import QF_Config
import os  # 导入os模块，用于处理操作系统相关的功能
from sys import argv, exit  # 导入用于处理命令行参数和退出程序的模块
from PySide6.QtWidgets import QApplication  # 导入用于创建应用程序的类
QF_Config.set_verbose(False)

os.environ["QT_FONT_DPI"] = "96"  # 设置环境变量，修复高DPI和缩放超过100%时的问题

if __name__ == '__main__':  # 确保该模块被直接运行时才执行以下代码
    from System_noLogin import RecMainWindow  # 导入自定义的主窗口类
    app = QApplication(argv)  # 创建一个QApplication对象，它是所有Qt应用程序的基础
    win = RecMainWindow()  # 创建一个FlowerMainWindow对象，它是我们自定义的主窗口类的实例
    win.showTime()  # 调用主窗口的showTime方法，显示当前时间
    exit(app.exec())  # 进入应用程序的主循环，并在退出时返回状态码