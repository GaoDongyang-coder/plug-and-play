from PyQt5.QtWidgets import QApplication
import sys
from login import LoginWindow  # 只需要登录界面

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 直接启动登录界面，登录成功后会直接进入分析界面
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())