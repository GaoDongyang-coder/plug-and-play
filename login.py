from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QLineEdit, QMessageBox, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QIcon, QFont
import json
import os

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("排球视频分析系统")
        self.setFixedSize(1200, 700)
        self.setup_ui()
        self.users = self.load_users()
        
    def setup_ui(self):
        # 设置背景图片 - 使用PyQt5支持的方式
        pixmap = QPixmap("img/back.png")
        if not pixmap.isNull():
            # 缩放背景图片以适应窗口
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette = QPalette()
            palette.setBrush(QPalette.Background, QBrush(scaled_pixmap))
            self.setPalette(palette)
        else:
            # 如果图片加载失败，使用明亮的渐变背景
            self.setStyleSheet("""
                QWidget {
                    background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(240, 248, 255, 255),
                        stop:0.3 rgba(173, 216, 230, 255),
                        stop:0.6 rgba(135, 206, 235, 255),
                        stop:1 rgba(100, 149, 237, 255));
                }
            """)
            
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建一个容器来包含标题和登录模块，使它们垂直居中
        center_container = QWidget()
        center_container.setStyleSheet("background: transparent;")
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(50)  # 标题和登录模块之间的间距
        
        # 系统标题 - 居中显示
        title_label = QLabel("排球视频分析系统")
        title_label.setStyleSheet("""
            QLabel {
                color: white;  
                font-size: 48px;
                font-weight: bold;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        
        # 创建登录模块容器
        login_container = QWidget()
        login_container.setStyleSheet("background: transparent;")
        login_container_layout = QHBoxLayout(login_container)
        login_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # 左侧留空用于平衡布局
        left_spacer = QWidget()
        left_spacer.setStyleSheet("background: transparent;")
        
        # 右侧留空用于平衡布局
        right_spacer = QWidget()
        right_spacer.setStyleSheet("background: transparent;")
        
        # 登录框
        login_frame = QFrame()
        login_frame.setFixedSize(380, 450)
        login_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 220);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 100);
            }
        """)
        
        login_layout = QVBoxLayout(login_frame)
        login_layout.setSpacing(25)
        login_layout.setContentsMargins(40, 40, 40, 40)
        
        # 登录标题
        login_title = QLabel("账号登录")
        login_title.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        login_title.setAlignment(Qt.AlignCenter)
        
        # 用户名输入框
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入账号名")
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 15px;
                border: 1px solid #DDDDDD;
                border-radius: 8px;
                background-color: #F8F8F8;
                color: #333333;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
                background-color: white;
            }
        """)
        
        # 创建密码输入区域的容器
        password_container = QWidget()
        password_layout = QHBoxLayout(password_container)
        password_layout.setContentsMargins(0, 0, 0, 0)
        password_layout.setSpacing(0)
        
        # 密码输入框
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 15px;
                border: 1px solid #DDDDDD;
                border-radius: 8px;
                background-color: #F8F8F8;
                color: #333333;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
                background-color: white;
            }
        """)
        
        # 密码显示切换按钮
        self.toggle_password_button = QPushButton()
        self.toggle_password_button.setFixedSize(30, 30)
        self.toggle_password_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                margin-left: -35px;
                color: #666666;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.1);
                border-radius: 15px;
            }
        """)
        self.toggle_password_button.setText("👁")
        self.toggle_password_button.clicked.connect(self.toggle_password_visibility)
        
        # 将密码输入框和切换按钮添加到容器中
        password_layout.addWidget(self.password_input)
        password_layout.addWidget(self.toggle_password_button)
        
        # 登录按钮
        self.login_button = QPushButton("登录")
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF5252;
            }
            QPushButton:pressed {
                background-color: #E53935;
            }
        """)
        self.login_button.clicked.connect(self.login)
        
        # 注册按钮
        self.register_button = QPushButton("注册账号")
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #666666;
                border: none;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover {
                color: #FF6B6B;
            }
        """)
        self.register_button.clicked.connect(self.register)
        
        # 添加所有控件到登录布局
        login_layout.addWidget(login_title)
        login_layout.addWidget(self.username_input)
        login_layout.addWidget(password_container)
        login_layout.addWidget(self.login_button)
        login_layout.addWidget(self.register_button)
        login_layout.addStretch()
        
        # 将登录框添加到登录容器的水平布局中，使其居中
        login_container_layout.addWidget(left_spacer, 1)
        login_container_layout.addWidget(login_frame, 0)
        login_container_layout.addWidget(right_spacer, 1)
        
        # 将标题和登录模块添加到垂直居中容器
        center_layout.addWidget(title_label)
        center_layout.addWidget(login_container)
        
        # 在主布局中将整个中心容器垂直居中
        main_layout.addStretch(1)
        main_layout.addWidget(center_container)
        main_layout.addStretch(1)

    def load_users(self):
        """加载用户数据"""
        try:
            if os.path.exists("users.json"):
                with open("users.json", "r") as f:
                    return json.load(f)
            return {}
        except:
            return {}
            
    def save_users(self):
        """保存用户数据"""
        with open("users.json", "w") as f:
            json.dump(self.users, f)
            
    def login(self):
        """登录处理"""
        username = self.username_input.text()
        password = self.password_input.text()
        
        if not username or not password:
            QMessageBox.warning(self, "警告", "请输入用户名和密码")
            return
            
        # 检查固定的用户名和密码
        if username == "admin" and password == "admin":
            # 直接导入并打开分析界面，而不是主菜单
            from guifinal12 import VolleyballAnalyzer
            self.analyzer_window = VolleyballAnalyzer()
            self.analyzer_window.show()
            self.close()
        else:
            QMessageBox.warning(self, "错误", "用户名或密码错误")

    def register(self):
        """注册处理"""
        username = self.username_input.text()
        password = self.password_input.text()
        
        if not username or not password:
            QMessageBox.warning(self, "警告", "请输入用户名和密码")
            return
            
        if username in self.users:
            QMessageBox.warning(self, "错误", "用户名已存在")
            return
            
        self.users[username] = password
        self.save_users()
        QMessageBox.information(self, "成功", "注册成功")

    def toggle_password_visibility(self):
        """切换密码显示/隐藏状态"""
        if self.password_input.echoMode() == QLineEdit.Password:
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.toggle_password_button.setText("🙈")  # 显示密码时用"看不见"的emoji
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_button.setText("👁")   # 隐藏密码时用"眼睛"的emoji

    def resizeEvent(self, event):
        """处理窗口大小调整事件"""
        super().resizeEvent(event)
        # 重新设置背景图片以适应新的窗口大小
        pixmap = QPixmap("img/back.png")
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette = QPalette()
            palette.setBrush(QPalette.Background, QBrush(scaled_pixmap))
            self.setPalette(palette)