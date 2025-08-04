import cv2
import numpy as np

class LineAnnotator:
    def __init__(self, image):
        self.image = image.copy()
        self.original = image.copy()
        self.temp_image = image.copy()  # 用于临时显示
        self.points = []
        self.lines = []
        self.current_point = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 记录第一个点
            self.current_point = (x, y)
            self.points.append((x, y))
            # 在点击位置画一个小圆
            cv2.circle(self.temp_image, (x, y), 3, (0, 0, 255), -1)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.current_point:
            # 实时显示临时线段
            self.temp_image = self.image.copy()
            cv2.line(self.temp_image, self.current_point, (x, y), (0, 255, 0), 2)
            
        elif event == cv2.EVENT_LBUTTONUP and self.current_point:
            # 完成当前线段
            end_point = (x, y)
            self.points.append(end_point)
            self.lines.append([self.current_point, end_point])
            
            # 在图像上绘制永久线段
            cv2.line(self.image, self.current_point, end_point, (0, 255, 0), 2)
            self.temp_image = self.image.copy()
            
            # 计算并显示线段信息
            length = np.sqrt((end_point[0]-self.current_point[0])**2 + 
                           (end_point[1]-self.current_point[1])**2)
            angle = np.arctan2(end_point[1]-self.current_point[1], 
                             end_point[0]-self.current_point[0]) * 180 / np.pi
            print(f"线段 {len(self.lines)}:")
            print(f"  起点: ({self.current_point[0]}, {self.current_point[1]})")
            print(f"  终点: ({end_point[0]}, {end_point[1]})")
            print(f"  长度: {length:.2f} 像素")
            print(f"  角度: {angle:.2f} 度")
            print()
            
            self.current_point = None

def annotate_court_lines(video_path):
    """
    手动标注场地线段
    参数:
        video_path: 视频文件路径
    返回:
        lines: 标注的线段列表
    """
    cap = cv2.VideoCapture(video_path)
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频")
        return
    
    # 创建标注器
    annotator = LineAnnotator(frame)
    cv2.namedWindow('Line Annotation')
    cv2.setMouseCallback('Line Annotation', annotator.mouse_callback)
    
    print("使用说明：")
    print("1. 点击起点和终点来画直线")
    print("2. 按 'r' 重置")
    print("3. 按 'q' 退出")
    print("4. 按 'z' 撤销上一条线")
    
    while True:
        cv2.imshow('Line Annotation', annotator.temp_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 重置图像
            annotator.image = annotator.original.copy()
            annotator.temp_image = annotator.original.copy()
            annotator.points = []
            annotator.lines = []
            print("已重置")
        elif key == ord('z') and annotator.lines:
            # 撤销上一条线
            annotator.lines.pop()
            annotator.points.pop()
            annotator.points.pop()
            # 重绘所有线
            annotator.image = annotator.original.copy()
            annotator.temp_image = annotator.original.copy()
            for line in annotator.lines:
                cv2.line(annotator.image, line[0], line[1], (0, 255, 0), 2)
            annotator.temp_image = annotator.image.copy()
            print("已撤销上一条线")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return annotator.lines

if __name__ == "__main__":
    video_path = r"C:\Users\GDY\Desktop\remote\volleyballdata\new-ui\mp4\MVI_6778.mp4"
    lines = annotate_court_lines(video_path)