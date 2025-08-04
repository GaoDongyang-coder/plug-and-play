"""
检测配置模块 - 包含防误检机制的配置和函数
"""

# 置信度阈值配置
CONFIDENCE_THRESHOLD = 0.9  # 目标检测置信度阈值

# 动作冷却时间配置（秒）
ACTION_COOLDOWN = 1.0  # 同一球员同一动作的冷却时间

def filter_detections(detect_results):
    """
    过滤目标检测结果，只保留置信度高于阈值的结果
    
    参数:
        detect_results: YOLO模型的目标检测结果
        
    返回:
        detect_boxes: 过滤后的边界框坐标
        detect_classes: 过滤后的类别索引
        detect_confidences: 过滤后的置信度
    """
    detect_boxes = []
    detect_classes = []
    detect_confidences = []
    
    if detect_results[0].boxes is not None:
        # 获取所有检测结果
        all_boxes = detect_results[0].boxes.xyxy.cpu().numpy()
        all_classes = detect_results[0].boxes.cls.cpu().numpy()
        all_confidences = detect_results[0].boxes.conf.cpu().numpy()
        
        # 过滤置信度低于阈值的结果
        for i, conf in enumerate(all_confidences):
            if conf >= CONFIDENCE_THRESHOLD:
                detect_boxes.append(all_boxes[i])
                detect_classes.append(all_classes[i])
                detect_confidences.append(conf)
        
        # 转换为numpy数组
        if detect_boxes:
            import numpy as np
            detect_boxes = np.array(detect_boxes)
            detect_classes = np.array(detect_classes)
            detect_confidences = np.array(detect_confidences)
    
    return detect_boxes, detect_classes, detect_confidences

class ActionCooldownManager:
    """
    动作冷却管理器 - 管理每个球员的动作冷却时间
    """
    def __init__(self, cooldown_time=ACTION_COOLDOWN):
        """
        初始化动作冷却管理器
        
        参数:
            cooldown_time: 动作冷却时间（秒）
        """
        self.cooldown_time = cooldown_time
        # 结构: {player_id: {action_label: timestamp, ...}, ...}
        self.player_last_actions = {}
    
    def is_action_in_cooldown(self, player_id, action_label, current_time):
        """
        检查指定球员的指定动作是否在冷却时间内
        
        参数:
            player_id: 球员ID
            action_label: 动作标签
            current_time: 当前时间戳
            
        返回:
            bool: 如果动作在冷却时间内返回True，否则返回False
        """
        if player_id not in self.player_last_actions:
            self.player_last_actions[player_id] = {}
            return False
            
        if action_label not in self.player_last_actions[player_id]:
            return False
            
        last_time = self.player_last_actions[player_id][action_label]
        time_diff = current_time - last_time
        
        return time_diff < self.cooldown_time
    
    def update_player_last_action(self, player_id, action_label, current_time):
        """
        更新指定球员的指定动作的最后执行时间
        
        参数:
            player_id: 球员ID
            action_label: 动作标签
            current_time: 当前时间戳
        """
        if player_id not in self.player_last_actions:
            self.player_last_actions[player_id] = {}
            
        self.player_last_actions[player_id][action_label] = current_time 