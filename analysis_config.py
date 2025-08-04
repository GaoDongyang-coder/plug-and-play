import numpy as np
from temporal_analysis import TemporalActionAnalyzer, PositionFeatures

# ID 映射关系 (类别索引 -> 球员ID)
player_id_mapping = {
    0: 1, 1: 18, 2: 17, 3: 13, 4: 8,
    5: 11, 6: 5, 7: 14, 8: 6
    # Add more mappings if your model detects more player classes
}

# 全局时序分析器字典 - 为每个球员ID维护独立的分析器
temporal_analyzers = {}

def get_or_create_temporal_analyzer(player_id):
    """获取或创建指定球员的时序分析器"""
    if player_id not in temporal_analyzers:
        temporal_analyzers[player_id] = TemporalActionAnalyzer(window_size=30, quality_threshold=0.6)
    return temporal_analyzers[player_id]

def reset_temporal_analyzers():
    """重置所有时序分析器"""
    global temporal_analyzers
    temporal_analyzers = {}

def recognize_action_with_temporal(keypoints, player_id=None, bbox=None):
    """
    结合时序分析的动作识别函数
    
    Args:
        keypoints: 关键点数组 (17, 2)
        player_id: 球员ID，用于维护独立的时序分析器
        bbox: 边界框信息
        
    Returns:
        tuple: (action_label, confidence, temporal_features)
    """
    # 首先进行传统的几何规则识别
    traditional_action = recognize_action(keypoints)
    
    # 如果没有提供球员ID，只返回传统识别结果
    if player_id is None:
        return traditional_action, 0.5, None
    
    # 获取该球员的时序分析器
    analyzer = get_or_create_temporal_analyzer(player_id)
    
    # 添加当前帧到时序分析器
    success = analyzer.add_frame(keypoints, bbox)
    
    if not success or not analyzer.is_ready_for_analysis():
        # 如果时序分析器还没准备好，返回传统识别结果
        return traditional_action, 0.5, None
    
    # 提取位置特征
    position_features = analyzer.get_position_features()
    
    if position_features is None:
        return traditional_action, 0.5, None
    
    # 基于时序特征调整动作识别结果和置信度
    enhanced_action, enhanced_confidence = enhance_action_with_temporal_features(
        traditional_action, position_features, keypoints
    )
    
    return enhanced_action, enhanced_confidence, position_features

def enhance_action_with_temporal_features(traditional_action, position_features, keypoints):
    """
    基于时序特征增强动作识别
    
    Args:
        traditional_action: 传统几何规则识别的动作
        position_features: 位置特征
        keypoints: 当前帧关键点
        
    Returns:
        tuple: (enhanced_action, confidence)
    """
    base_confidence = 0.6
    
    # 如果传统识别结果是Unknown，尝试基于时序特征识别
    if traditional_action == "Unknown":
        temporal_action = classify_action_from_temporal_features(position_features)
        if temporal_action != "Unknown":
            return temporal_action, 0.4  # 较低的置信度，因为只基于时序特征
    
    # 基于时序特征验证和增强传统识别结果
    confidence_adjustment = calculate_temporal_confidence_adjustment(
        traditional_action, position_features
    )
    
    enhanced_confidence = min(base_confidence + confidence_adjustment, 1.0)
    
    return traditional_action, enhanced_confidence

def classify_action_from_temporal_features(position_features):
    """
    基于时序特征进行动作分类
    
    Args:
        position_features: 位置特征
        
    Returns:
        str: 动作标签
    """
    if not position_features or not position_features.trajectory_stats:
        return "Unknown"
    
    # 分析手腕轨迹特征
    wrist_features = position_features.trajectory_stats.get('wrists', {})
    
    if wrist_features:
        avg_path_length = wrist_features.get('avg_path_length', 0)
        avg_y_range = wrist_features.get('avg_y_range', 0)
        
        # 基于轨迹长度和垂直范围进行简单分类
        if avg_path_length > 100 and avg_y_range > 80:
            # 大幅度运动，可能是攻击或发球
            return "Attack"
        elif avg_path_length > 50 and avg_y_range < 40:
            # 中等运动幅度，水平为主，可能是传球
            return "Set"
        elif avg_path_length < 30:
            # 小幅度运动，可能是接发球或防守
            return "Reception"
    
    return "Unknown"

def calculate_temporal_confidence_adjustment(action, position_features):
    """
    基于时序特征计算置信度调整值
    
    Args:
        action: 识别的动作
        position_features: 位置特征
        
    Returns:
        float: 置信度调整值 (-0.3 到 +0.3)
    """
    if not position_features:
        return 0.0
    
    adjustment = 0.0
    
    # 基于轨迹平滑度调整置信度
    wrist_features = position_features.trajectory_stats.get('wrists', {})
    if wrist_features:
        smoothness = wrist_features.get('avg_smoothness', 0.5)
        if smoothness > 0.7:
            adjustment += 0.1  # 平滑的轨迹增加置信度
        elif smoothness < 0.3:
            adjustment -= 0.1  # 不平滑的轨迹降低置信度
    
    # 基于稳定性调整置信度
    stability_features = position_features.stability_stats.get('wrists', {})
    if stability_features:
        stability = stability_features.get('avg_stability', 0.5)
        if action in ['Reception', 'Set'] and stability > 0.6:
            adjustment += 0.1  # 接发球和传球需要稳定性
        elif action in ['Attack', 'Serve'] and stability < 0.4:
            adjustment += 0.1  # 攻击和发球允许不稳定
    
    # 基于位移特征调整置信度
    displacement_features = position_features.displacement_stats.get('wrists', {})
    if displacement_features:
        avg_magnitude = displacement_features.get('avg_magnitude', 0)
        if action == 'Attack' and avg_magnitude > 100:
            adjustment += 0.2  # 攻击动作应该有大的位移
        elif action == 'Reception' and avg_magnitude < 50:
            adjustment += 0.1  # 接发球动作位移较小
    
    return max(-0.3, min(0.3, adjustment))

# --- 原始动作识别函数 ---
def recognize_action(keypoints):
    """Recognizes action based on keypoints. Supports 6 volleyball actions."""
    # Input: keypoints is likely a (17, 2) numpy array [x, y]
    # Basic validity check: ensure we have 17 points and no negative coords
    if keypoints is None or keypoints.shape != (17, 2) or np.any(keypoints < 0):
        return "Unknown"

    try:
        # Get coordinates
        nose_x, nose_y = keypoints[0]
        
        l_sho_x, l_sho_y = keypoints[5]
        r_sho_x, r_sho_y = keypoints[6]
        
        l_elb_x, l_elb_y = keypoints[7]
        r_elb_x, r_elb_y = keypoints[8]
        
        l_wri_x, l_wri_y = keypoints[9]
        r_wri_x, r_wri_y = keypoints[10]
        
        l_hip_x, l_hip_y = keypoints[11]
        r_hip_x, r_hip_y = keypoints[12]
        
        l_knee_x, l_knee_y = keypoints[13]
        r_knee_x, r_knee_y = keypoints[14]
        
        l_ankle_x, l_ankle_y = keypoints[15]
        r_ankle_x, r_ankle_y = keypoints[16]

        # --- Define thresholds and helpers ---
        # Check if crucial points are detected (Y > 0)
        head_points_valid = nose_y > 0
        shoulders_valid = l_sho_y > 0 and r_sho_y > 0
        wrists_valid = l_wri_y > 0 and r_wri_y > 0
        hips_valid = l_hip_y > 0 and r_hip_y > 0
        elbows_valid = l_elb_y > 0 and r_elb_y > 0
        knees_valid = l_knee_y > 0 and r_knee_y > 0
        ankles_valid = l_ankle_y > 0 and r_ankle_y > 0

        # Calculate average positions
        shoulder_avg_y = (l_sho_y + r_sho_y) / 2 if shoulders_valid else -1
        shoulder_avg_x = (l_sho_x + r_sho_x) / 2 if shoulders_valid else -1
        hip_avg_y = (l_hip_y + r_hip_y) / 2 if hips_valid else -1
        knee_avg_y = (l_knee_y + r_knee_y) / 2 if knees_valid else -1
        ankle_avg_y = (l_ankle_y + r_ankle_y) / 2 if ankles_valid else -1

        # Calculate torso length (shoulder to hip)
        torso_length = abs(shoulder_avg_y - hip_avg_y) if shoulders_valid and hips_valid else 100

        # Calculate head height proxy
        head_height_proxy = 50  # Default value
        if head_points_valid and shoulders_valid:
            dist = abs(shoulder_avg_y - nose_y)
            if dist > 10:
                head_height_proxy = dist

        # --- Action Recognition Rules ---

        # 1. Ser (Serve) - 发球
        if wrists_valid and head_points_valid and shoulders_valid:
            # 上手发球特征：一只手高举过头，另一只手持球或在前方
            overhand_serve = (l_wri_y < (nose_y - head_height_proxy * 0.2) or 
                             r_wri_y < (nose_y - head_height_proxy * 0.2))
            
            # 下手发球特征：一只手在腰部以下，另一只手在前方或轻微弯曲
            underhand_serve = False
            if hips_valid:
                left_hand_low = l_wri_y > hip_avg_y
                right_hand_low = r_wri_y > hip_avg_y
                hands_separated = abs(l_wri_x - r_wri_x) > torso_length * 0.5
                underhand_serve = (left_hand_low or right_hand_low) and hands_separated
            
            if overhand_serve or underhand_serve:
                return "Ser"

        # 2. Rec (Receive) - 接发球
        if wrists_valid and shoulders_valid and hips_valid:
            # 接发球特征：双手在前方伸直，身体保持准备姿势
            arms_extended_forward = (l_wri_y > shoulder_avg_y and r_wri_y > shoulder_avg_y and
                                   l_wri_y < (hip_avg_y + knee_avg_y) / 2 and r_wri_y < (hip_avg_y + knee_avg_y) / 2)
            hands_together = abs(l_wri_y - r_wri_y) < torso_length * 0.3
            hands_centered = abs((l_wri_x + r_wri_x) / 2 - shoulder_avg_x) < torso_length * 0.3
            
            # 膝盖微屈但不是深蹲姿势
            slight_knee_bend = False
            if knees_valid and ankles_valid:
                knee_bend_ratio = (knee_avg_y - hip_avg_y) / (ankle_avg_y - hip_avg_y)
                slight_knee_bend = 0.35 < knee_bend_ratio < 0.7
            
            if arms_extended_forward and hands_together and hands_centered and slight_knee_bend:
                return "Rec"

        # 3. Dig - 防守/救球
        if wrists_valid and shoulders_valid and hips_valid:
            # 防守特征：双手低于髋部，可能有单手救球，身体前倾或下潜
            wrists_low = l_wri_y > hip_avg_y or r_wri_y > hip_avg_y
            
            # 判断身体姿势：下蹲或前扑
            body_lowered = False
            if knees_valid and hips_valid:
                deep_knee_bend = knee_avg_y > (hip_avg_y + 0.3 * torso_length)
                body_lowered = deep_knee_bend
            
            # 单手或双手救球
            diving_save = False
            if wrists_valid and hips_valid:
                diving_save = (l_wri_y > hip_avg_y + 0.5 * torso_length or 
                              r_wri_y > hip_avg_y + 0.5 * torso_length)
            
            if (wrists_low and body_lowered) or diving_save:
                return "Dig"

        # 4. Attk (Attack) - 进攻
        if wrists_valid and shoulders_valid and hips_valid and ankles_valid:
            # 进攻特征：一只手高举(准备击球)，身体在空中或起跳阶段
            attacking_arm = l_wri_y < shoulder_avg_y - torso_length * 0.3 or r_wri_y < shoulder_avg_y - torso_length * 0.3
            
            # 判断是否腾空/起跳 (关键特征)
            jumping = False
            if hips_valid and ankles_valid:
                # 髋部位置异常高(相对于脚踝)表示可能在空中
                hip_height_ratio = (ankle_avg_y - hip_avg_y) / torso_length
                jumping = hip_height_ratio > 1.5  # 髋部与脚踝距离大于标准站立高度
            
            if attacking_arm and jumping:
                return "Attk"

        # 5. Blk (Block) - 拦网
        if wrists_valid and head_points_valid and shoulders_valid and hips_valid:
            # 拦网特征：双手高举，手臂伸直，身体垂直或在空中
            both_hands_high = (l_wri_y < shoulder_avg_y - torso_length * 0.3 and 
                              r_wri_y < shoulder_avg_y - torso_length * 0.3)
            
            # 手臂垂直伸展，双手接近中线
            arms_vertical = True
            if elbows_valid:
                arms_vertical = (l_elb_y > l_wri_y and l_elb_y < l_sho_y and 
                                r_elb_y > r_wri_y and r_elb_y < r_sho_y)
            
            # 手部间距适中(拦网手型)
            hands_blocking_position = abs(l_wri_x - r_wri_x) < torso_length * 0.8
            
            # 判断是否有跳起
            jumping = False
            if hips_valid and ankles_valid:
                hip_height_ratio = (ankle_avg_y - hip_avg_y) / torso_length
                jumping = hip_height_ratio > 1.3
            
            if both_hands_high and arms_vertical and hands_blocking_position and jumping:
                return "Blk"

        # 6. Set (Setter) - 传球/二传
        if wrists_valid and shoulders_valid and head_points_valid and elbows_valid:
            # 二传特征：双手在头顶上方，肘部明显弯曲，手指张开(难以检测)
            hands_overhead = (l_wri_y < nose_y - head_height_proxy * 0.3 and 
                            r_wri_y < nose_y - head_height_proxy * 0.3)
            
            # 肘部弯曲 - 二传手型
            elbows_bent = (l_elb_y > l_sho_y - torso_length * 0.2 and 
                          r_elb_y > r_sho_y - torso_length * 0.2)
            
            # 手部位置对称且接近
            hands_symmetric = abs(l_wri_y - r_wri_y) < head_height_proxy * 0.5
            hands_close = abs(l_wri_x - r_wri_x) < torso_length * 0.6
            
            if hands_overhead and elbows_bent and hands_symmetric and hands_close:
                return "Set"

        # 7. 准备姿势 (可选添加)
        if shoulders_valid and hips_valid and knees_valid and ankles_valid:
            # 准备姿势特征：轻微弯曲膝盖，手臂在身体前方，身体略微前倾
            slight_knee_bend = False
            if knees_valid and hips_valid and ankles_valid:
                knee_bend_ratio = (knee_avg_y - hip_avg_y) / (ankle_avg_y - hip_avg_y)
                slight_knee_bend = 0.4 < knee_bend_ratio < 0.65
                
            arms_ready_position = False
            if wrists_valid and shoulders_valid:
                arms_ready = (l_wri_y > shoulder_avg_y and r_wri_y > shoulder_avg_y and
                             l_wri_y < hip_avg_y and r_wri_y < hip_avg_y)
                arms_ready_position = arms_ready
                
            if slight_knee_bend and arms_ready_position:
                return "Ready"  # 可添加准备姿势

    except IndexError:
        print("IndexError accessing keypoints for action recognition.")
        return "Unknown"
    except Exception as e:
        print(f"Error during action recognition: {e}")
        return "Unknown"

    # Default if no specific action recognized
    return "Unknown"