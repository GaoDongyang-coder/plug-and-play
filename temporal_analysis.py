import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

@dataclass
class KeypointFrame:
    """单帧关键点数据结构"""
    frame_id: int
    timestamp: float
    keypoints: np.ndarray  # (17, 2)
    quality_score: float
    bbox: Optional[Tuple[int, int, int, int]] = None

@dataclass
class PositionFeatures:
    """位置特征数据结构"""
    trajectory_stats: Dict
    displacement_stats: Dict
    direction_stats: Dict
    stability_stats: Dict

class KeypointBuffer:
    """关键点缓存管理器"""
    
    def __init__(self, window_size: int = 30, quality_threshold: float = 0.6):
        """
        初始化关键点缓存器
        
        Args:
            window_size: 时间窗口大小（帧数）
            quality_threshold: 关键点质量阈值
        """
        self.window_size = window_size
        self.quality_threshold = quality_threshold
        self.buffer = deque(maxlen=window_size)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def add_frame(self, frame_id: int, keypoints: np.ndarray, bbox: Optional[Tuple] = None) -> bool:
        """
        添加新帧到缓存
        
        Args:
            frame_id: 帧ID
            keypoints: 关键点数组 (17, 2)
            bbox: 边界框 (x1, y1, x2, y2)
            
        Returns:
            bool: 是否成功添加
        """
        if keypoints is None or keypoints.shape != (17, 2):
            return False
        
        timestamp = time.time()
        quality_score = self._calculate_quality_score(keypoints)
        
        # 如果质量太低，尝试插值
        if quality_score < self.quality_threshold and len(self.buffer) >= 2:
            keypoints = self._interpolate_keypoints(keypoints)
            quality_score = self._calculate_quality_score(keypoints)
        
        # 异常值检测和平滑
        if len(self.buffer) > 0:
            keypoints = self._smooth_outliers(keypoints)
        
        frame_data = KeypointFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            keypoints=keypoints.copy(),
            quality_score=quality_score,
            bbox=bbox
        )
        
        self.buffer.append(frame_data)
        return True
    
    def get_sequence(self, length: Optional[int] = None) -> List[KeypointFrame]:
        """
        获取关键点序列
        
        Args:
            length: 序列长度，None表示返回全部
            
        Returns:
            List[KeypointFrame]: 关键点序列
        """
        if length is None:
            return list(self.buffer)
        else:
            return list(self.buffer)[-length:] if len(self.buffer) >= length else list(self.buffer)
    
    def _calculate_quality_score(self, keypoints: np.ndarray) -> float:
        """计算关键点质量分数"""
        if keypoints is None or keypoints.shape != (17, 2):
            return 0.0
        
        # 计算有效关键点数量
        valid_points = np.sum((keypoints[:, 0] > 0) & (keypoints[:, 1] > 0))
        quality_ratio = valid_points / 17.0
        
        # 检查关键点分布的合理性
        if valid_points >= 8:
            # 检查身体比例是否合理
            reasonableness_score = self._check_body_proportions(keypoints)
            quality_score = quality_ratio * 0.7 + reasonableness_score * 0.3
        else:
            quality_score = quality_ratio * 0.5  # 有效点太少，降低质量分数
        
        return min(quality_score, 1.0)
    
    def _check_body_proportions(self, keypoints: np.ndarray) -> float:
        """检查身体比例合理性"""
        try:
            # 检查肩膀到髋部的距离
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            if all(point[0] > 0 and point[1] > 0 for point in [left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_center = (left_shoulder + right_shoulder) / 2
                hip_center = (left_hip + right_hip) / 2
                torso_length = np.linalg.norm(shoulder_center - hip_center)
                
                # 检查躯干长度是否在合理范围内
                if 30 < torso_length < 300:  # 像素范围
                    return 1.0
                else:
                    return 0.5
            
            return 0.7  # 部分关键点缺失但可接受
            
        except Exception:
            return 0.3
    
    def _interpolate_keypoints(self, current_keypoints: np.ndarray) -> np.ndarray:
        """插值缺失的关键点"""
        if len(self.buffer) < 2:
            return current_keypoints
        
        # 使用最近两帧进行线性插值
        prev_frame1 = self.buffer[-1]
        prev_frame2 = self.buffer[-2]
        
        interpolated = current_keypoints.copy()
        
        for i in range(17):
            if current_keypoints[i, 0] <= 0 or current_keypoints[i, 1] <= 0:
                # 当前关键点无效，尝试插值
                if (prev_frame1.keypoints[i, 0] > 0 and prev_frame1.keypoints[i, 1] > 0 and
                    prev_frame2.keypoints[i, 0] > 0 and prev_frame2.keypoints[i, 1] > 0):
                    # 线性外推
                    velocity = prev_frame1.keypoints[i] - prev_frame2.keypoints[i]
                    interpolated[i] = prev_frame1.keypoints[i] + velocity
                elif prev_frame1.keypoints[i, 0] > 0 and prev_frame1.keypoints[i, 1] > 0:
                    # 使用前一帧的值
                    interpolated[i] = prev_frame1.keypoints[i]
        
        return interpolated
    
    def _smooth_outliers(self, keypoints: np.ndarray) -> np.ndarray:
        """平滑异常值"""
        if len(self.buffer) < 3:
            return keypoints
        
        smoothed = keypoints.copy()
        recent_frames = list(self.buffer)[-3:]  # 最近3帧
        
        for i in range(17):
            if keypoints[i, 0] > 0 and keypoints[i, 1] > 0:
                # 计算与最近几帧的距离
                distances = []
                for frame in recent_frames:
                    if frame.keypoints[i, 0] > 0 and frame.keypoints[i, 1] > 0:
                        dist = np.linalg.norm(keypoints[i] - frame.keypoints[i])
                        distances.append(dist)
                
                if distances:
                    avg_distance = np.mean(distances)
                    max_distance = np.max(distances)
                    
                    # 如果当前点距离过大，认为是异常值
                    if max_distance > avg_distance * 3 and max_distance > 50:  # 阈值可调
                        # 使用最近有效帧的加权平均
                        valid_points = []
                        weights = []
                        for j, frame in enumerate(recent_frames):
                            if frame.keypoints[i, 0] > 0 and frame.keypoints[i, 1] > 0:
                                valid_points.append(frame.keypoints[i])
                                weights.append(frame.quality_score * (j + 1))  # 越近的帧权重越大
                        
                        if valid_points:
                            weights = np.array(weights)
                            weights = weights / np.sum(weights)
                            smoothed[i] = np.average(valid_points, axis=0, weights=weights)
        
        return smoothed
    
    def is_ready(self, min_frames: int = 5) -> bool:
        """检查缓存是否准备好进行分析"""
        return len(self.buffer) >= min_frames
    
    def get_buffer_info(self) -> Dict:
        """获取缓存状态信息"""
        if not self.buffer:
            return {'size': 0, 'avg_quality': 0.0, 'time_span': 0.0}
        
        avg_quality = np.mean([frame.quality_score for frame in self.buffer])
        time_span = self.buffer[-1].timestamp - self.buffer[0].timestamp if len(self.buffer) > 1 else 0.0
        
        return {
            'size': len(self.buffer),
            'avg_quality': avg_quality,
            'time_span': time_span,
            'window_size': self.window_size
        }

class PositionFeatureExtractor:
    """位置特征提取器"""
    
    def __init__(self):
        """初始化位置特征提取器"""
        self.key_joints = {
            'wrists': [9, 10],      # 左右手腕
            'elbows': [7, 8],       # 左右肘部
            'shoulders': [5, 6],    # 左右肩膀
            'hips': [11, 12],       # 左右髋部
            'knees': [13, 14],      # 左右膝盖
            'ankles': [15, 16]      # 左右脚踝
        }
    
    def extract_features(self, keypoint_sequence: List[KeypointFrame]) -> PositionFeatures:
        """
        提取位置特征
        
        Args:
            keypoint_sequence: 关键点序列
            
        Returns:
            PositionFeatures: 位置特征
        """
        if len(keypoint_sequence) < 2:
            return self._empty_features()
        
        # 提取轨迹特征
        trajectory_stats = self._extract_trajectory_features(keypoint_sequence)
        
        # 提取位移特征
        displacement_stats = self._extract_displacement_features(keypoint_sequence)
        
        # 提取方向特征
        direction_stats = self._extract_direction_features(keypoint_sequence)
        
        # 提取稳定性特征
        stability_stats = self._extract_stability_features(keypoint_sequence)
        
        return PositionFeatures(
            trajectory_stats=trajectory_stats,
            displacement_stats=displacement_stats,
            direction_stats=direction_stats,
            stability_stats=stability_stats
        )
    
    def _extract_trajectory_features(self, sequence: List[KeypointFrame]) -> Dict:
        """提取轨迹特征"""
        features = {}
        
        for joint_name, joint_indices in self.key_joints.items():
            joint_trajectories = []
            
            for joint_idx in joint_indices:
                trajectory = []
                for frame in sequence:
                    if (frame.keypoints[joint_idx, 0] > 0 and 
                        frame.keypoints[joint_idx, 1] > 0):
                        trajectory.append(frame.keypoints[joint_idx])
                
                if len(trajectory) >= 2:
                    trajectory = np.array(trajectory)
                    
                    # 计算轨迹长度
                    path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
                    
                    # 计算轨迹平滑度
                    if len(trajectory) >= 3:
                        smoothness = self._calculate_smoothness(trajectory)
                    else:
                        smoothness = 1.0
                    
                    # 计算轨迹范围
                    trajectory_range = {
                        'x_range': np.max(trajectory[:, 0]) - np.min(trajectory[:, 0]),
                        'y_range': np.max(trajectory[:, 1]) - np.min(trajectory[:, 1])
                    }
                    
                    joint_trajectories.append({
                        'path_length': path_length,
                        'smoothness': smoothness,
                        'range': trajectory_range,
                        'points_count': len(trajectory)
                    })
            
            if joint_trajectories:
                # 计算该关节组的统计特征
                features[joint_name] = {
                    'avg_path_length': np.mean([t['path_length'] for t in joint_trajectories]),
                    'max_path_length': np.max([t['path_length'] for t in joint_trajectories]),
                    'avg_smoothness': np.mean([t['smoothness'] for t in joint_trajectories]),
                    'avg_x_range': np.mean([t['range']['x_range'] for t in joint_trajectories]),
                    'avg_y_range': np.mean([t['range']['y_range'] for t in joint_trajectories])
                }
        
        return features
    
    def _extract_displacement_features(self, sequence: List[KeypointFrame]) -> Dict:
        """提取位移特征"""
        features = {}
        
        if len(sequence) < 2:
            return features
        
        first_frame = sequence[0]
        last_frame = sequence[-1]
        
        for joint_name, joint_indices in self.key_joints.items():
            displacements = []
            
            for joint_idx in joint_indices:
                start_point = first_frame.keypoints[joint_idx]
                end_point = last_frame.keypoints[joint_idx]
                
                if (start_point[0] > 0 and start_point[1] > 0 and
                    end_point[0] > 0 and end_point[1] > 0):
                    
                    displacement = np.linalg.norm(end_point - start_point)
                    displacement_vector = end_point - start_point
                    
                    displacements.append({
                        'magnitude': displacement,
                        'x_displacement': displacement_vector[0],
                        'y_displacement': displacement_vector[1]
                    })
            
            if displacements:
                features[joint_name] = {
                    'avg_magnitude': np.mean([d['magnitude'] for d in displacements]),
                    'max_magnitude': np.max([d['magnitude'] for d in displacements]),
                    'avg_x_displacement': np.mean([d['x_displacement'] for d in displacements]),
                    'avg_y_displacement': np.mean([d['y_displacement'] for d in displacements])
                }
        
        return features
    
    def _extract_direction_features(self, sequence: List[KeypointFrame]) -> Dict:
        """提取方向特征"""
        features = {}
        
        for joint_name, joint_indices in self.key_joints.items():
            direction_changes = []
            
            for joint_idx in joint_indices:
                directions = []
                
                for i in range(len(sequence) - 1):
                    current_point = sequence[i].keypoints[joint_idx]
                    next_point = sequence[i + 1].keypoints[joint_idx]
                    
                    if (current_point[0] > 0 and current_point[1] > 0 and
                        next_point[0] > 0 and next_point[1] > 0):
                        
                        direction_vector = next_point - current_point
                        if np.linalg.norm(direction_vector) > 1e-6:  # 避免除零
                            direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
                            directions.append(direction_angle)
                
                if len(directions) >= 2:
                    # 计算方向变化
                    direction_changes_for_joint = []
                    for i in range(len(directions) - 1):
                        angle_change = abs(directions[i + 1] - directions[i])
                        # 处理角度跳跃
                        if angle_change > np.pi:
                            angle_change = 2 * np.pi - angle_change
                        direction_changes_for_joint.append(angle_change)
                    
                    if direction_changes_for_joint:
                        direction_changes.extend(direction_changes_for_joint)
            
            if direction_changes:
                features[joint_name] = {
                    'avg_direction_change': np.mean(direction_changes),
                    'max_direction_change': np.max(direction_changes),
                    'direction_stability': 1.0 / (1.0 + np.std(direction_changes))  # 稳定性指标
                }
        
        return features
    
    def _extract_stability_features(self, sequence: List[KeypointFrame]) -> Dict:
        """提取稳定性特征"""
        features = {}
        
        for joint_name, joint_indices in self.key_joints.items():
            stability_scores = []
            
            for joint_idx in joint_indices:
                positions = []
                for frame in sequence:
                    if (frame.keypoints[joint_idx, 0] > 0 and 
                        frame.keypoints[joint_idx, 1] > 0):
                        positions.append(frame.keypoints[joint_idx])
                
                if len(positions) >= 3:
                    positions = np.array(positions)
                    
                    # 计算位置方差（稳定性的逆指标）
                    position_variance = np.var(positions, axis=0)
                    total_variance = np.sum(position_variance)
                    
                    # 计算抖动程度
                    if len(positions) >= 3:
                        jitter = self._calculate_jitter(positions)
                    else:
                        jitter = 0.0
                    
                    stability_score = 1.0 / (1.0 + total_variance + jitter)
                    stability_scores.append(stability_score)
            
            if stability_scores:
                features[joint_name] = {
                    'avg_stability': np.mean(stability_scores),
                    'min_stability': np.min(stability_scores)
                }
        
        return features
    
    def _calculate_smoothness(self, trajectory: np.ndarray) -> float:
        """计算轨迹平滑度"""
        if len(trajectory) < 3:
            return 1.0
        
        # 计算二阶导数（加速度）的变化
        first_derivative = np.diff(trajectory, axis=0)
        second_derivative = np.diff(first_derivative, axis=0)
        
        # 平滑度与加速度变化的标准差成反比
        acceleration_magnitude = np.linalg.norm(second_derivative, axis=1)
        smoothness = 1.0 / (1.0 + np.std(acceleration_magnitude))
        
        return smoothness
    
    def _calculate_jitter(self, positions: np.ndarray) -> float:
        """计算抖动程度"""
        if len(positions) < 3:
            return 0.0
        
        # 计算相邻点之间的距离变化
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        distance_changes = np.abs(np.diff(distances))
        
        # 抖动程度为距离变化的标准差
        jitter = np.std(distance_changes) if len(distance_changes) > 0 else 0.0
        
        return jitter
    
    def _empty_features(self) -> PositionFeatures:
        """返回空的特征结构"""
        return PositionFeatures(
            trajectory_stats={},
            displacement_stats={},
            direction_stats={},
            stability_stats={}
        )

class TemporalActionAnalyzer:
    """时序动作分析器 - 整合缓存管理和特征提取"""
    
    def __init__(self, window_size: int = 30, quality_threshold: float = 0.6):
        """
        初始化时序动作分析器
        
        Args:
            window_size: 时间窗口大小
            quality_threshold: 关键点质量阈值
        """
        self.buffer = KeypointBuffer(window_size, quality_threshold)
        self.feature_extractor = PositionFeatureExtractor()
        self.frame_counter = 0
    
    def add_frame(self, keypoints: np.ndarray, bbox: Optional[Tuple] = None) -> bool:
        """
        添加新帧进行分析
        
        Args:
            keypoints: 关键点数组 (17, 2)
            bbox: 边界框
            
        Returns:
            bool: 是否成功添加
        """
        self.frame_counter += 1
        return self.buffer.add_frame(self.frame_counter, keypoints, bbox)
    
    def get_position_features(self, sequence_length: Optional[int] = None) -> Optional[PositionFeatures]:
        """
        获取位置特征
        
        Args:
            sequence_length: 分析序列长度
            
        Returns:
            PositionFeatures: 位置特征，如果数据不足则返回None
        """
        if not self.buffer.is_ready():
            return None
        
        sequence = self.buffer.get_sequence(sequence_length)
        return self.feature_extractor.extract_features(sequence)
    
    def is_ready_for_analysis(self) -> bool:
        """检查是否准备好进行分析"""
        return self.buffer.is_ready()
    
    def get_analysis_info(self) -> Dict:
        """获取分析器状态信息"""
        buffer_info = self.buffer.get_buffer_info()
        buffer_info['frame_counter'] = self.frame_counter
        buffer_info['ready_for_analysis'] = self.is_ready_for_analysis()
        return buffer_info
    
    def reset(self):
        """重置分析器"""
        self.buffer = KeypointBuffer(self.buffer.window_size, self.buffer.quality_threshold)
        self.frame_counter = 0 