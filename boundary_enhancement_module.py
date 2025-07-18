"""
边界增强模块(Boundary Enhancement Module)实现
用于伪装目标检测中的边界特征增强

作者：Ultralytics YOLO Extension
用途：增强目标边界特征，提高伪装目标检测性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class EdgeDetectionLayer(nn.Module):
    """
    边缘检测层 - 使用可学习的边缘检测卷积核
    
    输入通道变化：
    input: [B, C_in, H, W] -> output: [B, C_in, H, W]
    """
    
    def __init__(self, in_channels: int):
        super(EdgeDetectionLayer, self).__init__()
        self.in_channels = in_channels
        
        # Sobel算子的可学习版本
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                padding=1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                padding=1, groups=in_channels, bias=False)
        
        # 拉普拉斯算子
        self.laplacian = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  padding=1, groups=in_channels, bias=False)
        
        # 初始化边缘检测卷积核
        self._init_edge_kernels()
        
    def _init_edge_kernels(self):
        """初始化边缘检测卷积核"""
        # Sobel X 方向
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        # Sobel Y 方向  
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
        # 拉普拉斯算子
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                       dtype=torch.float32).view(1, 1, 3, 3)
        
        # 为每个输入通道复制卷积核
        for i in range(self.in_channels):
            self.sobel_x.weight.data[i] = sobel_x_kernel
            self.sobel_y.weight.data[i] = sobel_y_kernel  
            self.laplacian.weight.data[i] = laplacian_kernel
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C_in, H, W]
            
        Returns:
            边缘增强特征图 [B, C_in, H, W]
        """
        # 计算梯度幅值
        grad_x = self.sobel_x(x)  # [B, C_in, H, W]
        grad_y = self.sobel_y(x)  # [B, C_in, H, W]
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # [B, C_in, H, W]
        
        # 拉普拉斯边缘检测
        laplacian_edges = torch.abs(self.laplacian(x))  # [B, C_in, H, W]
        
        # 融合边缘信息
        edge_enhanced = gradient_magnitude + laplacian_edges  # [B, C_in, H, W]
        
        return edge_enhanced


class ChannelAttentionForBoundary(nn.Module):
    """
    针对边界特征的通道注意力模块
    
    通道变化：
    input: [B, C_in, H, W] -> output: [B, C_in, H, W]
    中间过程：[B, C_in, H, W] -> [B, C_in, 1, 1] -> [B, C_in//reduction, 1, 1] -> [B, C_in, 1, 1]
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttentionForBoundary, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # -> [B, C_in, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # -> [B, C_in, 1, 1]
        
        # 通道压缩和恢复
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),  # -> [B, C_in//reduction, 1, 1]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)   # -> [B, C_in, 1, 1]
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C_in, H, W]
            
        Returns:
            注意力加权特征图 [B, C_in, H, W]
        """
        # 全局平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x))  # [B, C_in, 1, 1]
        max_out = self.fc(self.max_pool(x))  # [B, C_in, 1, 1]
        
        # 融合注意力权重
        attention = self.sigmoid(avg_out + max_out)  # [B, C_in, 1, 1]
        
        return x * attention  # [B, C_in, H, W]


class SpatialAttentionForBoundary(nn.Module):
    """
    针对边界特征的空间注意力模块
    
    通道变化：
    input: [B, C_in, H, W] -> output: [B, C_in, H, W]
    中间过程：[B, C_in, H, W] -> [B, 2, H, W] -> [B, 1, H, W]
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttentionForBoundary, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)  # [B, 2, H, W] -> [B, 1, H, W]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C_in, H, W]
            
        Returns:
            空间注意力加权特征图 [B, C_in, H, W]
        """
        # 计算通道维度的统计信息
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 拼接统计信息
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # 生成空间注意力图
        attention = self.sigmoid(self.conv(combined))  # [B, 1, H, W]
        
        return x * attention  # [B, C_in, H, W]


class MultiscaleBoundaryFusion(nn.Module):
    """
    多尺度边界特征融合模块
    
    通道变化详解：
    输入多个尺度的特征图，输出融合后的特征图
    """
    
    def __init__(self, in_channels: List[int], out_channels: int):
        super(MultiscaleBoundaryFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 为每个尺度创建特征对齐层
        self.align_convs = nn.ModuleList()
        for ch in in_channels:
            self.align_convs.append(
                nn.Conv2d(ch, out_channels, 1, bias=False)  # 通道对齐：[B, ch, H, W] -> [B, out_channels, H, W]
            )
        
        # 特征融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels), out_channels, 3, padding=1, bias=False),  # 融合
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 多尺度特征图列表 [
                [B, C1, H1, W1],
                [B, C2, H2, W2],
                [B, C3, H3, W3], ...
            ]
            
        Returns:
            融合后的特征图 [B, out_channels, H_target, W_target]
        """
        # 获取目标尺寸（使用第一个特征图的尺寸）
        target_size = features[0].shape[2:]
        
        # 对齐所有特征图
        aligned_features = []
        for i, feat in enumerate(features):
            # 通道对齐
            aligned = self.align_convs[i](feat)  # [B, out_channels, H_i, W_i]
            
            # 空间对齐（上采样到目标尺寸）
            if aligned.shape[2:] != target_size:
                aligned = F.interpolate(aligned, size=target_size, 
                                      mode='bilinear', align_corners=False)  # [B, out_channels, H_target, W_target]
            
            aligned_features.append(aligned)
        
        # 拼接多尺度特征
        fused = torch.cat(aligned_features, dim=1)  # [B, out_channels * num_scales, H_target, W_target]
        
        # 特征融合
        output = self.fusion_conv(fused)  # [B, out_channels, H_target, W_target]
        
        return output


class BoundaryEnhancementModule(nn.Module):
    """
    完整的边界增强模块
    
    完整的通道变化流程：
    1. 输入: [B, C_in, H, W]
    2. 边缘检测: [B, C_in, H, W] -> [B, C_in, H, W]
    3. 通道注意力: [B, C_in, H, W] -> [B, C_in, H, W]
    4. 空间注意力: [B, C_in, H, W] -> [B, C_in, H, W]
    5. 特征精炼: [B, C_in, H, W] -> [B, C_out, H, W]
    6. 残差连接: [B, C_out, H, W]
    """
    
    def __init__(self, in_channels: int, out_channels: int = None, reduction: int = 16):
        super(BoundaryEnhancementModule, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 边缘检测层
        self.edge_detection = EdgeDetectionLayer(in_channels)
        
        # 注意力机制
        self.channel_attention = ChannelAttentionForBoundary(in_channels, reduction)
        self.spatial_attention = SpatialAttentionForBoundary()
        
        # 特征精炼层
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),  # 融合原始和边缘特征
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 残差连接的通道对齐
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 详细的通道变化过程
        
        Args:
            x: 输入特征图 [B, C_in, H, W]
            
        Returns:
            边界增强特征图 [B, C_out, H, W]
        """
        # 保存输入用于残差连接
        identity = x  # [B, C_in, H, W]
        
        # 步骤1: 边缘检测
        edge_features = self.edge_detection(x)  # [B, C_in, H, W]
        
        # 步骤2: 通道注意力
        channel_attended = self.channel_attention(x)  # [B, C_in, H, W]
        
        # 步骤3: 空间注意力
        spatial_attended = self.spatial_attention(channel_attended)  # [B, C_in, H, W]
        
        # 步骤4: 融合原始特征和边缘特征
        combined_features = torch.cat([spatial_attended, edge_features], dim=1)  # [B, C_in*2, H, W]
        
        # 步骤5: 特征精炼
        refined = self.refine_conv(combined_features)  # [B, C_out, H, W]
        
        # 步骤6: 残差连接
        identity = self.residual_conv(identity)  # [B, C_out, H, W]
        output = self.relu(refined + identity)  # [B, C_out, H, W]
        
        return output


class AdvancedBoundaryEnhancementModule(nn.Module):
    """
    高级边界增强模块 - 支持多尺度输入
    
    适用于YOLO等检测网络的多尺度特征融合
    """
    
    def __init__(self, 
                 in_channels_list: List[int] = [256, 512, 1024],  # P3, P4, P5层的通道数
                 out_channels: int = None,  # 如果为None，保持原始通道数
                 reduction: int = 16):
        super(AdvancedBoundaryEnhancementModule, self).__init__()
        
        self.num_scales = len(in_channels_list)
        self.in_channels_list = in_channels_list
        
        # 如果out_channels为None，保持每个尺度的原始通道数
        if out_channels is None:
            self.out_channels_list = in_channels_list
            self.keep_original_channels = True
        else:
            self.out_channels_list = [out_channels] * self.num_scales
            self.keep_original_channels = False
        
        # 为每个尺度创建边界增强模块
        self.boundary_modules = nn.ModuleList()
        for i, in_ch in enumerate(in_channels_list):
            out_ch = self.out_channels_list[i]
            self.boundary_modules.append(
                BoundaryEnhancementModule(in_ch, out_ch, reduction)
            )
        
        # 只有在统一通道数时才需要多尺度融合
        if not self.keep_original_channels:
            # 多尺度特征融合
            self.multiscale_fusion = MultiscaleBoundaryFusion(
                self.out_channels_list, 
                out_channels
            )
            
            # 最终输出投影
            self.output_projections = nn.ModuleList()
            for _ in range(self.num_scales):
                self.output_projections.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播 - 处理多尺度特征
        
        Args:
            features: 多尺度特征图列表 [
                [B, C1, H1, W1],  # P3
                [B, C2, H2, W2],  # P4  
                [B, C3, H3, W3]   # P5
            ]
            
        Returns:
            边界增强的多尺度特征图列表 [
                [B, out_channels_i, H1, W1],
                [B, out_channels_i, H2, W2], 
                [B, out_channels_i, H3, W3]
            ]
        """
        # 步骤1: 对每个尺度进行边界增强
        enhanced_features = []
        for i, feat in enumerate(features):
            enhanced = self.boundary_modules[i](feat)  # [B, out_channels_i, H_i, W_i]
            enhanced_features.append(enhanced)
        
        # 如果保持原始通道数，直接返回增强后的特征
        if self.keep_original_channels:
            return enhanced_features
        
        # 步骤2: 多尺度特征融合 (只有在统一通道数时才执行)
        fused_global = self.multiscale_fusion(enhanced_features)  # [B, out_channels, H_target, W_target]
        
        # 步骤3: 将融合特征分配回各个尺度
        outputs = []
        for i, feat in enumerate(enhanced_features):
            # 将全局融合特征调整到当前尺度
            target_size = feat.shape[2:]
            fused_resized = F.interpolate(fused_global, size=target_size, 
                                        mode='bilinear', align_corners=False)
            
            # 融合局部和全局特征
            combined = feat + fused_resized  # [B, out_channels, H_i, W_i]
            
            # 最终投影
            output = self.output_projections[i](combined)  # [B, out_channels, H_i, W_i]
            outputs.append(output)
        
        return outputs


# 使用示例和测试代码
if __name__ == "__main__":
    # 测试单尺度边界增强模块
    print("=== 单尺度边界增强模块测试 ===")
    
    # 创建模块
    bem = BoundaryEnhancementModule(in_channels=256, out_channels=256)
    
    # 创建测试输入
    x = torch.randn(2, 256, 64, 64)  # [Batch=2, Channels=256, Height=64, Width=64]
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = bem(x)
    print(f"输出形状: {output.shape}")
    print(f"通道变化: {x.shape[1]} -> {output.shape[1]}")
    
    print("\n=== 多尺度边界增强模块测试 ===")
    
    # 创建高级模块
    advanced_bem = AdvancedBoundaryEnhancementModule(
        in_channels_list=[256, 512, 1024],
        out_channels=256
    )
    
    # 创建多尺度测试输入（模拟YOLO的P3, P4, P5特征层）
    features = [
        torch.randn(2, 256, 80, 80),   # P3: [B, 256, 80, 80]
        torch.randn(2, 512, 40, 40),   # P4: [B, 512, 40, 40]  
        torch.randn(2, 1024, 20, 20)   # P5: [B, 1024, 20, 20]
    ]
    
    print("输入特征形状:")
    for i, feat in enumerate(features):
        print(f"  P{i+3}: {feat.shape}")
    
    # 前向传播
    enhanced_features = advanced_bem(features)
    
    print("输出特征形状:")
    for i, feat in enumerate(enhanced_features):
        print(f"  P{i+3}: {feat.shape}")
    
    print("\n=== 通道变化总结 ===")
    print("单尺度模块:")
    print("  输入: [B, C_in, H, W]")
    print("  边缘检测: [B, C_in, H, W] -> [B, C_in, H, W]")
    print("  通道注意力: [B, C_in, H, W] -> [B, C_in, H, W]")  
    print("  空间注意力: [B, C_in, H, W] -> [B, C_in, H, W]")
    print("  特征融合: [B, C_in*2, H, W] -> [B, C_out, H, W]")
    print("  输出: [B, C_out, H, W]")
    
    print("\n多尺度模块:")
    print("  P3: [B, 256, H1, W1] -> [B, 256, H1, W1]")
    print("  P4: [B, 512, H2, W2] -> [B, 256, H2, W2]") 
    print("  P5: [B, 1024, H3, W3] -> [B, 256, H3, W3]") 