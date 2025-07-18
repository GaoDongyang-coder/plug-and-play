"""
MobileViT增强模块(MobileViT Enhancement Module)实现
基于MobileViT Block设计的轻量化全局-局部特征增强模块

作者：Ultralytics YOLO Extension  
用途：结合CNN局部特征和ViT全局特征，提升伪装目标检测性能
设计理念：即插即用，不改变通道数，保持与YOLO11n完全兼容
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class MobileViTBlock(nn.Module):
    """
    轻量化MobileViT块
    
    通道变化详解：
    input: [B, C_in, H, W] 
    -> local_conv: [B, C_in, H, W] (局部特征)
    -> expand: [B, d, H, W] (通道扩展, d=2*C_in)
    -> unfold: [B, d, P, N] (patch化, P=patch_size^2, N=HW/P)
    -> transformer: [B, d, P, N] (全局建模)
    -> fold: [B, d, H, W] (重组空间)
    -> project: [B, C_in, H, W] (通道投影)
    -> fusion: [B, C_in, H, W] (特征融合)
    """
    
    def __init__(self, 
                 in_channels: int,
                 transformer_dim: Optional[int] = None,
                 patch_size: int = 2,
                 num_transformer_layers: int = 2,
                 num_heads: int = 4,
                 mlp_ratio: float = 2.0,
                 dropout: float = 0.0):
        super(MobileViTBlock, self).__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        
        # 如果未指定transformer维度，使用2倍输入通道数
        if transformer_dim is None:
            transformer_dim = in_channels * 2
        self.transformer_dim = transformer_dim
        
        # 局部特征提取（3x3卷积）
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        # 通道扩展（为Transformer准备）
        self.expand_conv = nn.Conv2d(in_channels, transformer_dim, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(transformer_dim)
        
        # Transformer层（轻量化版本）
        self.transformer_layers = nn.ModuleList([
            LightweightTransformerLayer(
                dim=transformer_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(num_transformer_layers)
        ])
        
        # 通道投影（恢复原始通道数）
        self.project_conv = nn.Conv2d(transformer_dim, in_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(in_channels)
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C_in, H, W]
            
        Returns:
            增强后的特征图 [B, C_in, H, W] (通道数保持不变)
        """
        B, C, H, W = x.shape
        
        # Step 1: 局部特征提取
        local_features = self.local_conv(x)  # [B, C_in, H, W]
        
        # Step 2: 通道扩展
        expanded = self.expand_conv(x)  # [B, transformer_dim, H, W]
        expanded = self.expand_bn(expanded)
        expanded = F.silu(expanded)
        
        # Step 3: Patch化 (Unfold)
        patches = self._unfold(expanded)  # [B, transformer_dim, N, P]
        
        # Step 4: Transformer处理
        for transformer_layer in self.transformer_layers:
            patches = transformer_layer(patches)  # [B, transformer_dim, N, P]
        
        # Step 5: 重组空间结构 (Fold)
        global_features = self._fold(patches, H, W)  # [B, transformer_dim, H, W]
        
        # Step 6: 通道投影
        global_features = self.project_conv(global_features)  # [B, C_in, H, W]
        global_features = self.project_bn(global_features)
        
        # Step 7: 特征融合
        fused = torch.cat([local_features, global_features], dim=1)  # [B, 2*C_in, H, W]
        output = self.fusion_conv(fused)  # [B, C_in, H, W]
        
        # 残差连接
        return output + x
    
    def _unfold(self, x: torch.Tensor) -> torch.Tensor:
        """
        将特征图转换为patches
        
        Args:
            x: [B, C, H, W]
            
        Returns:
            patches: [B, C, N, P] where N=num_patches, P=patch_size^2
        """
        B, C, H, W = x.shape
        P = self.patch_size
        
        # 确保尺寸能被patch_size整除
        if H % P != 0 or W % P != 0:
            # 填充到合适的尺寸
            pad_h = (P - H % P) % P
            pad_w = (P - W % P) % P
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2:]
        
        # 重组为patches
        # [B, C, H, W] -> [B, C, H//P, P, W//P, P] -> [B, C, H//P * W//P, P*P]
        patches = x.unfold(2, P, P).unfold(3, P, P)  # [B, C, H//P, W//P, P, P]
        patches = patches.contiguous().view(B, C, -1, P * P)  # [B, C, N, P*P]
        
        return patches
    
    def _fold(self, patches: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """
        将patches重组为特征图
        
        Args:
            patches: [B, C, N, P]
            target_h, target_w: 目标尺寸
            
        Returns:
            x: [B, C, H, W]
        """
        B, C, N, P_sq = patches.shape
        P = int(math.sqrt(P_sq))
        
        # 计算实际的H, W
        H_patches = int(math.sqrt(N))
        W_patches = N // H_patches
        H = H_patches * P
        W = W_patches * P
        
        # 重组patches
        # [B, C, N, P*P] -> [B, C, H//P, W//P, P, P] -> [B, C, H, W]
        patches = patches.view(B, C, H_patches, W_patches, P, P)
        x = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, C, H, W)
        
        # 如果需要，裁剪到目标尺寸
        if H > target_h or W > target_w:
            x = x[:, :, :target_h, :target_w]
        
        return x


class LightweightTransformerLayer(nn.Module):
    """
    轻量化Transformer层
    
    通道变化：
    input: [B, C, N, P] -> output: [B, C, N, P] (通道数保持不变)
    """
    
    def __init__(self, 
                 dim: int,
                 num_heads: int = 4,
                 mlp_ratio: float = 2.0,
                 dropout: float = 0.0):
        super(LightweightTransformerLayer, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 多头注意力
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, C, N, P] - patches
            
        Returns:
            output: [B, C, N, P]
        """
        B, C, N, P = x.shape
        
        # 重组为序列格式
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(B * N, P, C)  # [B*N, P, C]
        
        # 多头自注意力
        shortcut = x_seq
        x_seq = self.norm1(x_seq)
        
        # QKV计算
        qkv = self.qkv(x_seq).reshape(B * N, P, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*N, num_heads, P, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*N, num_heads, P, P]
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # 应用注意力
        x_seq = (attn @ v).transpose(1, 2).reshape(B * N, P, C)  # [B*N, P, C]
        x_seq = self.proj(x_seq)
        x_seq = self.proj_dropout(x_seq)
        
        # 残差连接
        x_seq = shortcut + x_seq
        
        # MLP
        x_seq = x_seq + self.mlp(self.norm2(x_seq))
        
        # 重组回patches格式
        output = x_seq.view(B, N, P, C).permute(0, 3, 1, 2)  # [B, C, N, P]
        
        return output


class ChannelAdaptiveMobileViT(nn.Module):
    """
    通道自适应MobileViT模块
    
    针对不同通道数自动调整transformer维度
    通道变化：[B, C_in, H, W] -> [B, C_in, H, W] (完全保持通道数不变)
    """
    
    def __init__(self, in_channels: int):
        super(ChannelAdaptiveMobileViT, self).__init__()
        
        self.in_channels = in_channels
        
        # 根据输入通道数自适应调整参数
        if in_channels <= 128:
            # 小通道数：更轻量的配置
            transformer_dim = in_channels * 2
            num_heads = 2
            num_layers = 1
            patch_size = 4
        elif in_channels <= 512:
            # 中等通道数：平衡配置
            transformer_dim = in_channels * 2
            num_heads = 4
            num_layers = 2
            patch_size = 2
        else:
            # 大通道数：更强的配置
            transformer_dim = min(in_channels * 2, 512)  # 限制最大维度
            num_heads = 8
            num_layers = 2
            patch_size = 2
        
        self.mobilevit_block = MobileViTBlock(
            in_channels=in_channels,
            transformer_dim=transformer_dim,
            patch_size=patch_size,
            num_transformer_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=2.0,
            dropout=0.1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mobilevit_block(x)


class MultiscaleMobileViTEnhancement(nn.Module):
    """
    多尺度MobileViT增强模块
    
    完整的通道变化流程：
    输入: [
        [B, C1, H1, W1],  # P3层特征
        [B, C2, H2, W2],  # P4层特征  
        [B, C3, H3, W3]   # P5层特征
    ]
    输出: [
        [B, C1, H1, W1],  # 增强后P3层特征 (通道数不变)
        [B, C2, H2, W2],  # 增强后P4层特征 (通道数不变)
        [B, C3, H3, W3]   # 增强后P5层特征 (通道数不变)
    ]
    """
    
    def __init__(self, 
                 in_channels_list: List[int] = [256, 512, 1024],  # P3, P4, P5层的通道数
                 enable_cross_scale_fusion: bool = True):
        super(MultiscaleMobileViTEnhancement, self).__init__()
        
        self.in_channels_list = in_channels_list
        self.enable_cross_scale_fusion = enable_cross_scale_fusion
        
        # 为每个尺度创建MobileViT模块
        self.mobilevit_modules = nn.ModuleList([
            ChannelAdaptiveMobileViT(ch) for ch in in_channels_list
        ])
        
        # 跨尺度特征融合（可选）
        if enable_cross_scale_fusion:
            self.cross_scale_fusion = CrossScaleFusion(in_channels_list)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 多尺度特征图列表 [
                [B, C1, H1, W1],  # P3
                [B, C2, H2, W2],  # P4
                [B, C3, H3, W3]   # P5
            ]
            
        Returns:
            enhanced_features: 增强后的特征图列表，通道数完全不变
        """
        # Step 1: 对每个尺度应用MobileViT增强
        enhanced_features = []
        for i, feat in enumerate(features):
            enhanced = self.mobilevit_modules[i](feat)  # 通道数保持不变
            enhanced_features.append(enhanced)
        
        # Step 2: 跨尺度特征融合（可选）
        if self.enable_cross_scale_fusion:
            enhanced_features = self.cross_scale_fusion(enhanced_features)
        
        return enhanced_features


class CrossScaleFusion(nn.Module):
    """
    跨尺度特征融合模块
    通过上下采样实现不同尺度特征的信息交换
    """
    
    def __init__(self, in_channels_list: List[int]):
        super(CrossScaleFusion, self).__init__()
        
        self.in_channels_list = in_channels_list
        
        # 特征对齐卷积
        self.align_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True)
            ) for ch in in_channels_list
        ])
        
        # 融合权重
        self.fusion_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in in_channels_list
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        跨尺度特征融合
        
        Args:
            features: [P3, P4, P5] 特征列表
            
        Returns:
            fused_features: 融合后的特征列表，通道数不变
        """
        # 获取每个尺度的目标尺寸
        target_sizes = [feat.shape[2:] for feat in features]
        
        fused_features = []
        
        for i, feat in enumerate(features):
            target_size = target_sizes[i]
            fused_feat = feat * self.fusion_weights[i]
            
            # 收集其他尺度的特征
            for j, other_feat in enumerate(features):
                if i != j:
                    # 调整空间尺寸
                    if other_feat.shape[2:] != target_size:
                        resized_feat = F.interpolate(
                            other_feat, size=target_size, 
                            mode='bilinear', align_corners=False
                        )
                    else:
                        resized_feat = other_feat
                    
                    # 调整通道数（如果不同）
                    if resized_feat.shape[1] != feat.shape[1]:
                        # 使用1x1卷积进行通道适配
                        if not hasattr(self, f'channel_adapter_{j}_to_{i}'):
                            # 动态创建通道适配器
                            adapter = nn.Conv2d(
                                resized_feat.shape[1], 
                                feat.shape[1], 
                                kernel_size=1, 
                                bias=False
                            ).to(resized_feat.device)
                            setattr(self, f'channel_adapter_{j}_to_{i}', adapter)
                        else:
                            adapter = getattr(self, f'channel_adapter_{j}_to_{i}')
                        
                        resized_feat = adapter(resized_feat)
                    
                    # 加权融合
                    weight = self.fusion_weights[j] * 0.1  # 降低其他尺度的权重
                    fused_feat = fused_feat + resized_feat * weight
            
            # 应用对齐卷积
            fused_feat = self.align_convs[i](fused_feat)
            fused_features.append(fused_feat)
        
        return fused_features 