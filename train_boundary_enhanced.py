"""
基于原始train.py的YOLO11n边界增强训练脚本
保持与原始训练脚本相同的简洁性，但集成边界增强模块

使用方法：
python train_boundary_enhanced.py
"""

from ultralytics import YOLO
import torch
import torch.nn as nn
from pathlib import Path
import sys

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入边界增强模块
from boundary_enhancement_module import AdvancedBoundaryEnhancementModule


class YOLO11nWithBoundaryEnhancement:
    """
    简化的YOLO11n边界增强包装器
    直接在标准YOLO11n基础上添加边界增强功能
    """
    
    def __init__(self, model_path="yolo11n.pt"):
        print("=" * 50)
        print("🚀 加载YOLO11n + 边界增强模块")
        print("=" * 50)
        
        # 加载标准YOLO11n模型
        self.yolo = YOLO(model_path)
        self.original_model = self.yolo.model
        
        # 分析模型结构，找到P3, P4, P5层
        self._analyze_model_structure()
        
        # 创建边界增强模块
        self._add_boundary_enhancement()
        
        # 修改模型前向传播
        self._modify_forward_pass()
        
        print("✅ 边界增强模块集成完成!")
        
    def _analyze_model_structure(self):
        """分析YOLO11n模型结构"""
        print("📊 分析YOLO11n模型结构...")
        
        # YOLO11n的标准通道配置
        self.backbone_channels = [256, 512, 1024]  # P3, P4, P5
        
        # 找到检测头
        self.detection_head = None
        for module in self.original_model.modules():
            if hasattr(module, 'cv2') and hasattr(module, 'cv3'):
                self.detection_head = module
                break
                
        if self.detection_head is None:
            raise ValueError("未找到YOLO检测头")
            
        print(f"   P3通道: {self.backbone_channels[0]}")
        print(f"   P4通道: {self.backbone_channels[1]}")  
        print(f"   P5通道: {self.backbone_channels[2]}")
        
    def _add_boundary_enhancement(self):
        """添加边界增强模块"""
        print("🔧 创建边界增强模块...")
        
        # 创建边界增强模块，保持原始通道数
        self.boundary_enhancement = AdvancedBoundaryEnhancementModule(
            in_channels_list=self.backbone_channels,
            out_channels=None,  # 保持原始通道数
            reduction=16
        )
        
        # 将边界增强模块添加到原始模型中
        self.original_model.boundary_enhancement = self.boundary_enhancement
        
    def _modify_forward_pass(self):
        """修改模型的前向传播以集成边界增强"""
        print("⚡ 修改前向传播路径...")
        
        # 保存原始前向传播方法
        original_forward = self.detection_head.forward
        
        def enhanced_forward(x):
            """增强的前向传播"""
            # Step 1: 边界增强
            if hasattr(self.original_model, 'boundary_enhancement'):
                enhanced_features = self.original_model.boundary_enhancement(x)
            else:
                enhanced_features = x
                
            # Step 2: 使用原始检测头处理增强后的特征
            return original_forward(enhanced_features)
        
        # 替换检测头的前向传播
        self.detection_head.forward = enhanced_forward
        
    def train(self, **kwargs):
        """训练方法，与标准YOLO接口完全一致"""
        print("🎯 开始边界增强训练...")
        
        # 设置不同的学习率（如果有优化器参数）
        if 'lr0' not in kwargs:
            kwargs['lr0'] = 0.001  # 适合边界增强的学习率
            
        if 'optimizer' not in kwargs:
            kwargs['optimizer'] = 'AdamW'  # 推荐的优化器
            
        # 使用原始YOLO训练方法
        results = self.yolo.train(**kwargs)
        
        print("✅ 边界增强训练完成!")
        return results
    
    def val(self, **kwargs):
        """验证方法"""
        return self.yolo.val(**kwargs)
    
    def predict(self, **kwargs):
        """预测方法"""
        return self.yolo.predict(**kwargs)
    
    @property
    def model(self):
        """获取模型"""
        return self.yolo.model


def main():
    """主训练函数 - 与原始train.py保持相同的简洁性"""
    print("🎯 YOLO11n边界增强训练开始")
    
    # 创建边界增强模型（自动加载yolo11n.pt）
    model = YOLO11nWithBoundaryEnhancement("./yolo11n.pt")
    
    # 训练配置 - 与原始train.py保持一致
    results = model.train(
        data="./VOC.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=24,
        optimizer='AdamW', 
        name='voc_boundary_enhanced'
    )
    
    print("🎉 训练完成!")
    print(f"📊 结果: {results}")
    
    # 可选：运行验证
    print("\n📈 运行验证...")
    val_results = model.val()
    print(f"📊 验证结果: {val_results}")


if __name__ == '__main__':
    # 检查依赖
    try:
        from boundary_enhancement_module import AdvancedBoundaryEnhancementModule
        print("✅ 边界增强模块导入成功")
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保 boundary_enhancement_module.py 在同一目录下")
        sys.exit(1)
    
    # 开始训练
    main() 