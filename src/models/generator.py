#!/usr/bin/env python3
"""
最小可行深度精修生成器
基于U-Net架构的轻量化深度图精修生成器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """基础卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, use_bn: bool = True, 
                 activation: str = 'relu'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=3, stride=stride, 
                             padding=1, bias=not use_bn)
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class DepthRGBFusionBlock(nn.Module):
    """深度-RGB特征融合模块 (简化版几何感知)"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        # 分别处理深度和RGB特征
        self.depth_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # 深度卷积
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制 (简化版)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, depth_feat: torch.Tensor, rgb_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_feat: 深度特征 [B, C, H, W]
            rgb_feat: RGB特征 [B, C, H, W]
        """
        # 分支处理
        depth_enhanced = self.depth_branch(depth_feat)
        rgb_enhanced = self.rgb_branch(rgb_feat)
        
        # 特征融合
        fused = torch.cat([depth_enhanced, rgb_enhanced], dim=1)
        fused = self.fusion(fused)
        
        # 注意力加权
        attention_weights = self.attention(fused)
        attended_features = fused * attention_weights
        
        return attended_features


class MinimalDepthGenerator(nn.Module):
    """
    最小可行深度精修生成器
    
    架构：简化的U-Net + 深度-RGB融合
    目标：轻量、快速、能工作
    """
    
    def __init__(self, 
                 input_channels: int = 4,  # 1(depth) + 3(RGB)
                 output_channels: int = 1, 
                 base_channels: int = 32,  # 减小通道数以提高速度
                 num_levels: int = 4):     # 减少层数
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        
        # 输入处理
        self.input_conv = ConvBlock(input_channels, base_channels)
        
        # 编码器
        self.encoders = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_levels):
            out_ch = base_channels * (2 ** i)
            stride = 2 if i > 0 else 1  # 第一层不下采样
            encoder_block = nn.Sequential(
                ConvBlock(in_ch, out_ch, stride=stride),
                ConvBlock(out_ch, out_ch)
            )
            self.encoders.append(encoder_block)
            in_ch = out_ch
        
        # 瓶颈层 + 深度-RGB融合
        bottleneck_channels = base_channels * (2 ** (num_levels - 1))
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_channels, bottleneck_channels),
            ConvBlock(bottleneck_channels, bottleneck_channels)
        )
        
        # RGB特征提取器 (用于瓶颈层融合)
        self.rgb_feature_extractor = nn.Sequential(
            nn.Conv2d(3, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((bottleneck_channels // base_channels, bottleneck_channels // base_channels))  # 自适应池化到合适尺寸
        )
        
        # RGB特征映射到瓶颈层通道数
        self.rgb_feature_mapper = nn.Sequential(
            nn.Conv2d(base_channels, bottleneck_channels, 1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        # 深度-RGB融合模块
        self.fusion_block = DepthRGBFusionBlock(bottleneck_channels)
        
        # 解码器
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            if i == num_levels - 1:
                in_ch = bottleneck_channels
            else:
                in_ch = base_channels * (2 ** (i + 1))
            
            out_ch = base_channels * (2 ** i) if i > 0 else base_channels
            
            # 上采样 + 跳跃连接处理
            decoder_block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ConvBlock(out_ch * 2, out_ch),  # *2 因为跳跃连接
                ConvBlock(out_ch, out_ch)
            )
            self.decoders.append(decoder_block)
        
        # 输出层
        self.output_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels // 2),
            nn.Conv2d(base_channels // 2, output_channels, 3, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]，后续可以缩放
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, noisy_depth: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            noisy_depth: 噪声深度图 [B, 1, H, W]
            rgb: RGB图像 [B, 3, H, W]
            
        Returns:
            refined_depth: 精修深度图 [B, 1, H, W]
        """
        # 输入处理
        x = torch.cat([noisy_depth, rgb], dim=1)  # [B, 4, H, W]
        x = self.input_conv(x)
        
        # 编码器 + 跳跃连接存储
        skip_connections = []
        for encoder in self.encoders:
            skip_connections.append(x)
            x = encoder(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # RGB特征提取并融合
        rgb_feat = self.rgb_feature_extractor(rgb)
        x = self.fusion_block(x, rgb_feat)
        
        # 解码器
        for i, decoder in enumerate(self.decoders):
            # 上采样
            if i == 0:
                # 第一个解码器直接处理瓶颈层输出
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
                x = decoder[1:](x)  # 跳过ConvTranspose2d
            else:
                x = decoder[:3](x)  # ConvTranspose2d + BN + ReLU
            
            # 跳跃连接
            skip_idx = self.num_levels - 1 - i
            if skip_idx >= 0 and skip_idx < len(skip_connections):
                skip = skip_connections[skip_idx]
                
                # 确保尺寸匹配
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
                
                x = torch.cat([x, skip], dim=1)
                x = decoder[3:](x)  # 后续卷积层
        
        # 输出层
        refined_depth = self.output_conv(x)
        
        # 残差连接 (将精修结果与原始深度相加)
        if refined_depth.shape == noisy_depth.shape:
            refined_depth = noisy_depth + refined_depth
        
        return refined_depth
    
    def get_model_size(self) -> Tuple[int, float]:
        """获取模型大小信息"""
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)  # 假设float32
        return param_count, param_size_mb


# ============================================================================
# 轻量化版本
# ============================================================================

class UltraLightDepthGenerator(nn.Module):
    """
    超轻量深度精修生成器 - 修复版
    简化架构，确保通道数完全匹配
    """
    
    def __init__(self):
        super().__init__()
        
        # 简单的编码-解码结构
        self.encoder = nn.Sequential(
            # 输入: [B, 4, H, W] (1 depth + 3 RGB)
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # [B, 32, H, W]
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # [B, 64, H/2, W/2]
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # [B, 128, H/4, W/4]
        )
        
        self.decoder = nn.Sequential(
            # [B, 128, H/4, W/4]
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # [B, 64, H/2, W/2]
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # [B, 32, H, W]
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
            # [B, 1, H, W]
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, noisy_depth, rgb):
        """前向传播"""
        # 合并输入
        x = torch.cat([noisy_depth, rgb], dim=1)  # [B, 4, H, W]
        
        # 编码
        encoded = self.encoder(x)  # [B, 128, H/4, W/4]
        
        # 解码
        decoded = self.decoder(encoded)  # [B, 1, H, W]
        
        # 残差连接
        output = noisy_depth + 0.1 * decoded  # 小的残差权重
        
        return output
    
    def get_model_size(self):
        """获取模型大小"""
        param_count = sum(p.numel() for p in self.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)
        return param_count, param_size_mb


class StandardDepthGenerator(UltraLightDepthGenerator):
    """标准版本 - 暂时使用相同的简化架构"""
    
    def __init__(self):
        super().__init__()


# ============================================================================
# 测试和验证函数
# ============================================================================

def test_generator():
    """测试生成器"""
    print("🧪 测试深度精修生成器...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试参数
    batch_size = 2
    H, W = 256, 256
    
    print("1. 测试超轻量化版本...")
    ultra_light_gen = UltraLightDepthGenerator().to(device)
    param_count, param_size = ultra_light_gen.get_model_size()
    print(f"   参数数量: {param_count:,}")
    print(f"   模型大小: {param_size:.2f} MB")
    
    # 测试前向传播
    noisy_depth = torch.rand(batch_size, 1, H, W, device=device)
    rgb = torch.rand(batch_size, 3, H, W, device=device)
    
    with torch.no_grad():
        output = ultra_light_gen(noisy_depth, rgb)
    
    print(f"   ✅ 输入形状: depth{noisy_depth.shape}, rgb{rgb.shape}")
    print(f"   ✅ 输出形状: {output.shape}")
    assert output.shape == noisy_depth.shape, "输出形状应与输入深度图相同"
    
    print("2. 测试标准版本...")
    standard_gen = StandardDepthGenerator().to(device)
    param_count, param_size = standard_gen.get_model_size()
    print(f"   参数数量: {param_count:,}")
    print(f"   模型大小: {param_size:.2f} MB")
    
    with torch.no_grad():
        output = standard_gen(noisy_depth, rgb)
    print(f"   ✅ 输出形状: {output.shape}")
    
    print("3. 测试梯度传播...")
    noisy_depth.requires_grad_(True)
    output = ultra_light_gen(noisy_depth, rgb)
    loss = output.mean()
    loss.backward()
    
    assert noisy_depth.grad is not None, "梯度应该不为空"
    grad_norm = noisy_depth.grad.norm().item()
    print(f"   ✅ 梯度范数: {grad_norm:.6f}")
    
    print("4. 测试不同输入尺寸...")
    test_sizes = [(128, 128), (256, 256), (512, 512)]
    
    for h, w in test_sizes:
        test_depth = torch.rand(1, 1, h, w, device=device)
        test_rgb = torch.rand(1, 3, h, w, device=device)
        
        with torch.no_grad():
            test_output = ultra_light_gen(test_depth, test_rgb)
        
        print(f"   ✅ {h}x{w}: {test_output.shape}")
        assert test_output.shape == test_depth.shape
    
    print("\n🎉 生成器测试通过!")
    return True


def benchmark_generator():
    """性能基准测试"""
    print("\n🚀 生成器性能基准测试...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    models = {
        'UltraLight': UltraLightDepthGenerator().to(device),
        'Standard': StandardDepthGenerator().to(device)
    }
    
    # 测试不同分辨率
    resolutions = [(128, 128), (256, 256), (512, 512)]
    batch_size = 4
    
    for model_name, model in models.items():
        print(f"\n📊 {model_name} 模型:")
        param_count, param_size = model.get_model_size()
        print(f"   参数: {param_count:,} ({param_size:.2f} MB)")
        
        model.eval()
        
        for H, W in resolutions:
            print(f"\n   分辨率 {H}x{W}:")
            
            # 准备数据
            noisy_depth = torch.rand(batch_size, 1, H, W, device=device)
            rgb = torch.rand(batch_size, 3, H, W, device=device)
            
            # 预热
            with torch.no_grad():
                for _ in range(5):
                    _ = model(noisy_depth, rgb)
            
            # 基准测试
            times = []
            for _ in range(20):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                with torch.no_grad():
                    output = model(noisy_depth, rgb)
                end_time.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                times.append(elapsed_time)
            
            avg_time = np.mean(times[5:])  # 排除前5次
            fps = 1000 / avg_time
            
            print(f"     推理时间: {avg_time:.2f} ms")
            print(f"     FPS: {fps:.1f}")
            
            # 内存使用情况
            if device == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024
                print(f"     GPU内存: {memory_used:.1f} MB")
                torch.cuda.reset_peak_memory_stats()


def compare_with_baseline():
    """与基线方法对比"""
    print("\n📈 与基线方法对比...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建测试数据
    batch_size = 4
    H, W = 256, 256
    
    noisy_depth = torch.rand(batch_size, 1, H, W, device=device)
    rgb = torch.rand(batch_size, 3, H, W, device=device)
    
    # 基线方法：简单双线性上采样
    def baseline_method(depth, rgb):
        return F.interpolate(depth, scale_factor=1.0, mode='bilinear')
    
    # 我们的方法
    our_model = UltraLightDepthGenerator().to(device)
    
    print("比较推理时间:")
    
    # 基线方法计时
    times = []
    for _ in range(100):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            _ = baseline_method(noisy_depth, rgb)
        end_time.record()
        
        torch.cuda.synchronize()
        times.append(start_time.elapsed_time(end_time))
    
    baseline_time = np.mean(times[10:])
    print(f"   基线方法: {baseline_time:.2f} ms")
    
    # 我们的方法计时
    times = []
    for _ in range(100):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            _ = our_model(noisy_depth, rgb)
        end_time.record()
        
        torch.cuda.synchronize()
        times.append(start_time.elapsed_time(end_time))
    
    our_time = np.mean(times[10:])
    print(f"   我们的方法: {our_time:.2f} ms")
    print(f"   速度比: {our_time/baseline_time:.1f}x")


if __name__ == "__main__":
    test_generator()
    benchmark_generator()
    compare_with_baseline()