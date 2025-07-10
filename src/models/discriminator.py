#!/usr/bin/env python3
"""
多尺度PatchGAN判别器
基于PatchGAN的条件判别器，支持多尺度判别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class SpectralNorm(nn.Module):
    """谱归一化模块 - 提高训练稳定性"""
    
    def __init__(self, module: nn.Module, name: str = 'weight', power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        if not self._made_params():
            self._make_params()
    
    def _update_u_v(self):
        """更新奇异向量"""
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = F.normalize(torch.mv(w.view(height, -1).data, v.data))
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    
    def _made_params(self):
        """检查是否已创建参数"""
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        """创建奇异向量参数"""
        w = getattr(self.module, self.name)
        
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data)
        v.data = F.normalize(v.data)
        w_bar = nn.Parameter(w.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def spectral_norm(module: nn.Module) -> nn.Module:
    """为模块添加谱归一化"""
    return SpectralNorm(module)


class PatchGANDiscriminator(nn.Module):
    """单尺度PatchGAN判别器"""
    
    def __init__(self, 
                 input_channels: int = 4,  # 1(depth) + 3(RGB)
                 base_channels: int = 64,
                 num_layers: int = 3,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # 构建判别器层
        layers = []
        
        # 第一层：不使用BatchNorm
        conv = nn.Conv2d(input_channels, base_channels, 4, stride=2, padding=1)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        layers.extend([
            conv,
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # 中间层
        in_channels = base_channels
        for i in range(1, num_layers):
            out_channels = min(base_channels * (2 ** i), 512)  # 限制最大通道数
            stride = 2 if i < num_layers - 1 else 1  # 最后一层不下采样
            
            conv = nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            
            layers.extend([
                conv,
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            in_channels = out_channels
        
        # 输出层
        final_conv = nn.Conv2d(in_channels, 1, 4, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        layers.append(final_conv)
        
        self.main = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入 [B, C, H, W]
            
        Returns:
            output: 判别结果 [B, 1, H', W']
        """
        return self.main(x)


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度PatchGAN判别器
    
    在不同尺度上进行判别，提高判别器的感受野和判别能力
    """
    
    def __init__(self, 
                 input_channels: int = 4,
                 base_channels: int = 64,
                 num_scales: int = 3,
                 num_layers: int = 3,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.num_scales = num_scales
        
        # 创建多个尺度的判别器
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            disc = PatchGANDiscriminator(
                input_channels=input_channels,
                base_channels=base_channels,
                num_layers=num_layers,
                use_spectral_norm=use_spectral_norm
            )
            self.discriminators.append(disc)
        
        # 下采样层
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, depth: torch.Tensor, rgb: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            depth: 深度图 [B, 1, H, W]
            rgb: RGB图像 [B, 3, H, W]
            
        Returns:
            results: 多尺度判别结果列表
        """
        # 拼接输入
        x = torch.cat([depth, rgb], dim=1)  # [B, 4, H, W]
        
        results = []
        
        for i, discriminator in enumerate(self.discriminators):
            # 判别当前尺度
            result = discriminator(x)
            results.append(result)
            
            # 下采样到下一个尺度
            if i < self.num_scales - 1:
                x = self.downsample(x)
        
        return results
    
    def get_scale_info(self) -> List[Tuple[int, int]]:
        """获取各尺度的信息"""
        # 假设输入尺寸为256x256
        scales = []
        h, w = 256, 256
        
        for i in range(self.num_scales):
            scales.append((h, w))
            h = h // 2
            w = w // 2
        
        return scales


class ConditionalDiscriminator(MultiScaleDiscriminator):
    """
    条件判别器 - 专门为深度精修任务设计
    
    除了判别真假，还考虑深度-RGB的一致性
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 添加额外的一致性判别分支
        self.consistency_branch = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, depth: torch.Tensor, rgb: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        前向传播
        
        Returns:
            multi_scale_results: 多尺度判别结果
            consistency_score: 一致性分数 [B, 1, H, W]
        """
        # 多尺度判别
        multi_scale_results = super().forward(depth, rgb)
        
        # 一致性判别
        x = torch.cat([depth, rgb], dim=1)
        consistency_score = self.consistency_branch(x)
        
        return multi_scale_results, consistency_score


# ============================================================================
# 轻量化版本
# ============================================================================

class LightweightDiscriminator(MultiScaleDiscriminator):
    """轻量化判别器 - 适用于资源受限环境"""
    
    def __init__(self):
        super().__init__(
            input_channels=4,
            base_channels=32,  # 减小通道数
            num_scales=2,      # 减少尺度数
            num_layers=2,      # 减少层数
            use_spectral_norm=False  # 关闭谱归一化以减少计算
        )


class StandardDiscriminator(MultiScaleDiscriminator):
    """标准判别器 - 平衡性能和质量"""
    
    def __init__(self):
        super().__init__(
            input_channels=4,
            base_channels=64,
            num_scales=3,
            num_layers=3,
            use_spectral_norm=True
        )


# ============================================================================
# 测试和验证函数
# ============================================================================

def test_discriminator():
    """测试判别器"""
    print("🧪 测试多尺度判别器...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试参数
    batch_size = 2
    H, W = 256, 256
    
    print("1. 测试轻量化版本...")
    light_disc = LightweightDiscriminator().to(device)
    param_count = sum(p.numel() for p in light_disc.parameters())
    param_size = param_count * 4 / (1024 * 1024)
    print(f"   参数数量: {param_count:,}")
    print(f"   模型大小: {param_size:.2f} MB")
    
    # 测试前向传播
    depth = torch.rand(batch_size, 1, H, W, device=device)
    rgb = torch.rand(batch_size, 3, H, W, device=device)
    
    with torch.no_grad():
        results = light_disc(depth, rgb)
    
    print(f"   ✅ 输入形状: depth{depth.shape}, rgb{rgb.shape}")
    print(f"   ✅ 输出尺度数: {len(results)}")
    for i, result in enumerate(results):
        print(f"   ✅ 尺度{i}: {result.shape}")
    
    print("2. 测试标准版本...")
    standard_disc = StandardDiscriminator().to(device)
    param_count = sum(p.numel() for p in standard_disc.parameters())
    param_size = param_count * 4 / (1024 * 1024)
    print(f"   参数数量: {param_count:,}")
    print(f"   模型大小: {param_size:.2f} MB")
    
    with torch.no_grad():
        results = standard_disc(depth, rgb)
    print(f"   ✅ 输出尺度数: {len(results)}")
    for i, result in enumerate(results):
        print(f"   ✅ 尺度{i}: {result.shape}")
    
    print("3. 测试条件判别器...")
    cond_disc = ConditionalDiscriminator().to(device)
    
    with torch.no_grad():
        multi_results, consistency = cond_disc(depth, rgb)
    
    print(f"   ✅ 多尺度结果数: {len(multi_results)}")
    print(f"   ✅ 一致性分数形状: {consistency.shape}")
    
    print("4. 测试梯度传播...")
    depth.requires_grad_(True)
    rgb.requires_grad_(True)
    
    results = light_disc(depth, rgb)
    loss = sum(result.mean() for result in results)
    loss.backward()
    
    assert depth.grad is not None and rgb.grad is not None, "梯度应该不为空"
    depth_grad_norm = depth.grad.norm().item()
    rgb_grad_norm = rgb.grad.norm().item()
    print(f"   ✅ 深度梯度范数: {depth_grad_norm:.6f}")
    print(f"   ✅ RGB梯度范数: {rgb_grad_norm:.6f}")
    
    print("5. 测试不同输入尺寸...")
    test_sizes = [(128, 128), (256, 256), (512, 512)]
    
    for h, w in test_sizes:
        test_depth = torch.rand(1, 1, h, w, device=device)
        test_rgb = torch.rand(1, 3, h, w, device=device)
        
        with torch.no_grad():
            test_results = light_disc(test_depth, test_rgb)
        
        print(f"   ✅ {h}x{w}: {len(test_results)} 个尺度")
        for i, result in enumerate(test_results):
            print(f"       尺度{i}: {result.shape}")
    
    print("\n🎉 判别器测试通过!")
    return True


def test_spectral_norm():
    """测试谱归一化"""
    print("\n🧪 测试谱归一化...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建带谱归一化的卷积层
    conv = nn.Conv2d(4, 64, 4, stride=2, padding=1)
    sn_conv = spectral_norm(conv).to(device)
    
    # 测试前向传播
    x = torch.rand(2, 4, 64, 64, device=device)
    
    print("1. 测试谱归一化前向传播...")
    with torch.no_grad():
        output = sn_conv(x)
    print(f"   ✅ 输出形状: {output.shape}")
    
    print("2. 检查权重谱范数...")
    # 获取权重
    weight = getattr(sn_conv.module, 'weight_bar')
    u = getattr(sn_conv.module, 'weight_u')
    v = getattr(sn_conv.module, 'weight_v')
    
    # 计算谱范数
    height = weight.shape[0]
    sigma = u.dot(weight.view(height, -1).mv(v))
    print(f"   ✅ 权重谱范数: {sigma.item():.6f}")
    
    # 谱归一化后的谱范数应该接近1
    normalized_weight = getattr(sn_conv.module, 'weight')
    normalized_sigma = torch.svd(normalized_weight.view(height, -1))[1][0]
    print(f"   ✅ 归一化后谱范数: {normalized_sigma.item():.6f}")
    
    assert abs(normalized_sigma.item() - 1.0) < 0.1, "谱归一化后的谱范数应该接近1"
    
    print("   ✅ 谱归一化工作正常!")


def benchmark_discriminator():
    """性能基准测试"""
    print("\n🚀 判别器性能基准测试...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    models = {
        'Lightweight': LightweightDiscriminator().to(device),
        'Standard': StandardDiscriminator().to(device),
        'Conditional': ConditionalDiscriminator().to(device)
    }
    
    # 测试不同分辨率
    resolutions = [(128, 128), (256, 256), (512, 512)]
    batch_size = 4
    
    for model_name, model in models.items():
        print(f"\n📊 {model_name} 判别器:")
        param_count = sum(p.numel() for p in model.parameters())
        param_size = param_count * 4 / (1024 * 1024)
        print(f"   参数: {param_count:,} ({param_size:.2f} MB)")
        
        model.eval()
        
        for H, W in resolutions:
            print(f"\n   分辨率 {H}x{W}:")
            
            # 准备数据
            depth = torch.rand(batch_size, 1, H, W, device=device)
            rgb = torch.rand(batch_size, 3, H, W, device=device)
            
            # 预热
            with torch.no_grad():
                for _ in range(5):
                    if isinstance(model, ConditionalDiscriminator):
                        _, _ = model(depth, rgb)
                    else:
                        _ = model(depth, rgb)
            
            # 基准测试
            times = []
            for _ in range(20):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                with torch.no_grad():
                    if isinstance(model, ConditionalDiscriminator):
                        results, consistency = model(depth, rgb)
                    else:
                        results = model(depth, rgb)
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


def test_adversarial_loss():
    """测试对抗损失计算"""
    print("\n🧪 测试对抗损失计算...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建判别器
    discriminator = StandardDiscriminator().to(device)
    
    # 准备数据
    batch_size = 2
    H, W = 256, 256
    
    real_depth = torch.rand(batch_size, 1, H, W, device=device)
    fake_depth = torch.rand(batch_size, 1, H, W, device=device)
    rgb = torch.rand(batch_size, 3, H, W, device=device)
    
    print("1. 测试真实样本判别...")
    with torch.no_grad():
        real_results = discriminator(real_depth, rgb)
    
    print(f"   ✅ 真实样本输出尺度: {len(real_results)}")
    for i, result in enumerate(real_results):
        mean_score = torch.sigmoid(result).mean().item()
        print(f"   ✅ 尺度{i}平均分数: {mean_score:.3f}")
    
    print("2. 测试虚假样本判别...")
    with torch.no_grad():
        fake_results = discriminator(fake_depth, rgb)
    
    print(f"   ✅ 虚假样本输出尺度: {len(fake_results)}")
    for i, result in enumerate(fake_results):
        mean_score = torch.sigmoid(result).mean().item()
        print(f"   ✅ 尺度{i}平均分数: {mean_score:.3f}")
    
    print("3. 计算对抗损失...")
    
    # LSGAN损失
    def lsgan_loss(pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)
    
    # 判别器损失
    d_real_loss = sum(lsgan_loss(pred, True) for pred in real_results) / len(real_results)
    d_fake_loss = sum(lsgan_loss(pred, False) for pred in fake_results) / len(fake_results)
    d_total_loss = (d_real_loss + d_fake_loss) * 0.5
    
    print(f"   ✅ 判别器真实损失: {d_real_loss.item():.6f}")
    print(f"   ✅ 判别器虚假损失: {d_fake_loss.item():.6f}")
    print(f"   ✅ 判别器总损失: {d_total_loss.item():.6f}")
    
    # 生成器损失
    g_loss = sum(lsgan_loss(pred, True) for pred in fake_results) / len(fake_results)
    print(f"   ✅ 生成器对抗损失: {g_loss.item():.6f}")
    
    print("   ✅ 对抗损失计算正常!")


def analyze_discriminator_receptive_field():
    """分析判别器感受野"""
    print("\n📐 分析判别器感受野...")
    
    def calculate_receptive_field(kernel_sizes, strides):
        """计算感受野大小"""
        rf = 1
        for k, s in zip(kernel_sizes, strides):
            rf = rf + (k - 1) * np.prod(strides[:len(strides)-len(strides[strides.index(s):])])
        return rf
    
    # 分析不同判别器的感受野
    configs = {
        'Lightweight': {'num_layers': 2, 'kernel_size': 4},
        'Standard': {'num_layers': 3, 'kernel_size': 4}
    }
    
    for name, config in configs.items():
        print(f"\n{name} 判别器:")
        
        # 假设stride都是2，除了最后一层是1
        strides = [2] * (config['num_layers'] - 1) + [1]
        kernel_sizes = [config['kernel_size']] * config['num_layers']
        
        # 简化的感受野计算
        rf = 1
        stride_product = 1
        for i, (k, s) in enumerate(zip(kernel_sizes, strides)):
            rf = rf + (k - 1) * stride_product
            if i < len(strides) - 1:  # 最后一层不更新stride_product
                stride_product *= s
        
        print(f"   感受野大小: {rf}x{rf} 像素")
        
        # 输出特征图大小 (假设输入256x256)
        h, w = 256, 256
        for s in strides[:-1]:  # 除了最后一层
            h //= s
            w //= s
        
        print(f"   输出特征图: {h}x{w}")
        print(f"   参数数量: ~{config['num_layers'] * 64 * 64 * config['kernel_size']**2:,}")


if __name__ == "__main__":
    test_discriminator()
    test_spectral_norm()
    benchmark_discriminator()
    test_adversarial_loss()
    analyze_discriminator_receptive_field()