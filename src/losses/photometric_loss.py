#!/usr/bin/env python3
"""
光度一致性损失 - 核心创新模块
利用多视图几何约束指导深度图精修的核心损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

# 正确的项目结构导入
try:
    # 生产环境：从src.utils导入
    from ..utils.geometry_utils import GeometryUtils
except ImportError:
    try:
        # 开发/测试环境：直接从utils导入
        from src.utils.geometry_utils import GeometryUtils
    except ImportError:
        # 最后的兜底方案：添加路径导入
        import sys
        import os
        # 添加项目根目录到路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.insert(0, project_root)
        from src.utils.geometry_utils import GeometryUtils


class PhotometricConsistencyLoss(nn.Module):
    """
    光度一致性损失 - 核心创新
    
    基本思想：
    1. 使用精修后的深度图将当前帧投影到下一帧
    2. 计算投影后的RGB与下一帧真实RGB的光度误差
    3. 通过最小化光度误差来约束深度图的几何正确性
    
    这是传统SLAM中光度误差在深度学习中的创新应用
    """
    
    def __init__(self, 
                 ssim_weight: float = 0.85,
                 l1_weight: float = 0.15,
                 edge_threshold: float = 0.1,
                 min_valid_pixels: int = 10,        # 大幅降低最小像素要求
                 occlusion_handling: bool = False,  # 默认关闭遮挡处理
                 robust_loss: bool = True):
        """
        Args:
            ssim_weight: SSIM损失权重
            l1_weight: L1损失权重  
            edge_threshold: 边缘检测阈值
            min_valid_pixels: 最小有效像素数
            occlusion_handling: 是否启用遮挡处理
            robust_loss: 是否使用鲁棒损失
        """
        super().__init__()
        
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.edge_threshold = edge_threshold
        self.min_valid_pixels = min_valid_pixels
        self.occlusion_handling = occlusion_handling
        self.robust_loss = robust_loss
        
        # SSIM损失计算
        self.ssim_loss = SSIMModule(window_size=11)
        
        # 用于遮挡检测的参数
        self.occlusion_threshold = 0.01  # 深度差异阈值
        
    def forward(self, 
                refined_depth: torch.Tensor,
                rgb_current: torch.Tensor, 
                rgb_next: torch.Tensor,
                intrinsics: torch.Tensor,
                pose_t_to_t1: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            refined_depth: 精修深度图 [B, 1, H, W]
            rgb_current: 当前帧RGB [B, 3, H, W]  
            rgb_next: 下一帧RGB [B, 3, H, W]
            intrinsics: 相机内参 [B, 3, 3]
            pose_t_to_t1: 从t到t+1的位姿变换 [B, 4, 4]
            mask: 可选的有效区域掩码 [B, 1, H, W]
            
        Returns:
            losses: 包含各种损失的字典
        """
        # 1. 深度图转3D点云
        points_3d = GeometryUtils.depth_to_pointcloud(refined_depth, intrinsics)
        
        # 2. 应用位姿变换到下一帧坐标系
        points_3d_t1 = GeometryUtils.transform_points(points_3d, pose_t_to_t1)
        
        # 3. 投影到下一帧图像平面
        projected_coords, valid_proj_mask = GeometryUtils.project_to_image(
            points_3d_t1, intrinsics
        )
        
        # 4. 处理遮挡 (如果启用)
        if self.occlusion_handling:
            occlusion_mask = self._detect_occlusions(
                refined_depth, points_3d_t1, intrinsics, projected_coords, valid_proj_mask
            )
            valid_mask = valid_proj_mask & occlusion_mask
        else:
            valid_mask = valid_proj_mask
        
        # 5. 应用用户提供的掩码
        if mask is not None:
            mask_2d = mask.squeeze(1)  # [B, H, W]
            valid_mask = valid_mask & mask_2d.bool()
        
        # 6. 检查有效像素数量
        valid_pixel_count = valid_mask.sum()
        if valid_pixel_count < self.min_valid_pixels:
            warnings.warn(f"有效像素数量过少: {valid_pixel_count} < {self.min_valid_pixels}")
            # 返回零损失但记录警告
            return {
                'photometric_loss': torch.tensor(0.0, device=refined_depth.device),
                'ssim_loss': torch.tensor(0.0, device=refined_depth.device),
                'l1_loss': torch.tensor(0.0, device=refined_depth.device),
                'valid_pixel_ratio': torch.tensor(0.0, device=refined_depth.device),
                'valid_pixel_count': valid_pixel_count.float()
            }
        
        # 7. 双线性采样当前帧RGB到下一帧
        warped_rgb = GeometryUtils.bilinear_sample(rgb_current, projected_coords, valid_mask)
        
        # 8. 计算光度损失
        photometric_losses = self._compute_photometric_loss(
            warped_rgb, rgb_next, valid_mask
        )
        
        # 9. 组合最终损失
        total_photometric_loss = (
            self.ssim_weight * photometric_losses['ssim_loss'] + 
            self.l1_weight * photometric_losses['l1_loss']
        )
        
        return {
            'photometric_loss': total_photometric_loss,
            'ssim_loss': photometric_losses['ssim_loss'],
            'l1_loss': photometric_losses['l1_loss'],
            'valid_pixel_ratio': valid_mask.float().mean(),
            'valid_pixel_count': valid_pixel_count.float()
        }
    
    def _detect_occlusions(self, 
                          depth: torch.Tensor,
                          points_3d_t1: torch.Tensor,
                          intrinsics: torch.Tensor,
                          projected_coords: torch.Tensor,
                          valid_proj_mask: torch.Tensor) -> torch.Tensor:
        """
        检测遮挡区域
        
        基本思想：如果一个3D点投影后的深度与该像素位置的深度差异很大，
        说明可能存在遮挡
        
        Args:
            depth: 原始深度图 [B, 1, H, W]
            points_3d_t1: 变换后的3D点 [B, H, W, 3]  
            intrinsics: 相机内参 [B, 3, 3]
            projected_coords: 投影坐标 [B, H, W, 2]
            valid_proj_mask: 有效投影掩码 [B, H, W]
            
        Returns:
            occlusion_mask: 非遮挡区域掩码 [B, H, W]
        """
        B, _, H, W = depth.shape
        
        # 获取投影点的深度值
        projected_depth = points_3d_t1[..., 2]  # [B, H, W]
        
        # 在下一帧深度图中采样对应位置的深度
        # 这里我们假设下一帧的深度图就是当前深度图（简化处理）
        # 在实际应用中，应该使用真实的下一帧深度图
        depth_squeezed = depth.squeeze(1)  # [B, H, W]
        sampled_depth = GeometryUtils.bilinear_sample(
            depth, projected_coords, valid_proj_mask
        ).squeeze(1)  # [B, H, W]
        
        # 计算深度差异
        depth_diff = torch.abs(projected_depth - sampled_depth)
        
        # 基于深度差异阈值判断遮挡
        occlusion_mask = depth_diff < self.occlusion_threshold
        
        return occlusion_mask & valid_proj_mask
    
    def _compute_photometric_loss(self, 
                                 warped_image: torch.Tensor,
                                 target_image: torch.Tensor, 
                                 valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算光度损失 (L1 + SSIM)
        
        Args:
            warped_image: 投影后的图像 [B, 3, H, W]
            target_image: 目标图像 [B, 3, H, W]
            valid_mask: 有效像素掩码 [B, H, W]
            
        Returns:
            losses: 包含L1和SSIM损失的字典
        """
        # 扩展掩码到通道维度
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(warped_image)
        
        # 计算L1损失
        l1_loss = F.l1_loss(warped_image, target_image, reduction='none')
        masked_l1_loss = l1_loss * valid_mask_expanded.float()
        
        if self.robust_loss:
            # 使用Huber损失替代L1，更鲁棒
            huber_loss = F.smooth_l1_loss(warped_image, target_image, reduction='none')
            masked_l1_loss = huber_loss * valid_mask_expanded.float()
        
        # 归一化L1损失
        valid_pixels = valid_mask_expanded.float().sum()
        if valid_pixels > 0:
            final_l1_loss = masked_l1_loss.sum() / valid_pixels
        else:
            final_l1_loss = torch.tensor(0.0, device=warped_image.device)
        
        # 计算SSIM损失
        try:
            ssim_loss = self.ssim_loss(warped_image, target_image, valid_mask)
        except Exception as e:
            warnings.warn(f"SSIM计算失败: {e}")
            ssim_loss = torch.tensor(0.0, device=warped_image.device)
        
        return {
            'l1_loss': final_l1_loss,
            'ssim_loss': ssim_loss
        }


class SSIMModule(nn.Module):
    """结构相似性指数损失模块"""
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size: int, sigma: float):
        """创建高斯核"""
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int):
        """创建SSIM计算窗口"""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor, 
              window: torch.Tensor, window_size: int, 
              channel: int, size_average: bool = True):
        """计算SSIM"""
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            img1, img2: 输入图像 [B, C, H, W]
            mask: 可选掩码 [B, H, W]
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        # 如果提供了掩码，先将无效区域置零
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(img1)
            img1_masked = img1 * mask_expanded.float()
            img2_masked = img2 * mask_expanded.float()
        else:
            img1_masked = img1
            img2_masked = img2
        
        ssim_value = self._ssim(img1_masked, img2_masked, window, 
                               self.window_size, channel, self.size_average)
        
        # SSIM损失 = 1 - SSIM
        return 1 - ssim_value


# ============================================================================
# 测试和验证函数
# ============================================================================

def test_photometric_loss():
    """测试光度一致性损失"""
    print("🧪 测试光度一致性损失模块...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试参数
    batch_size = 2
    H, W = 128, 128
    
    # 创建测试数据
    print("1. 准备测试数据...")
    refined_depth = torch.rand(batch_size, 1, H, W, device=device) * 5.0 + 0.5
    rgb_current = torch.rand(batch_size, 3, H, W, device=device)
    rgb_next = rgb_current + 0.1 * torch.randn_like(rgb_current)  # 相似但有差异
    
    # 相机内参
    intrinsics = GeometryUtils.create_intrinsics_matrix(
        fx=200.0, fy=200.0, cx=W/2, cy=H/2, 
        batch_size=batch_size, device=device
    )
    
    # 位姿变换 (小的平移和旋转)
    translation = torch.randn(batch_size, 3, device=device) * 0.1
    rotation_angles = torch.randn(batch_size, 3, device=device) * 0.05
    pose_transform = GeometryUtils.create_pose_matrix(translation, rotation_angles)
    
    print("2. 创建光度一致性损失模块...")
    photometric_loss = PhotometricConsistencyLoss(
        ssim_weight=0.8,
        l1_weight=0.2,
        occlusion_handling=True,
        robust_loss=True
    ).to(device)
    
    print("3. 测试前向传播...")
    with torch.no_grad():
        losses = photometric_loss(
            refined_depth=refined_depth,
            rgb_current=rgb_current,
            rgb_next=rgb_next,
            intrinsics=intrinsics,
            pose_t_to_t1=pose_transform
        )
    
    print("4. 验证输出...")
    expected_keys = ['photometric_loss', 'ssim_loss', 'l1_loss', 
                    'valid_pixel_ratio', 'valid_pixel_count']
    
    for key in expected_keys:
        assert key in losses, f"缺少输出键: {key}"
        print(f"   ✅ {key}: {losses[key].item():.6f}")
    
    # 检查损失范围合理性
    assert 0 <= losses['valid_pixel_ratio'] <= 1, "有效像素比例超出范围"
    assert losses['photometric_loss'] >= 0, "光度损失应为非负"
    assert losses['valid_pixel_count'] > 0, "应该有有效像素"
    
    print("5. 测试梯度传播...")
    refined_depth.requires_grad_(True)
    losses = photometric_loss(
        refined_depth=refined_depth,
        rgb_current=rgb_current,
        rgb_next=rgb_next,
        intrinsics=intrinsics,
        pose_t_to_t1=pose_transform
    )
    
    loss = losses['photometric_loss']
    loss.backward()
    
    assert refined_depth.grad is not None, "梯度应该不为空"
    grad_norm = refined_depth.grad.norm().item()
    print(f"   ✅ 梯度范数: {grad_norm:.6f}")
    
    print("6. 测试边界情况...")
    # 测试零深度情况
    zero_depth = torch.zeros_like(refined_depth)
    with torch.no_grad():
        zero_losses = photometric_loss(
            refined_depth=zero_depth,
            rgb_current=rgb_current,
            rgb_next=rgb_next,
            intrinsics=intrinsics,
            pose_t_to_t1=pose_transform
        )
    print(f"   ✅ 零深度有效像素比例: {zero_losses['valid_pixel_ratio'].item():.6f}")
    
    print("\n🎉 光度一致性损失模块测试通过!")
    print("\n📊 测试总结:")
    print(f"   - 光度损失: {losses['photometric_loss'].item():.6f}")
    print(f"   - SSIM损失: {losses['ssim_loss'].item():.6f}")
    print(f"   - L1损失: {losses['l1_loss'].item():.6f}")
    print(f"   - 有效像素比例: {losses['valid_pixel_ratio'].item():.3f}")
    print(f"   - 梯度传播: ✅")
    
    return True


def benchmark_photometric_loss():
    """性能基准测试"""
    print("\n🚀 光度一致性损失性能基准测试...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    photometric_loss = PhotometricConsistencyLoss().to(device)
    
    # 不同分辨率测试
    resolutions = [(64, 64), (128, 128), (256, 256)]
    batch_size = 4
    
    for H, W in resolutions:
        print(f"\n测试分辨率: {H}x{W}")
        
        # 准备数据
        refined_depth = torch.rand(batch_size, 1, H, W, device=device)
        rgb_current = torch.rand(batch_size, 3, H, W, device=device)
        rgb_next = torch.rand(batch_size, 3, H, W, device=device)
        intrinsics = GeometryUtils.create_intrinsics_matrix(
            fx=200.0, fy=200.0, cx=W/2, cy=H/2, 
            batch_size=batch_size, device=device
        )
        pose_transform = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 基准测试
        times = []
        for _ in range(20):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with torch.no_grad():
                losses = photometric_loss(
                    refined_depth, rgb_current, rgb_next, 
                    intrinsics, pose_transform
                )
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            times.append(elapsed_time)
        
        avg_time = np.mean(times[5:])  # 排除前5次预热
        print(f"   平均推理时间: {avg_time:.2f} ms")
        print(f"   有效像素比例: {losses['valid_pixel_ratio'].item():.3f}")


if __name__ == "__main__":
    test_photometric_loss()
    benchmark_photometric_loss()