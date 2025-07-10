#!/usr/bin/env python3
"""
几何变换工具模块
提供深度图精修项目所需的基础几何变换功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GeometryUtils:
    """几何变换工具类"""
    
    @staticmethod
    def depth_to_pointcloud(depth: torch.Tensor, 
                          intrinsics: torch.Tensor) -> torch.Tensor:
        """
        将深度图转换为3D点云
        
        Args:
            depth: 深度图 [B, 1, H, W]
            intrinsics: 相机内参矩阵 [B, 3, 3]
            
        Returns:
            points_3d: 3D点云 [B, H, W, 3]
        """
        B, _, H, W = depth.shape
        device = depth.device
        
        # 创建像素坐标网格 (i, j)
        i, j = torch.meshgrid(
            torch.arange(W, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            indexing='xy'
        )
        
        # 齐次像素坐标 [H, W, 3]
        pixel_coords = torch.stack([i, j, torch.ones_like(i)], dim=-1)
        pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 3]
        
        # 计算相机内参逆矩阵
        K_inv = torch.inverse(intrinsics)  # [B, 3, 3]
        
        # 像素坐标转归一化相机坐标
        # einsum: [B,3,3] × [B,H,W,3] -> [B,H,W,3]
        cam_coords = torch.einsum('bij,bhwj->bhwi', K_inv, pixel_coords)
        
        # 乘以深度得到3D点云
        depth_expanded = depth.permute(0, 2, 3, 1)  # [B, H, W, 1]
        points_3d = cam_coords * depth_expanded  # [B, H, W, 3]
        
        return points_3d
    
    @staticmethod
    def transform_points(points: torch.Tensor, 
                        transform: torch.Tensor) -> torch.Tensor:
        """
        应用4x4变换矩阵到3D点
        
        Args:
            points: 3D点 [B, H, W, 3]
            transform: 4x4变换矩阵 [B, 4, 4]
            
        Returns:
            transformed_points: 变换后的3D点 [B, H, W, 3]
        """
        B, H, W, _ = points.shape
        device = points.device
        
        # 转为齐次坐标
        ones = torch.ones(B, H, W, 1, device=device)
        points_homo = torch.cat([points, ones], dim=-1)  # [B, H, W, 4]
        
        # 应用变换
        # einsum: [B,4,4] × [B,H,W,4] -> [B,H,W,4]
        transformed_homo = torch.einsum('bij,bhwj->bhwi', transform, points_homo)
        
        # 返回前3个坐标
        return transformed_homo[..., :3]
    
    @staticmethod
    def project_to_image(points_3d: torch.Tensor, 
                        intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将3D点投影到图像平面
        
        Args:
            points_3d: 3D点 [B, H, W, 3]
            intrinsics: 相机内参 [B, 3, 3]
            
        Returns:
            pixel_coords: 像素坐标 [B, H, W, 2]
            valid_mask: 有效掩码 [B, H, W]
        """
        B, H, W, _ = points_3d.shape
        
        # 投影到相机平面
        # einsum: [B,3,3] × [B,H,W,3] -> [B,H,W,3]
        projected_homo = torch.einsum('bij,bhwj->bhwi', intrinsics, points_3d)
        
        # 透视除法
        z_coords = projected_homo[..., 2:3]  # [B, H, W, 1]
        pixel_coords = projected_homo[..., :2] / (z_coords + 1e-8)  # [B, H, W, 2]
        
        # 计算有效掩码
        valid_depth = z_coords.squeeze(-1) > 0  # 深度为正
        valid_x = (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < W)
        valid_y = (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < H)
        valid_mask = valid_depth & valid_x & valid_y
        
        return pixel_coords, valid_mask
    
    @staticmethod
    def bilinear_sample(image: torch.Tensor, 
                       coords: torch.Tensor, 
                       valid_mask: torch.Tensor) -> torch.Tensor:
        """
        双线性插值采样图像
        
        Args:
            image: 输入图像 [B, C, H, W]
            coords: 采样坐标 [B, H, W, 2]
            valid_mask: 有效掩码 [B, H, W]
            
        Returns:
            sampled: 采样后的图像 [B, C, H, W]
        """
        B, C, H, W = image.shape
        
        # 归一化坐标到[-1, 1]
        normalized_coords = coords.clone()
        normalized_coords[..., 0] = (coords[..., 0] / (W - 1)) * 2 - 1
        normalized_coords[..., 1] = (coords[..., 1] / (H - 1)) * 2 - 1
        
        # grid_sample要求坐标为[B, H, W, 2]格式
        sampled = F.grid_sample(
            image, 
            normalized_coords, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        
        # 应用有效掩码
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(sampled)
        sampled = sampled * valid_mask_expanded.float()
        
        return sampled
    
    @staticmethod
    def compute_image_gradients(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算图像梯度
        
        Args:
            image: 输入图像 [B, C, H, W]
            
        Returns:
            grad_x: X方向梯度 [B, C, H, W-1]
            grad_y: Y方向梯度 [B, C, H-1, W]
        """
        grad_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        grad_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
        
        return grad_x, grad_y
    
    @staticmethod
    def edge_aware_weights(image: torch.Tensor, 
                          sigma: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算边缘感知权重
        
        Args:
            image: 输入图像 [B, C, H, W]
            sigma: 权重调节参数
            
        Returns:
            weight_x: X方向权重 [B, C, H, W-1]  
            weight_y: Y方向权重 [B, C, H-1, W]
        """
        # 转换为灰度图
        if image.shape[1] == 3:  # RGB
            gray = torch.mean(image, dim=1, keepdim=True)
        else:
            gray = image
        
        # 计算梯度
        grad_x, grad_y = GeometryUtils.compute_image_gradients(gray)
        
        # 计算边缘感知权重 (梯度越大，权重越小)
        weight_x = torch.exp(-grad_x / sigma)
        weight_y = torch.exp(-grad_y / sigma)
        
        return weight_x, weight_y
    
    @staticmethod
    def create_intrinsics_matrix(fx: float, fy: float, 
                               cx: float, cy: float, 
                               batch_size: int = 1,
                               device: str = 'cpu') -> torch.Tensor:
        """
        创建相机内参矩阵
        
        Args:
            fx, fy: 焦距
            cx, cy: 主点坐标
            batch_size: 批次大小
            device: 设备
            
        Returns:
            intrinsics: 内参矩阵 [B, 3, 3]
        """
        K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy], 
            [0.0, 0.0, 1.0]
        ], device=device, dtype=torch.float32)
        
        return K.unsqueeze(0).expand(batch_size, -1, -1)
    
    @staticmethod
    def create_pose_matrix(translation: torch.Tensor, 
                          rotation_angles: torch.Tensor) -> torch.Tensor:
        """
        从平移和旋转角度创建4x4位姿矩阵
        
        Args:
            translation: 平移向量 [B, 3]
            rotation_angles: 旋转角度(弧度) [B, 3] (roll, pitch, yaw)
            
        Returns:
            pose: 4x4位姿矩阵 [B, 4, 4]
        """
        B = translation.shape[0]
        device = translation.device
        
        # 创建旋转矩阵 (简化的欧拉角转换)
        roll, pitch, yaw = rotation_angles[:, 0], rotation_angles[:, 1], rotation_angles[:, 2]
        
        # 各轴旋转矩阵
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        
        # 组合旋转矩阵 R = R_z(yaw) * R_y(pitch) * R_x(roll)
        R = torch.zeros(B, 3, 3, device=device)
        
        R[:, 0, 0] = cos_y * cos_p
        R[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
        R[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
        
        R[:, 1, 0] = sin_y * cos_p
        R[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
        R[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
        
        R[:, 2, 0] = -sin_p
        R[:, 2, 1] = cos_p * sin_r
        R[:, 2, 2] = cos_p * cos_r
        
        # 创建4x4位姿矩阵
        pose = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        pose[:, :3, :3] = R
        pose[:, :3, 3] = translation
        
        return pose


# ============================================================================
# 测试函数
# ============================================================================

def test_geometry_utils():
    """测试几何工具函数"""
    print("🧪 测试几何工具模块...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    H, W = 64, 64
    
    # 1. 测试深度图转点云
    print("1. 测试深度图转点云...")
    depth = torch.rand(batch_size, 1, H, W, device=device)
    intrinsics = GeometryUtils.create_intrinsics_matrix(
        fx=100.0, fy=100.0, cx=32.0, cy=32.0, 
        batch_size=batch_size, device=device
    )
    
    points_3d = GeometryUtils.depth_to_pointcloud(depth, intrinsics)
    print(f"   ✅ 点云形状: {points_3d.shape}")
    assert points_3d.shape == (batch_size, H, W, 3)
    
    # 2. 测试点变换
    print("2. 测试点变换...")
    translation = torch.randn(batch_size, 3, device=device) * 0.1
    rotation_angles = torch.randn(batch_size, 3, device=device) * 0.1
    pose = GeometryUtils.create_pose_matrix(translation, rotation_angles)
    
    transformed_points = GeometryUtils.transform_points(points_3d, pose)
    print(f"   ✅ 变换后点云形状: {transformed_points.shape}")
    assert transformed_points.shape == (batch_size, H, W, 3)
    
    # 3. 测试投影
    print("3. 测试投影...")
    pixel_coords, valid_mask = GeometryUtils.project_to_image(points_3d, intrinsics)
    print(f"   ✅ 像素坐标形状: {pixel_coords.shape}")
    print(f"   ✅ 有效掩码形状: {valid_mask.shape}")
    print(f"   ✅ 有效像素比例: {valid_mask.float().mean():.3f}")
    
    # 4. 测试双线性采样
    print("4. 测试双线性采样...")
    image = torch.rand(batch_size, 3, H, W, device=device)
    sampled = GeometryUtils.bilinear_sample(image, pixel_coords, valid_mask)
    print(f"   ✅ 采样图像形状: {sampled.shape}")
    
    # 5. 测试图像梯度
    print("5. 测试图像梯度...")
    grad_x, grad_y = GeometryUtils.compute_image_gradients(image)
    print(f"   ✅ X梯度形状: {grad_x.shape}")
    print(f"   ✅ Y梯度形状: {grad_y.shape}")
    
    # 6. 测试边缘感知权重
    print("6. 测试边缘感知权重...")
    weight_x, weight_y = GeometryUtils.edge_aware_weights(image)
    print(f"   ✅ X权重形状: {weight_x.shape}")
    print(f"   ✅ Y权重形状: {weight_y.shape}")
    
    print("\n🎉 几何工具模块测试通过!")
    
    return True


if __name__ == "__main__":
    test_geometry_utils()