#!/usr/bin/env python3
"""
TUM RGB-D数据集加载器
支持加载TUM数据集用于深度图精修训练
"""

import torch
import torch.utils.data as data
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.spatial.transform import Rotation as R
import warnings

# 导入项目模块
try:
    from src.utils.geometry_utils import GeometryUtils
except ImportError:
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)
    from src.utils.geometry_utils import GeometryUtils


class TUMRGBDDataset(data.Dataset):
    """
    TUM RGB-D数据集加载器
    
    数据集结构:
    - rgb/: RGB图像 (timestamp.png)
    - depth/: 深度图 (timestamp.png)
    - groundtruth.txt: 相机位姿
    - rgb.txt, depth.txt: 时间戳映射
    """
    
    def __init__(self, 
                 data_root: str,
                 sequence: str = 'rgbd_dataset_freiburg1_desk',
                 image_size: Tuple[int, int] = (256, 256),
                 depth_scale: float = 5000.0,  # TUM使用5000来存储深度
                 min_depth: float = 0.1,
                 max_depth: float = 10.0,
                 stride: int = 1,  # 帧间隔
                 split: str = 'train',  # train/val/test
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 augment: bool = True,
                 noise_level: float = 0.05):
        """
        Args:
            data_root: TUM数据集根目录
            sequence: 序列名称
            image_size: 输出图像大小 (H, W)
            depth_scale: 深度缩放因子
            min_depth: 最小深度值
            max_depth: 最大深度值
            stride: 采样间隔
            split: 数据集分割
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            augment: 是否数据增强
            noise_level: 深度噪声水平
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.sequence_path = self.data_root / sequence
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.stride = stride
        self.split = split
        self.augment = augment and (split == 'train')
        self.noise_level = noise_level
        
        # 检查数据集是否存在
        if not self.sequence_path.exists():
            raise ValueError(f"数据集路径不存在: {self.sequence_path}")
        
        # 加载数据索引
        self.rgb_depth_matches = self._load_associations()
        self.poses = self._load_poses()
        self.intrinsics = self._load_intrinsics()
        
        # 创建数据对
        self.data_pairs = self._create_data_pairs()
        
        # 数据集分割
        self._split_dataset(train_ratio, val_ratio)
        
        print(f"✅ 加载TUM {sequence} - {split}集")
        print(f"   样本数: {len(self)}")
        print(f"   图像尺寸: {image_size}")
        print(f"   帧间隔: {stride}")
    
    def _load_associations(self) -> List[Dict]:
        """加载RGB-深度图像关联"""
        assoc_file = self.sequence_path / 'associate.txt'
        
        if assoc_file.exists():
            # 使用官方的associate.txt文件
            matches = []
            with open(assoc_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        matches.append({
                            'timestamp': float(parts[0]),
                            'rgb': parts[1],
                            'depth_timestamp': float(parts[2]),
                            'depth': parts[3]
                        })
            return matches
        else:
            # 如果没有associate.txt，自动关联
            return self._auto_associate()
    
    def _auto_associate(self) -> List[Dict]:
        """自动关联RGB和深度图像"""
        rgb_dir = self.sequence_path / 'rgb'
        depth_dir = self.sequence_path / 'depth'
        
        # 获取所有RGB图像
        rgb_files = sorted([f for f in rgb_dir.glob('*.png')])
        depth_files = sorted([f for f in depth_dir.glob('*.png')])
        
        if len(rgb_files) == 0:
            raise ValueError(f"未找到RGB图像: {rgb_dir}")
        if len(depth_files) == 0:
            raise ValueError(f"未找到深度图像: {depth_dir}")
        
        # 基于时间戳匹配
        matches = []
        for rgb_file in rgb_files:
            rgb_timestamp = float(rgb_file.stem)
            
            # 找最近的深度图
            best_depth = None
            best_diff = float('inf')
            
            for depth_file in depth_files:
                depth_timestamp = float(depth_file.stem)
                diff = abs(rgb_timestamp - depth_timestamp)
                
                if diff < best_diff and diff < 0.02:  # 20ms容差
                    best_diff = diff
                    best_depth = depth_file
            
            if best_depth:
                matches.append({
                    'timestamp': rgb_timestamp,
                    'rgb': f'rgb/{rgb_file.name}',
                    'depth_timestamp': float(best_depth.stem),
                    'depth': f'depth/{best_depth.name}'
                })
        
        return matches
    
    def _load_poses(self) -> Dict[float, np.ndarray]:
        """加载相机位姿"""
        pose_file = self.sequence_path / 'groundtruth.txt'
        poses = {}
        
        if not pose_file.exists():
            warnings.warn(f"未找到位姿文件: {pose_file}")
            return poses
        
        # 读取位姿数据
        with open(pose_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 8:
                    timestamp = float(parts[0])
                    # tx ty tz qx qy qz qw
                    translation = np.array([float(parts[i]) for i in range(1, 4)])
                    quaternion = np.array([float(parts[i]) for i in range(4, 8)])
                    
                    # 构建4x4变换矩阵
                    pose = np.eye(4)
                    # 四元数转旋转矩阵 (注意TUM使用的是xyzw顺序)
                    rotation = R.from_quat(quaternion)
                    pose[:3, :3] = rotation.as_matrix()
                    pose[:3, 3] = translation
                    
                    poses[timestamp] = pose
        
        return poses
    
    def _load_intrinsics(self) -> np.ndarray:
        """加载相机内参"""
        # TUM Freiburg 1 相机内参
        # 这些是常用的默认值，可以根据实际相机调整
        if 'freiburg1' in str(self.sequence_path):
            # Freiburg 1: Kinect
            fx, fy = 517.3, 516.5
            cx, cy = 318.6, 255.3
        elif 'freiburg2' in str(self.sequence_path):
            # Freiburg 2: Kinect
            fx, fy = 520.9, 521.0
            cx, cy = 325.1, 249.7
        elif 'freiburg3' in str(self.sequence_path):
            # Freiburg 3: Asus Xtion
            fx, fy = 535.4, 539.2
            cx, cy = 320.1, 247.6
        else:
            # 默认值
            warnings.warn("使用默认相机内参，可能需要调整")
            fx, fy = 525.0, 525.0
            cx, cy = 319.5, 239.5
        
        # 缩放内参以匹配调整后的图像尺寸
        scale_x = self.image_size[1] / 640.0  # 原始宽度640
        scale_y = self.image_size[0] / 480.0  # 原始高度480
        
        K = np.array([
            [fx * scale_x, 0, cx * scale_x],
            [0, fy * scale_y, cy * scale_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def _create_data_pairs(self) -> List[Dict]:
        """创建训练数据对"""
        pairs = []
        
        for i in range(0, len(self.rgb_depth_matches) - self.stride, self.stride):
            current = self.rgb_depth_matches[i]
            next_frame = self.rgb_depth_matches[i + self.stride]
            
            # 检查是否有对应的位姿
            curr_timestamp = current['timestamp']
            next_timestamp = next_frame['timestamp']
            
            curr_pose = self._get_nearest_pose(curr_timestamp)
            next_pose = self._get_nearest_pose(next_timestamp)
            
            if curr_pose is not None and next_pose is not None:
                pairs.append({
                    'current': current,
                    'next': next_frame,
                    'pose_current': curr_pose,
                    'pose_next': next_pose
                })
        
        return pairs
    
    def _get_nearest_pose(self, timestamp: float, max_diff: float = 0.01) -> Optional[np.ndarray]:
        """获取最近的位姿"""
        if not self.poses:
            return None
        
        best_timestamp = min(self.poses.keys(), key=lambda t: abs(t - timestamp))
        if abs(best_timestamp - timestamp) < max_diff:
            return self.poses[best_timestamp]
        return None
    
    def _split_dataset(self, train_ratio: float, val_ratio: float):
        """划分数据集"""
        n_total = len(self.data_pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # 设置随机种子保证可重复性
        np.random.seed(42)
        indices = np.random.permutation(n_total)
        
        if self.split == 'train':
            self.indices = indices[:n_train]
        elif self.split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        else:  # test
            self.indices = indices[n_train + n_val:]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个训练样本"""
        real_idx = self.indices[idx]
        pair = self.data_pairs[real_idx]
        
        # 加载RGB图像
        rgb_current = self._load_rgb(pair['current']['rgb'])
        rgb_next = self._load_rgb(pair['next']['rgb'])
        
        # 加载深度图
        depth_current = self._load_depth(pair['current']['depth'])
        
        # 添加噪声生成噪声深度
        noisy_depth = self._add_depth_noise(depth_current)
        
        # 计算相对位姿变换
        pose_transform = self._compute_relative_pose(
            pair['pose_current'], 
            pair['pose_next']
        )
        
        # 数据增强
        if self.augment:
            rgb_current, rgb_next, depth_current, noisy_depth = self._apply_augmentation(
                rgb_current, rgb_next, depth_current, noisy_depth
            )
        
        # 转换为tensor
        sample = {
            'rgb_current': self._to_tensor(rgb_current),
            'rgb_next': self._to_tensor(rgb_next),
            'real_depth': torch.from_numpy(depth_current).unsqueeze(0),
            'noisy_depth': torch.from_numpy(noisy_depth).unsqueeze(0),
            'intrinsics': torch.from_numpy(self.intrinsics),
            'pose_transform': torch.from_numpy(pose_transform),
            'frame_id': pair['current']['timestamp'],
            'sequence_id': 0
        }
        
        return sample
    
    def _load_rgb(self, path: str) -> np.ndarray:
        """加载RGB图像"""
        full_path = self.sequence_path / path
        img = cv2.imread(str(full_path))
        if img is None:
            raise ValueError(f"无法加载图像: {full_path}")
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        
        # 归一化到[0,1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def _load_depth(self, path: str) -> np.ndarray:
        """加载深度图"""
        full_path = self.sequence_path / path
        depth = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"无法加载深度图: {full_path}")
        
        # 转换为米
        depth = depth.astype(np.float32) / self.depth_scale
        
        # 调整大小
        depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]))
        
        # 裁剪到有效范围
        depth = np.clip(depth, self.min_depth, self.max_depth)
        
        return depth
    
    def _add_depth_noise(self, depth: np.ndarray) -> np.ndarray:
        """添加深度噪声"""
        noisy_depth = depth.copy()
        
        # 高斯噪声
        gaussian_noise = np.random.randn(*depth.shape) * self.noise_level
        noisy_depth += gaussian_noise
        
        # 深度相关噪声
        depth_noise = depth * 0.01 * np.random.randn(*depth.shape)
        noisy_depth += depth_noise
        
        # 随机孔洞
        if np.random.rand() < 0.3:
            n_holes = np.random.randint(1, 5)
            for _ in range(n_holes):
                h, w = depth.shape
                hole_size = np.random.randint(5, 20)
                x = np.random.randint(0, w - hole_size)
                y = np.random.randint(0, h - hole_size)
                noisy_depth[y:y+hole_size, x:x+hole_size] = 0
        
        # 确保深度值合理
        noisy_depth = np.clip(noisy_depth, 0, self.max_depth)
        
        return noisy_depth
    
    def _compute_relative_pose(self, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """计算相对位姿变换 T1->T2"""
        # T_1_to_2 = T_2 * T_1^(-1)
        pose1_inv = np.linalg.inv(pose1)
        relative_pose = pose2 @ pose1_inv
        return relative_pose.astype(np.float32)
    
    def _apply_augmentation(self, rgb1: np.ndarray, rgb2: np.ndarray, 
                          depth: np.ndarray, noisy_depth: np.ndarray):
        """应用数据增强"""
        # 随机水平翻转
        if np.random.rand() > 0.5:
            rgb1 = np.fliplr(rgb1)
            rgb2 = np.fliplr(rgb2)
            depth = np.fliplr(depth)
            noisy_depth = np.fliplr(noisy_depth)
        
        # 随机颜色抖动
        if np.random.rand() > 0.5:
            brightness = 0.8 + 0.4 * np.random.rand()
            rgb1 = np.clip(rgb1 * brightness, 0, 1)
            rgb2 = np.clip(rgb2 * brightness, 0, 1)
        
        return rgb1, rgb2, depth, noisy_depth
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """转换为tensor格式 [C, H, W]"""
        if len(img.shape) == 2:
            return torch.from_numpy(img).unsqueeze(0)
        else:
            return torch.from_numpy(img.transpose(2, 0, 1))
    
    def get_sample_visualization(self, idx: int) -> Dict:
        """获取用于可视化的样本"""
        sample = self[idx]
        
        # 转回numpy用于可视化
        vis_data = {
            'rgb_current': sample['rgb_current'].permute(1, 2, 0).numpy(),
            'rgb_next': sample['rgb_next'].permute(1, 2, 0).numpy(),
            'real_depth': sample['real_depth'].squeeze().numpy(),
            'noisy_depth': sample['noisy_depth'].squeeze().numpy(),
            'intrinsics': sample['intrinsics'].numpy(),
            'pose_transform': sample['pose_transform'].numpy()
        }
        
        return vis_data


def create_tum_dataloaders(data_root: str,
                          sequence: str = 'rgbd_dataset_freiburg1_desk',
                          batch_size: int = 4,
                          image_size: Tuple[int, int] = (256, 256),
                          num_workers: int = 4,
                          **kwargs) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    创建TUM数据加载器
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = TUMRGBDDataset(
        data_root=data_root,
        sequence=sequence,
        image_size=image_size,
        split='train',
        augment=True,
        **kwargs
    )
    
    val_dataset = TUMRGBDDataset(
        data_root=data_root,
        sequence=sequence,
        image_size=image_size,
        split='val',
        augment=False,
        **kwargs
    )
    
    test_dataset = TUMRGBDDataset(
        data_root=data_root,
        sequence=sequence,
        image_size=image_size,
        split='test',
        augment=False,
        **kwargs
    )
    
    # 创建加载器
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# 测试和可视化
# ============================================================================

def test_tum_dataset():
    """测试TUM数据集加载器"""
    print("🧪 测试TUM数据集加载器...")
    
    # 假设数据在这个路径
    data_root = "data/tum_rgbd"
    
    # 检查数据是否存在
    if not os.path.exists(data_root):
        print(f"❌ 数据目录不存在: {data_root}")
        print("请先下载TUM数据集")
        return False
    
    try:
        # 创建数据集
        dataset = TUMRGBDDataset(
            data_root=data_root,
            sequence='rgbd_dataset_freiburg1_desk',
            image_size=(128, 128),
            split='train',
            stride=1
        )
        
        print(f"✅ 数据集大小: {len(dataset)}")
        
        # 测试加载一个样本
        sample = dataset[0]
        
        print("\n样本内容:")
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {value}")
        
        # 测试数据加载器
        loader = data.DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        
        print("\n批次形状:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_tum_sample():
    """可视化TUM数据样本"""
    try:
        import matplotlib.pyplot as plt
        
        data_root = "data/tum_rgbd"
        
        if not os.path.exists(data_root):
            print("数据不存在，跳过可视化")
            return
        
        dataset = TUMRGBDDataset(
            data_root=data_root,
            sequence='rgbd_dataset_freiburg1_desk',
            image_size=(256, 256),
            split='train'
        )
        
        # 获取可视化数据
        vis_data = dataset.get_sample_visualization(0)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # RGB当前帧
        axes[0, 0].imshow(vis_data['rgb_current'])
        axes[0, 0].set_title('RGB Current')
        axes[0, 0].axis('off')
        
        # RGB下一帧
        axes[0, 1].imshow(vis_data['rgb_next'])
        axes[0, 1].set_title('RGB Next')
        axes[0, 1].axis('off')
        
        # 真实深度
        im1 = axes[1, 0].imshow(vis_data['real_depth'], cmap='viridis')
        axes[1, 0].set_title('Real Depth')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0])
        
        # 噪声深度
        im2 = axes[1, 1].imshow(vis_data['noisy_depth'], cmap='viridis')
        axes[1, 1].set_title('Noisy Depth')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('tum_sample.png')
        print("✅ 可视化已保存为 tum_sample.png")
        plt.close()
        
    except ImportError:
        print("matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"可视化失败: {e}")


if __name__ == "__main__":
    test_tum_dataset()
    visualize_tum_sample()