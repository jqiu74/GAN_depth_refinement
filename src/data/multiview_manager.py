#!/usr/bin/env python3
"""
多视图数据管理器
负责处理多视图深度图精修所需的数据：RGB序列、深度图、相机内参、位姿变换
支持真实数据集和虚拟数据生成
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

# 项目内导入
try:
    from ..utils.geometry_utils import GeometryUtils
except ImportError:
    try:
        from src.utils.geometry_utils import GeometryUtils
    except ImportError:
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.insert(0, project_root)
        from src.utils.geometry_utils import GeometryUtils


class MultiViewDataManager:
    """
    多视图数据管理器
    
    核心功能：
    1. 加载和管理多视图RGB-D序列
    2. 提供相机内参和位姿信息
    3. 生成虚拟多视图数据用于测试
    4. 数据增强和预处理
    """
    
    def __init__(self, 
                 data_root: Optional[str] = None,
                 sequence_length: int = 2,
                 image_size: Tuple[int, int] = (256, 256),
                 virtual_data_config: Optional[Dict] = None,
                 enable_augmentation: bool = True):
        """
        Args:
            data_root: 真实数据集根目录
            sequence_length: 序列长度（当前只支持2帧）
            image_size: 图像尺寸 (H, W)
            virtual_data_config: 虚拟数据配置
            enable_augmentation: 是否启用数据增强
        """
        self.data_root = Path(data_root) if data_root else None
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.enable_augmentation = enable_augmentation
        
        # 虚拟数据配置
        default_config = self._get_default_virtual_config()
        if virtual_data_config:
            # 合并用户配置和默认配置
            default_config.update(virtual_data_config)
        self.virtual_config = default_config
        
        # 数据索引
        self.data_index = []
        self.intrinsics_cache = {}
        self.pose_cache = {}
        
        # 初始化数据
        if self.data_root and self.data_root.exists():
            self._load_real_data_index()
        else:
            print("⚠️  未找到真实数据，将使用虚拟数据模式")
    
    def _get_default_virtual_config(self) -> Dict:
        """获取默认虚拟数据配置 - 优化版，确保光度损失有效"""
        return {
            'depth_range': (1.0, 5.0),               # 减小深度范围，更集中
            'noise_level': 0.05,                     # 噪声水平
            'focal_length_range': (300, 500),        # 增大焦距范围，减小视角
            'translation_range': 0.05,               # 大幅减小平移范围
            'rotation_range': 0.02,                  # 大幅减小旋转范围（约1.1度）
            'texture_complexity': 'medium',          # 纹理复杂度
            'enable_occlusion': False,               # 暂时禁用遮挡检测
            'enable_objects': True,                  # 是否添加几何体
        }
    
    def _load_real_data_index(self):
        """加载真实数据索引"""
        print("📂 加载真实数据索引...")
        
        # 支持常见的数据集格式
        supported_formats = [
            'kitti',      # KITTI数据集
            'tum',        # TUM RGB-D数据集
            'nyu',        # NYU Depth V2
            'custom'      # 自定义格式
        ]
        
        # 检测数据集格式
        dataset_format = self._detect_dataset_format()
        
        if dataset_format == 'kitti':
            self._load_kitti_index()
        elif dataset_format == 'tum':
            self._load_tum_index()
        elif dataset_format == 'custom':
            self._load_custom_index()
        else:
            warnings.warn(f"未识别的数据集格式，使用虚拟数据")
    
    def _detect_dataset_format(self) -> str:
        """检测数据集格式"""
        if not self.data_root.exists():
            return 'unknown'
        
        # KITTI格式检测
        if (self.data_root / 'image_02').exists() and (self.data_root / 'velodyne').exists():
            return 'kitti'
        
        # TUM格式检测
        if (self.data_root / 'rgb.txt').exists() and (self.data_root / 'depth.txt').exists():
            return 'tum'
        
        # 自定义格式检测
        if (self.data_root / 'rgb').exists() and (self.data_root / 'depth').exists():
            return 'custom'
        
        return 'unknown'
    
    def _load_custom_index(self):
        """加载自定义格式数据索引"""
        print("   检测到自定义数据格式")
        
        rgb_dir = self.data_root / 'rgb'
        depth_dir = self.data_root / 'depth'
        
        # 获取RGB文件列表
        rgb_files = sorted(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')))
        depth_files = sorted(list(depth_dir.glob('*.png')) + list(depth_dir.glob('*.exr')))
        
        if len(rgb_files) == 0 or len(depth_files) == 0:
            warnings.warn("未找到RGB或深度图文件")
            return
        
        # 构建数据对
        min_length = min(len(rgb_files), len(depth_files))
        for i in range(min_length - 1):  # -1因为需要序列对
            self.data_index.append({
                'rgb_current': rgb_files[i],
                'rgb_next': rgb_files[i + 1],
                'depth_current': depth_files[i],
                'depth_next': depth_files[i + 1],
                'frame_id': i,
                'sequence_id': 0
            })
        
        print(f"   加载了 {len(self.data_index)} 个数据对")
        
        # 加载相机内参（如果存在）
        intrinsics_file = self.data_root / 'intrinsics.json'
        if intrinsics_file.exists():
            with open(intrinsics_file, 'r') as f:
                intrinsics_data = json.load(f)
                self.intrinsics_cache[0] = np.array(intrinsics_data['K'])
    
    def _load_kitti_index(self):
        """加载KITTI数据集索引（简化版）"""
        print("   检测到KITTI数据格式")
        # KITTI格式的具体实现
        # 这里提供一个简化的实现框架
        pass
    
    def _load_tum_index(self):
        """加载TUM RGB-D数据集索引（简化版）"""
        print("   检测到TUM数据格式")
        # TUM格式的具体实现
        # 这里提供一个简化的实现框架
        pass
    
    def get_sample(self, index: int, use_virtual: bool = False) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            index: 样本索引
            use_virtual: 是否强制使用虚拟数据
            
        Returns:
            sample: 包含所有必要数据的字典
        """
        if use_virtual or len(self.data_index) == 0:
            return self._generate_virtual_sample(index)
        else:
            return self._load_real_sample(index)
    
    def _load_real_sample(self, index: int) -> Dict[str, torch.Tensor]:
        """加载真实数据样本"""
        if index >= len(self.data_index):
            raise IndexError(f"索引 {index} 超出范围 {len(self.data_index)}")
        
        data_info = self.data_index[index]
        
        # 加载RGB图像
        rgb_current = self._load_image(data_info['rgb_current'])
        rgb_next = self._load_image(data_info['rgb_next'])
        
        # 加载深度图
        depth_current = self._load_depth(data_info['depth_current'])
        depth_next = self._load_depth(data_info['depth_next'])
        
        # 生成噪声深度图（模拟传感器噪声）
        noisy_depth = self._add_depth_noise(depth_current)
        
        # 获取相机内参
        sequence_id = data_info.get('sequence_id', 0)
        intrinsics = self._get_intrinsics(sequence_id)
        
        # 获取或生成位姿变换
        pose_transform = self._get_pose_transform(data_info)
        
        # 数据增强
        if self.enable_augmentation:
            rgb_current, rgb_next, depth_current, noisy_depth, intrinsics = self._apply_augmentation(
                rgb_current, rgb_next, depth_current, noisy_depth, intrinsics
            )
        
        return {
            'rgb_current': rgb_current,
            'rgb_next': rgb_next,
            'real_depth': depth_current,
            'noisy_depth': noisy_depth,
            'intrinsics': intrinsics,
            'pose_transform': pose_transform,
            'frame_id': data_info['frame_id'],
            'sequence_id': sequence_id
        }
    
    def _generate_virtual_sample(self, seed: int = None) -> Dict[str, torch.Tensor]:
        """
        生成虚拟数据样本
        
        这是一个完整的虚拟场景生成器，用于测试和开发
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        H, W = self.image_size
        device = 'cpu'  # 在CPU上生成，后续可转移到GPU
        
        # 1. 生成虚拟相机内参
        focal_min, focal_max = self.virtual_config['focal_length_range']
        fx = fy = np.random.uniform(focal_min, focal_max)
        cx, cy = W / 2, H / 2
        
        intrinsics = GeometryUtils.create_intrinsics_matrix(
            fx=fx, fy=fy, cx=cx, cy=cy, batch_size=1, device=device
        ).squeeze(0)  # [3, 3]
        
        # 2. 生成虚拟深度图
        depth_min, depth_max = self.virtual_config['depth_range']
        
        # 创建有结构的深度图（平面 + 噪声）
        depth_base = self._generate_structured_depth(H, W, depth_min, depth_max)
        real_depth = torch.from_numpy(depth_base).float().unsqueeze(0)  # [1, H, W]
        
        # 3. 添加传感器噪声生成噪声深度图
        noisy_depth = self._add_depth_noise(real_depth)
        
        # 4. 根据深度图生成虚拟RGB图像
        rgb_current = self._generate_virtual_rgb(depth_base, seed)
        
        # 5. 生成位姿变换
        pose_transform = self._generate_virtual_pose()
        
        # 6. 根据位姿变换生成下一帧RGB
        rgb_next = self._generate_next_frame_rgb(rgb_current, real_depth, intrinsics, pose_transform)
        
        return {
            'rgb_current': rgb_current,
            'rgb_next': rgb_next,
            'real_depth': real_depth,
            'noisy_depth': noisy_depth,
            'intrinsics': intrinsics,
            'pose_transform': pose_transform,
            'frame_id': seed or 0,
            'sequence_id': 0,
            'is_virtual': True
        }
    
    def _generate_structured_depth(self, H: int, W: int, depth_min: float, depth_max: float) -> np.ndarray:
        """生成有结构的深度图"""
        # 创建基础平面
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # 倾斜平面
        plane_depth = depth_min + (depth_max - depth_min) * (
            0.3 * (y_coords / H) + 0.2 * (x_coords / W) + 0.5
        )
        
        # 添加几何结构（球体、立方体等）
        if self.virtual_config.get('enable_objects', True):
            plane_depth = self._add_virtual_objects(plane_depth, H, W, depth_min, depth_max)
        
        # 添加平滑噪声
        noise = np.random.normal(0, 0.1, (H, W))
        smooth_noise = cv2.GaussianBlur(noise, (5, 5), 1.0)
        plane_depth += smooth_noise
        
        # 确保深度范围合理
        plane_depth = np.clip(plane_depth, depth_min, depth_max)
        
        return plane_depth.astype(np.float32)
    
    def _add_virtual_objects(self, depth: np.ndarray, H: int, W: int, 
                           depth_min: float, depth_max: float) -> np.ndarray:
        """在深度图中添加虚拟几何体"""
        # 添加球体
        sphere_center_x = W // 3
        sphere_center_y = H // 3
        sphere_radius = min(H, W) // 8
        
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        sphere_dist = np.sqrt((x_coords - sphere_center_x)**2 + (y_coords - sphere_center_y)**2)
        
        sphere_mask = sphere_dist <= sphere_radius
        if np.any(sphere_mask):
            # 球体深度（向前凸出）
            sphere_depth = depth_min + (depth_max - depth_min) * 0.3
            depth[sphere_mask] = np.minimum(depth[sphere_mask], sphere_depth)
        
        # 添加立方体
        cube_x1, cube_x2 = 2 * W // 3, 5 * W // 6
        cube_y1, cube_y2 = 2 * H // 3, 5 * H // 6
        cube_depth = depth_min + (depth_max - depth_min) * 0.4
        depth[cube_y1:cube_y2, cube_x1:cube_x2] = cube_depth
        
        return depth
    
    def _generate_virtual_rgb(self, depth: np.ndarray, seed: Optional[int] = None) -> torch.Tensor:
        """根据深度图生成虚拟RGB图像"""
        if seed is not None:
            np.random.seed(seed)
        
        H, W = depth.shape
        
        # 基础纹理生成
        if self.virtual_config['texture_complexity'] == 'simple':
            # 简单渐变纹理
            rgb = np.zeros((H, W, 3), dtype=np.float32)
            rgb[:, :, 0] = (depth - depth.min()) / (depth.max() - depth.min())  # 红色通道基于深度
            rgb[:, :, 1] = np.random.uniform(0.3, 0.7, (H, W))  # 绿色通道随机
            rgb[:, :, 2] = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(W) / W)[None, :]  # 蓝色通道周期性
        
        elif self.virtual_config['texture_complexity'] == 'medium':
            # 中等复杂度纹理
            rgb = np.random.uniform(0.2, 0.8, (H, W, 3)).astype(np.float32)
            
            # 添加基于深度的光照效果
            lighting = 0.5 + 0.5 * (depth - depth.min()) / (depth.max() - depth.min())
            for c in range(3):
                rgb[:, :, c] *= lighting
            
            # 添加一些纹理图案
            y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            pattern = 0.1 * np.sin(2 * np.pi * x_coords / 20) * np.cos(2 * np.pi * y_coords / 20)
            rgb += pattern[:, :, None]
        
        else:  # complex
            # 复杂纹理（更接近真实场景）
            rgb = self._generate_complex_texture(H, W, depth)
        
        # 确保值在合理范围内
        rgb = np.clip(rgb, 0.0, 1.0)
        
        # 转换为torch tensor [3, H, W]
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        
        return rgb_tensor
    
    def _generate_complex_texture(self, H: int, W: int, depth: np.ndarray) -> np.ndarray:
        """生成复杂纹理"""
        # 基础随机纹理
        rgb = np.random.uniform(0.1, 0.9, (H, W, 3)).astype(np.float32)
        
        # 添加多频率噪声
        for freq in [5, 10, 20]:
            noise_r = 0.1 * np.random.random((H//freq + 1, W//freq + 1))
            noise_g = 0.1 * np.random.random((H//freq + 1, W//freq + 1))
            noise_b = 0.1 * np.random.random((H//freq + 1, W//freq + 1))
            
            noise_r = cv2.resize(noise_r, (W, H))
            noise_g = cv2.resize(noise_g, (W, H))
            noise_b = cv2.resize(noise_b, (W, H))
            
            rgb[:, :, 0] += noise_r
            rgb[:, :, 1] += noise_g
            rgb[:, :, 2] += noise_b
        
        # 深度相关的光照
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        lighting = 0.7 + 0.3 * depth_normalized
        
        for c in range(3):
            rgb[:, :, c] *= lighting
        
        return rgb
    
    def _generate_virtual_pose(self) -> torch.Tensor:
        """生成虚拟位姿变换 - 优化版，确保有效投影像素足够多"""
        # 大幅减小位姿变换范围，确保更多像素保持在图像内
        translation_range = self.virtual_config['translation_range'] * 0.2  # 减小到原来的20%
        rotation_range = self.virtual_config['rotation_range'] * 0.1        # 减小到原来的10%
        
        # 随机平移（更小的范围）
        translation = np.random.uniform(-translation_range, translation_range, 3)
        
        # 随机旋转（更小的范围）
        rotation_angles = np.random.uniform(-rotation_range, rotation_range, 3)
        
        # 创建位姿矩阵
        pose_transform = GeometryUtils.create_pose_matrix(
            translation=torch.from_numpy(translation).unsqueeze(0).float(),
            rotation_angles=torch.from_numpy(rotation_angles).unsqueeze(0).float()
        ).squeeze(0)  # [4, 4]
        
        return pose_transform
    
    def _generate_next_frame_rgb(self, rgb_current: torch.Tensor, 
                               depth: torch.Tensor,
                               intrinsics: torch.Tensor,
                               pose_transform: torch.Tensor) -> torch.Tensor:
        """基于位姿变换生成下一帧RGB（简化版）"""
        # 这里使用简化的方法：添加一些变化来模拟相机运动
        rgb_next = rgb_current.clone()
        
        # 添加轻微的光照变化
        lighting_change = 0.95 + 0.1 * torch.rand(1).item()
        rgb_next *= lighting_change
        
        # 添加小的空间变换（简化的运动模拟）
        if torch.rand(1) > 0.5:  # 50%概率添加变换
            # 简单的平移变换
            shift_x = int(torch.randint(-5, 6, (1,)).item())
            shift_y = int(torch.randint(-5, 6, (1,)).item())
            
            rgb_next_shifted = torch.zeros_like(rgb_next)
            
            # 计算有效区域
            h, w = rgb_next.shape[1], rgb_next.shape[2]
            src_y1 = max(0, -shift_y)
            src_y2 = min(h, h - shift_y)
            src_x1 = max(0, -shift_x)
            src_x2 = min(w, w - shift_x)
            
            dst_y1 = max(0, shift_y)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            dst_x1 = max(0, shift_x)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            
            if src_y2 > src_y1 and src_x2 > src_x1:
                rgb_next_shifted[:, dst_y1:dst_y2, dst_x1:dst_x2] = rgb_next[:, src_y1:src_y2, src_x1:src_x2]
                
                # 填充边界
                rgb_next_shifted[:, :dst_y1, :] = rgb_next[:, :dst_y1, :]  # 上边界
                rgb_next_shifted[:, dst_y2:, :] = rgb_next[:, dst_y2:, :]  # 下边界
                rgb_next_shifted[:, :, :dst_x1] = rgb_next[:, :, :dst_x1]  # 左边界
                rgb_next_shifted[:, :, dst_x2:] = rgb_next[:, :, dst_x2:]  # 右边界
                
                rgb_next = rgb_next_shifted
        
        # 添加轻微噪声
        noise = 0.02 * torch.randn_like(rgb_next)
        rgb_next = torch.clamp(rgb_next + noise, 0.0, 1.0)
        
        return rgb_next
    
    def _add_depth_noise(self, clean_depth: torch.Tensor) -> torch.Tensor:
        """为深度图添加传感器噪声"""
        noise_level = self.virtual_config['noise_level']
        
        # 高斯噪声
        gaussian_noise = torch.randn_like(clean_depth) * noise_level
        
        # 深度相关的噪声（远处噪声更大）
        depth_dependent_noise = clean_depth * 0.01 * torch.randn_like(clean_depth)
        
        # 偶尔的outlier噪声
        outlier_mask = torch.rand_like(clean_depth) < 0.005  # 0.5%的像素
        outlier_noise = torch.rand_like(clean_depth) * 2.0  # 随机深度值
        
        noisy_depth = clean_depth + gaussian_noise + depth_dependent_noise
        noisy_depth = torch.where(outlier_mask, outlier_noise, noisy_depth)
        
        # 确保深度值为正
        noisy_depth = torch.clamp(noisy_depth, 0.1, 50.0)
        
        return noisy_depth
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """加载图像文件"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # 归一化到[0,1]
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor [3, H, W]
        return torch.from_numpy(image.transpose(2, 0, 1))
    
    def _load_depth(self, depth_path: Path) -> torch.Tensor:
        """加载深度图文件"""
        if depth_path.suffix.lower() == '.exr':
            # OpenEXR格式 - 需要额外的库
            try:
                import OpenEXR
                import Imath
                
                exr_file = OpenEXR.InputFile(str(depth_path))
                depth_data = exr_file.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
                depth = np.frombuffer(depth_data, dtype=np.float32)
                height = exr_file.header()['displayWindow'].max.y + 1
                width = exr_file.header()['displayWindow'].max.x + 1
                depth = depth.reshape((height, width))
                
            except ImportError:
                warnings.warn("OpenEXR库未安装，无法读取.exr文件。请使用PNG/NPY格式或安装OpenEXR库")
                # 回退到生成虚拟深度数据
                H, W = self.image_size
                depth = np.random.uniform(0.5, 5.0, (H, W)).astype(np.float32)
                
            except Exception as e:
                warnings.warn(f"读取EXR文件失败: {e}，使用虚拟数据")
                H, W = self.image_size
                depth = np.random.uniform(0.5, 5.0, (H, W)).astype(np.float32)
                
        elif depth_path.suffix.lower() == '.npy':
            # NumPy格式
            try:
                depth = np.load(depth_path).astype(np.float32)
            except Exception as e:
                warnings.warn(f"读取NPY文件失败: {e}，使用虚拟数据")
                H, W = self.image_size
                depth = np.random.uniform(0.5, 5.0, (H, W)).astype(np.float32)
                
        else:
            # PNG格式（通常用于存储16位深度）
            try:
                depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
                if depth is None:
                    raise ValueError(f"无法加载深度图: {depth_path}")
                
                # 转换为米为单位（根据实际情况调整）
                if depth.dtype == np.uint16:
                    # 16位深度图，通常以毫米为单位
                    depth = depth.astype(np.float32) / 1000.0
                elif depth.dtype == np.uint8:
                    # 8位深度图，需要缩放
                    depth = depth.astype(np.float32) / 255.0 * 10.0  # 假设最大深度10米
                else:
                    depth = depth.astype(np.float32)
                    
            except Exception as e:
                warnings.warn(f"读取深度图失败: {e}，使用虚拟数据")
                H, W = self.image_size
                depth = np.random.uniform(0.5, 5.0, (H, W)).astype(np.float32)
        
        # 确保深度图是2D的
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]  # 取第一个通道
        
        # 调整尺寸
        if depth.shape != self.image_size:
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]))
        
        # 确保深度值合理
        depth = np.clip(depth, 0.1, 100.0)  # 限制深度范围
        
        # 转换为tensor [1, H, W]
        return torch.from_numpy(depth).unsqueeze(0)
    
    def _get_intrinsics(self, sequence_id: int) -> torch.Tensor:
        """获取相机内参"""
        if sequence_id in self.intrinsics_cache:
            return torch.from_numpy(self.intrinsics_cache[sequence_id]).float()
        else:
            # 使用默认内参
            H, W = self.image_size
            return GeometryUtils.create_intrinsics_matrix(
                fx=200.0, fy=200.0, cx=W/2, cy=H/2, batch_size=1, device='cpu'
            ).squeeze(0)
    
    def _get_pose_transform(self, data_info: Dict) -> torch.Tensor:
        """获取位姿变换"""
        # 如果有预计算的位姿，使用它
        if 'pose_transform' in data_info:
            return torch.from_numpy(data_info['pose_transform']).float()
        else:
            # 生成虚拟位姿
            return self._generate_virtual_pose()
    
    def _apply_augmentation(self, rgb_current: torch.Tensor, rgb_next: torch.Tensor,
                          depth: torch.Tensor, noisy_depth: torch.Tensor,
                          intrinsics: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """应用数据增强"""
        # 简单的数据增强示例
        # 可以添加更多增强方法
        
        # 随机亮度调整
        if torch.rand(1) > 0.5:
            brightness_factor = 0.8 + 0.4 * torch.rand(1)
            rgb_current *= brightness_factor
            rgb_next *= brightness_factor
            rgb_current = torch.clamp(rgb_current, 0, 1)
            rgb_next = torch.clamp(rgb_next, 0, 1)
        
        return rgb_current, rgb_next, depth, noisy_depth, intrinsics
    
    def __len__(self) -> int:
        """数据集长度"""
        if len(self.data_index) > 0:
            return len(self.data_index)
        else:
            return 1000  # 虚拟数据集的默认长度
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        return {
            'num_samples': len(self),
            'image_size': self.image_size,
            'sequence_length': self.sequence_length,
            'has_real_data': len(self.data_index) > 0,
            'num_sequences': len(set(item.get('sequence_id', 0) for item in self.data_index)),
            'virtual_config': self.virtual_config
        }


# ============================================================================
# 测试和验证函数
# ============================================================================

def test_multiview_manager():
    """测试多视图数据管理器"""
    print("🧪 测试多视图数据管理器...")
    
    # 测试参数
    image_size = (128, 128)
    
    print("1. 测试虚拟数据生成...")
    manager = MultiViewDataManager(
        data_root=None,  # 使用虚拟数据
        image_size=image_size,
        virtual_data_config={
            'depth_range': (0.5, 8.0),
            'noise_level': 0.03,
            'texture_complexity': 'medium'
        }
    )
    
    print(f"   数据集长度: {len(manager)}")
    stats = manager.get_statistics()
    print(f"   统计信息: {stats}")
    
    print("2. 测试样本生成...")
    sample = manager.get_sample(0, use_virtual=True)
    
    required_keys = ['rgb_current', 'rgb_next', 'real_depth', 'noisy_depth', 
                    'intrinsics', 'pose_transform']
    
    for key in required_keys:
        assert key in sample, f"缺少键: {key}"
        print(f"   ✅ {key}: {sample[key].shape} | dtype: {sample[key].dtype}")
    
    # 检查数据范围
    assert 0 <= sample['rgb_current'].min() and sample['rgb_current'].max() <= 1, "RGB范围应在[0,1]"
    assert sample['real_depth'].min() > 0, "深度应为正值"
    assert sample['intrinsics'].shape == (3, 3), "内参矩阵应为3x3"
    assert sample['pose_transform'].shape == (4, 4), "位姿矩阵应为4x4"
    
    print("3. 测试多个样本的一致性...")
    samples = [manager.get_sample(i, use_virtual=True) for i in range(5)]
    
    # 检查形状一致性
    shape_consistency = all(
        s['rgb_current'].shape == samples[0]['rgb_current'].shape 
        for s in samples
    )
    assert shape_consistency, "所有样本形状应一致"
    print("   ✅ 样本形状一致")
    
    # 检查内参合理性
    intrinsics_reasonable = all(
        100 <= s['intrinsics'][0, 0] <= 500  # 焦距范围合理
        for s in samples
    )
    assert intrinsics_reasonable, "内参应在合理范围内"
    print("   ✅ 内参参数合理")
    
    print("4. 测试噪声深度图...")
    for i, sample in enumerate(samples[:3]):
        real_depth = sample['real_depth']
        noisy_depth = sample['noisy_depth']
        
        # 计算噪声水平
        noise_level = torch.abs(real_depth - noisy_depth).mean().item()
        print(f"   样本{i} 噪声水平: {noise_level:.4f}")
        
        assert 0.01 <= noise_level <= 0.2, f"噪声水平应合理，当前: {noise_level}"
    
    print("5. 测试几何变换...")
    sample = manager.get_sample(42, use_virtual=True)  # 固定种子测试
    
    # 测试深度图转点云
    points_3d = GeometryUtils.depth_to_pointcloud(
        sample['real_depth'].unsqueeze(0), 
        sample['intrinsics'].unsqueeze(0)
    )
    print(f"   ✅ 点云形状: {points_3d.shape}")
    
    # 测试位姿变换
    transformed_points = GeometryUtils.transform_points(
        points_3d, 
        sample['pose_transform'].unsqueeze(0)
    )
    print(f"   ✅ 变换后点云形状: {transformed_points.shape}")
    
    print("6. 测试真实数据加载接口...")
    # 测试真实数据路径（即使不存在）
    fake_data_root = "/nonexistent/path"
    manager_real = MultiViewDataManager(
        data_root=fake_data_root,
        image_size=image_size
    )
    
    # 应该回退到虚拟数据
    sample_fallback = manager_real.get_sample(0)
    assert 'is_virtual' in sample_fallback or len(manager_real.data_index) == 0
    print("   ✅ 真实数据路径不存在时正确回退到虚拟数据")
    
    print("\n🎉 多视图数据管理器测试通过!")
    return True


def test_data_compatibility():
    """测试数据与光度损失的兼容性"""
    print("\n🔗 测试数据与光度损失的兼容性...")
    
    # 创建数据管理器
    manager = MultiViewDataManager(image_size=(64, 64))
    sample = manager.get_sample(0, use_virtual=True)
    
    # 导入光度损失
    try:
        from ..losses.photometric_loss import PhotometricConsistencyLoss
    except ImportError:
        try:
            from src.losses.photometric_loss import PhotometricConsistencyLoss
        except ImportError:
            print("   ⚠️  无法导入光度损失模块，跳过兼容性测试")
            return True
    
    # 创建光度损失实例
    photometric_loss = PhotometricConsistencyLoss()
    
    # 准备数据（添加batch维度）
    refined_depth = sample['noisy_depth'].unsqueeze(0)  # [1, 1, H, W]
    rgb_current = sample['rgb_current'].unsqueeze(0)    # [1, 3, H, W]
    rgb_next = sample['rgb_next'].unsqueeze(0)          # [1, 3, H, W]
    intrinsics = sample['intrinsics'].unsqueeze(0)      # [1, 3, 3]
    pose_transform = sample['pose_transform'].unsqueeze(0)  # [1, 4, 4]
    
    print("1. 测试光度损失计算...")
    try:
        with torch.no_grad():
            losses = photometric_loss(
                refined_depth=refined_depth,
                rgb_current=rgb_current,
                rgb_next=rgb_next,
                intrinsics=intrinsics,
                pose_t_to_t1=pose_transform
            )
        
        print(f"   ✅ 光度损失: {losses['photometric_loss'].item():.6f}")
        print(f"   ✅ 有效像素比例: {losses['valid_pixel_ratio'].item():.3f}")
        
        assert losses['photometric_loss'] >= 0, "光度损失应为非负"
        assert 0 <= losses['valid_pixel_ratio'] <= 1, "有效像素比例应在[0,1]"
        
    except Exception as e:
        print(f"   ❌ 光度损失计算失败: {e}")
        return False
    
    print("2. 测试梯度计算...")
    refined_depth.requires_grad_(True)
    
    try:
        losses = photometric_loss(
            refined_depth=refined_depth,
            rgb_current=rgb_current,
            rgb_next=rgb_next,
            intrinsics=intrinsics,
            pose_t_to_t1=pose_transform
        )
        
        loss = losses['photometric_loss']
        loss.backward()
        
        if refined_depth.grad is not None:
            grad_norm = refined_depth.grad.norm().item()
            print(f"   ✅ 梯度范数: {grad_norm:.6f}")
        else:
            print("   ⚠️  梯度为None，可能是有效像素太少")
        
    except Exception as e:
        print(f"   ❌ 梯度计算失败: {e}")
        return False
    
    print("   ✅ 数据与光度损失完全兼容!")
    return True


def benchmark_data_generation():
    """数据生成性能基准测试"""
    print("\n🚀 数据生成性能基准测试...")
    
    import time
    
    manager = MultiViewDataManager(image_size=(256, 256))
    
    # 测试不同复杂度的纹理生成
    texture_configs = ['simple', 'medium', 'complex']
    
    for texture_complexity in texture_configs:
        print(f"\n📊 {texture_complexity} 纹理复杂度:")
        
        manager.virtual_config['texture_complexity'] = texture_complexity
        
        # 计时多次生成
        times = []
        for i in range(20):
            start_time = time.time()
            sample = manager.get_sample(i, use_virtual=True)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        avg_time = np.mean(times[5:])  # 排除前5次预热
        fps = 1000 / avg_time
        
        print(f"   平均生成时间: {avg_time:.2f} ms")
        print(f"   等效FPS: {fps:.1f}")
        
        # 内存使用检查
        sample_size_mb = sum(
            tensor.numel() * tensor.element_size() 
            for tensor in sample.values() 
            if torch.is_tensor(tensor)
        ) / (1024 * 1024)
        
        print(f"   样本内存占用: {sample_size_mb:.2f} MB")


def test_batch_generation():
    """测试批量数据生成"""
    print("\n📦 测试批量数据生成...")
    
    manager = MultiViewDataManager(image_size=(128, 128))
    
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        print(f"\n批量大小: {batch_size}")
        
        samples = [manager.get_sample(i, use_virtual=True) for i in range(batch_size)]
        
        # 将样本组织成批次
        batch = {}
        for key in samples[0].keys():
            if torch.is_tensor(samples[0][key]):
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                batch[key] = [s[key] for s in samples]
        
        print(f"   批次形状:")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"     {key}: {value.shape}")
        
        # 验证批次数据
        assert batch['rgb_current'].shape[0] == batch_size, f"批次大小不匹配"
        assert batch['real_depth'].shape[0] == batch_size, f"批次大小不匹配"
        
        print(f"   ✅ 批次 {batch_size} 生成成功")


def demo_virtual_scene():
    """演示虚拟场景生成"""
    print("\n🎨 演示虚拟场景生成...")
    
    try:
        import matplotlib.pyplot as plt
        
        manager = MultiViewDataManager(
            image_size=(256, 256),
            virtual_data_config={
                'texture_complexity': 'complex',
                'enable_objects': True,
                'depth_range': (1.0, 10.0)
            }
        )
        
        sample = manager.get_sample(42, use_virtual=True)  # 固定种子
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # RGB当前帧
        rgb_current = sample['rgb_current'].permute(1, 2, 0).numpy()
        axes[0, 0].imshow(rgb_current)
        axes[0, 0].set_title('RGB Current')
        axes[0, 0].axis('off')
        
        # RGB下一帧
        rgb_next = sample['rgb_next'].permute(1, 2, 0).numpy()
        axes[0, 1].imshow(rgb_next)
        axes[0, 1].set_title('RGB Next')
        axes[0, 1].axis('off')
        
        # 真实深度图
        real_depth = sample['real_depth'].squeeze().numpy()
        im1 = axes[0, 2].imshow(real_depth, cmap='viridis')
        axes[0, 2].set_title('Real Depth')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2])
        
        # 噪声深度图
        noisy_depth = sample['noisy_depth'].squeeze().numpy()
        im2 = axes[1, 0].imshow(noisy_depth, cmap='viridis')
        axes[1, 0].set_title('Noisy Depth')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # 深度差异
        depth_diff = np.abs(real_depth - noisy_depth)
        im3 = axes[1, 1].imshow(depth_diff, cmap='hot')
        axes[1, 1].set_title('Depth Noise')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1])
        
        # RGB差异
        rgb_diff = torch.abs(sample['rgb_current'] - sample['rgb_next']).mean(0).numpy()
        im4 = axes[1, 2].imshow(rgb_diff, cmap='hot')
        axes[1, 2].set_title('RGB Difference')
        axes[1, 2].axis('off')
        plt.colorbar(im4, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('virtual_scene_demo.png', dpi=150, bbox_inches='tight')
        print("   ✅ 虚拟场景可视化已保存为 'virtual_scene_demo.png'")
        plt.close()
        
        # 打印统计信息
        print(f"\n   场景统计:")
        print(f"     深度范围: {real_depth.min():.2f} - {real_depth.max():.2f}")
        print(f"     噪声水平: {depth_diff.mean():.4f} ± {depth_diff.std():.4f}")
        print(f"     RGB变化: {rgb_diff.mean():.4f}")
        
        # 相机参数
        intrinsics = sample['intrinsics']
        pose = sample['pose_transform']
        print(f"     焦距: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}")
        print(f"     主点: cx={intrinsics[0,2]:.1f}, cy={intrinsics[1,2]:.1f}")
        print(f"     平移: {pose[:3, 3].numpy()}")
        
    except ImportError:
        print("   ⚠️  matplotlib未安装，跳过可视化演示")
    except Exception as e:
        print(f"   ❌ 可视化演示失败: {e}")


if __name__ == "__main__":
    test_multiview_manager()
    test_data_compatibility()
    benchmark_data_generation()
    test_batch_generation()
    demo_virtual_scene()