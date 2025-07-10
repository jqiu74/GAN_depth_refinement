#!/usr/bin/env python3
"""
使用TUM RGB-D数据集训练深度精修GAN
快速开始脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from src.models.generator import UltraLightDepthGenerator, StandardDepthGenerator
from src.models.discriminator import LightweightDiscriminator, StandardDiscriminator
from src.losses.photometric_loss import PhotometricConsistencyLoss
from src.data.dataset import TUMRGBDDataset, create_tum_dataloaders
from src.utils.geometry_utils import GeometryUtils


class TUMTrainer:
    """使用TUM数据集的训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 初始化TUM训练器")
        print(f"   设备: {self.device}")
        print(f"   数据路径: {config['data_root']}")
        print(f"   序列: {config['sequence']}")
        
        # 创建模型
        self._create_models()
        
        # 创建优化器
        self._create_optimizers()
        
        # 创建损失函数
        self._create_losses()
        
        # 创建数据加载器
        self._create_dataloaders()
        
        # 训练状态
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        
    def _create_models(self):
        """创建模型"""
        print("📦 创建模型...")
        
        if self.config['model_size'] == 'light':
            self.generator = UltraLightDepthGenerator().to(self.device)
            self.discriminator = LightweightDiscriminator().to(self.device)
        else:
            self.generator = StandardDepthGenerator().to(self.device)
            self.discriminator = StandardDiscriminator().to(self.device)
        
        # 打印模型信息
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"   生成器参数: {g_params:,}")
        print(f"   判别器参数: {d_params:,}")
    
    def _create_optimizers(self):
        """创建优化器"""
        print("⚙️  创建优化器...")
        
        lr = self.config['learning_rate']
        
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )
        
        # 学习率调度器
        self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_g, mode='min', patience=5, factor=0.5
        )
        self.scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_d, mode='min', patience=5, factor=0.5
        )
    
    def _create_losses(self):
        """创建损失函数"""
        print("📊 创建损失函数...")
        
        # 光度一致性损失
        self.photometric_loss = PhotometricConsistencyLoss(
            ssim_weight=0.8,
            l1_weight=0.2,
            occlusion_handling=False  # TUM数据集较简单，暂时关闭
        ).to(self.device)
        
        # 对抗损失
        self.adversarial_loss = nn.MSELoss()
        
        # L1损失
        self.l1_loss = nn.L1Loss()
        
        # 损失权重
        self.loss_weights = {
            'adversarial': 1.0,
            'l1': 100.0,
            'photometric': 10.0
        }
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        print("📂 创建数据加载器...")
        
        self.train_loader, self.val_loader, self.test_loader = create_tum_dataloaders(
            data_root=self.config['data_root'],
            sequence=self.config['sequence'],
            batch_size=self.config['batch_size'],
            image_size=self.config['image_size'],
            num_workers=self.config['num_workers'],
            stride=self.config['frame_stride']
        )
        
        print(f"   训练集: {len(self.train_loader.dataset)} 样本")
        print(f"   验证集: {len(self.val_loader.dataset)} 样本")
        print(f"   测试集: {len(self.test_loader.dataset)} 样本")
    
    def train_step(self, batch):
        """单步训练"""
        # 数据准备
        rgb_current = batch['rgb_current'].to(self.device)
        rgb_next = batch['rgb_next'].to(self.device)
        real_depth = batch['real_depth'].to(self.device)
        noisy_depth = batch['noisy_depth'].to(self.device)
        intrinsics = batch['intrinsics'].to(self.device)
        pose_transform = batch['pose_transform'].to(self.device)
        
        batch_size = rgb_current.shape[0]
        
        # ========== 训练判别器 ==========
        self.discriminator.train()
        self.optimizer_d.zero_grad()
        
        # 生成假深度图
        with torch.no_grad():
            fake_depth = self.generator(noisy_depth, rgb_current)
        
        # 判别真假
        real_pred = self.discriminator(real_depth, rgb_current)
        fake_pred = self.discriminator(fake_depth.detach(), rgb_current)
        
        # 判别器损失
        d_real_loss = sum(
            self.adversarial_loss(pred, torch.ones_like(pred))
            for pred in real_pred
        ) / len(real_pred)
        
        d_fake_loss = sum(
            self.adversarial_loss(pred, torch.zeros_like(pred))
            for pred in fake_pred
        ) / len(fake_pred)
        
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        
        d_loss.backward()
        self.optimizer_d.step()
        
        # ========== 训练生成器 ==========
        self.generator.train()
        self.optimizer_g.zero_grad()
        
        # 生成深度图
        fake_depth = self.generator(noisy_depth, rgb_current)
        
        # 对抗损失
        fake_pred_for_g = self.discriminator(fake_depth, rgb_current)
        g_adv_loss = sum(
            self.adversarial_loss(pred, torch.ones_like(pred))
            for pred in fake_pred_for_g
        ) / len(fake_pred_for_g)
        
        # L1重建损失
        g_l1_loss = self.l1_loss(fake_depth, real_depth)
        
        # 光度一致性损失
        photometric_losses = self.photometric_loss(
            refined_depth=fake_depth,
            rgb_current=rgb_current,
            rgb_next=rgb_next,
            intrinsics=intrinsics,
            pose_t_to_t1=pose_transform
        )
        g_photo_loss = photometric_losses['photometric_loss']
        
        # 总损失
        g_loss = (
            self.loss_weights['adversarial'] * g_adv_loss +
            self.loss_weights['l1'] * g_l1_loss +
            self.loss_weights['photometric'] * g_photo_loss
        )
        
        g_loss.backward()
        self.optimizer_g.step()
        
        # 返回损失
        return {
            'g_loss': g_loss.item(),
            'g_adv': g_adv_loss.item(),
            'g_l1': g_l1_loss.item(),
            'g_photo': g_photo_loss.item(),
            'd_loss': d_loss.item(),
            'valid_pixels': photometric_losses['valid_pixel_ratio'].item()
        }
    
    def validate(self):
        """验证"""
        self.generator.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                rgb_current = batch['rgb_current'].to(self.device)
                real_depth = batch['real_depth'].to(self.device)
                noisy_depth = batch['noisy_depth'].to(self.device)
                
                # 生成深度图
                fake_depth = self.generator(noisy_depth, rgb_current)
                
                # 计算L1损失
                l1_loss = self.l1_loss(fake_depth, real_depth)
                val_losses.append(l1_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # 更新学习率
        self.scheduler_g.step(avg_val_loss)
        self.scheduler_d.step(avg_val_loss)
        
        return avg_val_loss
    
    def train(self, num_epochs):
        """主训练循环"""
        print(f"\n🎯 开始训练 {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # 训练
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                
                # 更新进度条
                pbar.set_postfix({
                    'G': f"{losses['g_loss']:.3f}",
                    'D': f"{losses['d_loss']:.3f}",
                    'L1': f"{losses['g_l1']:.3f}",
                    'Photo': f"{losses['g_photo']:.3f}"
                })
                
                self.step += 1
            
            # 计算平均损失
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([l[key] for l in epoch_losses])
            
            # 验证
            val_loss = self.validate()
            
            # 打印结果
            print(f"\nEpoch {epoch} 完成:")
            print(f"  训练损失: G={avg_losses['g_loss']:.4f}, D={avg_losses['d_loss']:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  有效像素: {avg_losses['valid_pixels']:.3f}")
            print(f"  学习率: {self.optimizer_g.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f"  ✅ 保存最佳模型 (val_loss: {val_loss:.4f})")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_d_state': self.optimizer_d.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        save_dir = Path('checkpoints')
        save_dir.mkdir(exist_ok=True)
        torch.save(checkpoint, save_dir / filename)
    
    def test(self):
        """测试模型"""
        print("\n📊 测试模型...")
        
        self.generator.eval()
        
        test_results = {
            'l1_errors': [],
            'rmse_errors': [],
            'abs_rel_errors': []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="测试"):
                rgb_current = batch['rgb_current'].to(self.device)
                real_depth = batch['real_depth'].to(self.device)
                noisy_depth = batch['noisy_depth'].to(self.device)
                
                # 生成深度图
                fake_depth = self.generator(noisy_depth, rgb_current)
                
                # 计算误差
                l1_error = torch.abs(fake_depth - real_depth).mean().item()
                rmse_error = torch.sqrt(((fake_depth - real_depth) ** 2).mean()).item()
                abs_rel_error = (torch.abs(fake_depth - real_depth) / real_depth.clamp(min=0.1)).mean().item()
                
                test_results['l1_errors'].append(l1_error)
                test_results['rmse_errors'].append(rmse_error)
                test_results['abs_rel_errors'].append(abs_rel_error)
        
        # 打印结果
        print("\n测试结果:")
        print(f"  L1误差: {np.mean(test_results['l1_errors']):.4f} ± {np.std(test_results['l1_errors']):.4f}")
        print(f"  RMSE: {np.mean(test_results['rmse_errors']):.4f} ± {np.std(test_results['rmse_errors']):.4f}")
        print(f"  相对误差: {np.mean(test_results['abs_rel_errors']):.4f} ± {np.std(test_results['abs_rel_errors']):.4f}")
        
        return test_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用TUM数据集训练深度精修GAN")
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='data/tum_rgbd',
                        help='TUM数据集根目录')
    parser.add_argument('--sequence', type=str, default='rgbd_dataset_freiburg1_desk',
                        help='TUM序列名称')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批大小')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='图像尺寸 H W')
    
    # 模型参数
    parser.add_argument('--model_size', choices=['light', 'standard'], 
                        default='light', help='模型大小')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--frame_stride', type=int, default=1, help='帧间隔')
    parser.add_argument('--test_only', action='store_true', help='仅测试')
    parser.add_argument('--checkpoint', type=str, help='加载检查点')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'data_root': args.data_root,
        'sequence': args.sequence,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'image_size': tuple(args.image_size),
        'model_size': args.model_size,
        'num_workers': args.num_workers,
        'frame_stride': args.frame_stride
    }
    
    print("🚀 深度精修GAN - TUM训练")
    print(f"配置: {config}")
    
    # 创建训练器
    trainer = TUMTrainer(config)
    
    # 加载检查点
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        trainer.generator.load_state_dict(checkpoint['generator_state'])
        trainer.discriminator.load_state_dict(checkpoint['discriminator_state'])
        trainer.optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
        trainer.optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
        trainer.epoch = checkpoint['epoch']
        trainer.step = checkpoint['step']
        print(f"✅ 从检查点恢复: epoch {trainer.epoch}")
    
    # 训练或测试
    if args.test_only:
        trainer.test()
    else:
        trainer.train(args.epochs)
        trainer.test()
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()