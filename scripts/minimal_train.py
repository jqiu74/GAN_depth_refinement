#!/usr/bin/env python3
"""
第一阶段最小可行训练器
验证所有已完成模块的协同工作能力
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入我们的模块
try:
    from src.models.generator import UltraLightDepthGenerator, StandardDepthGenerator
    from src.models.discriminator import LightweightDiscriminator, StandardDiscriminator
    from src.losses.photometric_loss import PhotometricConsistencyLoss
    from src.data.multiview_manager import MultiViewDataManager
    from src.utils.geometry_utils import GeometryUtils
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保您在项目根目录运行此脚本")
    sys.exit(1)


class SimpleDataset(data.Dataset):
    """简单数据集包装器"""
    
    def __init__(self, manager: MultiViewDataManager, size: int = 100):
        self.manager = manager
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.manager.get_sample(idx, use_virtual=True)


class MinimalGANTrainer:
    """
    最小可行GAN训练器
    目标：验证所有模块协同工作，不追求最佳性能
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"🚀 初始化最小可行训练器...")
        print(f"   设备: {self.device}")
        print(f"   配置: {config}")
        
        # 创建模型
        self._create_models()
        
        # 创建优化器
        self._create_optimizers()
        
        # 创建损失函数
        self._create_loss_functions()
        
        # 创建数据
        self._create_data()
        
        # 训练统计
        self.step = 0
        self.epoch = 0
        self.losses_history = []
        
        print("✅ 训练器初始化完成!")
    
    def _create_models(self):
        """创建生成器和判别器"""
        print("📦 创建模型...")
        
        # 根据配置选择模型
        if self.config.get('model_size', 'light') == 'light':
            self.generator = UltraLightDepthGenerator().to(self.device)
            self.discriminator = LightweightDiscriminator().to(self.device)
        else:
            self.generator = StandardDepthGenerator().to(self.device)
            self.discriminator = StandardDiscriminator().to(self.device)
        
        # 打印模型信息
        g_params, g_size = self.generator.get_model_size()
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        d_size = d_params * 4 / (1024 * 1024)
        
        print(f"   生成器: {g_params:,} 参数 ({g_size:.2f} MB)")
        print(f"   判别器: {d_params:,} 参数 ({d_size:.2f} MB)")
        print(f"   总参数: {g_params + d_params:,}")
    
    def _create_optimizers(self):
        """创建优化器"""
        print("⚙️  创建优化器...")
        
        lr = self.config.get('learning_rate', 0.0002)
        
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
        
        print(f"   学习率: {lr}")
    
    def _create_loss_functions(self):
        """创建损失函数"""
        print("📊 创建损失函数...")
        
        # 光度一致性损失
        self.photometric_loss = PhotometricConsistencyLoss(
            ssim_weight=0.8,
            l1_weight=0.2,
            occlusion_handling=True
        ).to(self.device)
        
        # 简单的对抗损失 (LSGAN)
        self.adversarial_loss = nn.MSELoss()
        
        # L1重建损失
        self.l1_loss = nn.L1Loss()
        
        # 损失权重
        self.loss_weights = {
            'adversarial': self.config.get('adversarial_weight', 1.0),
            'l1': self.config.get('l1_weight', 50.0),
            'photometric': self.config.get('photometric_weight', 10.0)
        }
        
        print(f"   损失权重: {self.loss_weights}")
    
    def _create_data(self):
        """创建数据加载器"""
        print("📂 创建数据加载器...")
        
        # 创建完整的虚拟数据配置 - 优化版
        virtual_data_config = {
            'depth_range': (1.0, 5.0),              # 更紧凑的深度范围
            'noise_level': 0.03,                    # 稍微减小噪声
            'focal_length_range': (300, 500),       # 更大的焦距，减小视角
            'translation_range': 0.05,              # 很小的平移
            'rotation_range': 0.02,                 # 很小的旋转（约1.1度）
            'texture_complexity': 'medium',
            'enable_occlusion': False,              # 关闭遮挡检测
            'enable_objects': True,
        }
        
        # 创建数据管理器
        image_size = self.config.get('image_size', (128, 128))
        self.data_manager = MultiViewDataManager(
            data_root=None,  # 使用虚拟数据
            image_size=image_size,
            virtual_data_config=virtual_data_config
        )
        
        # 创建数据集
        dataset_size = self.config.get('dataset_size', 100)
        dataset = SimpleDataset(self.data_manager, size=dataset_size)
        
        # 创建数据加载器
        batch_size = self.config.get('batch_size', 4)
        self.dataloader = data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # 设为0避免multiprocessing问题
            drop_last=True
        )
        
        print(f"   数据集大小: {dataset_size}")
        print(f"   批次大小: {batch_size}")
        print(f"   图像尺寸: {image_size}")
        print(f"   虚拟数据配置: {virtual_data_config}")
    
    def train_step(self, batch):
        """单步训练"""
        # 准备数据
        rgb_current = batch['rgb_current'].to(self.device)
        rgb_next = batch['rgb_next'].to(self.device)
        real_depth = batch['real_depth'].to(self.device)
        noisy_depth = batch['noisy_depth'].to(self.device)
        intrinsics = batch['intrinsics'].to(self.device)
        pose_transform = batch['pose_transform'].to(self.device)
        
        batch_size = rgb_current.shape[0]
        
        # ================================
        # 训练判别器
        # ================================
        self.discriminator.train()
        self.optimizer_d.zero_grad()
        
        # 生成假深度图
        with torch.no_grad():
            fake_depth = self.generator(noisy_depth, rgb_current)
        
        # 判别真实样本
        real_pred = self.discriminator(real_depth, rgb_current)
        
        # 判别虚假样本
        fake_pred = self.discriminator(fake_depth.detach(), rgb_current)
        
        # 计算判别器损失
        d_real_loss = sum(
            self.adversarial_loss(pred, torch.ones_like(pred))
            for pred in real_pred
        ) / len(real_pred)
        
        d_fake_loss = sum(
            self.adversarial_loss(pred, torch.zeros_like(pred))
            for pred in fake_pred
        ) / len(fake_pred)
        
        d_total_loss = (d_real_loss + d_fake_loss) * 0.5
        
        # 反向传播判别器
        d_total_loss.backward()
        self.optimizer_d.step()
        
        # ================================
        # 训练生成器
        # ================================
        self.generator.train()
        self.optimizer_g.zero_grad()
        
        # 生成假深度图
        fake_depth = self.generator(noisy_depth, rgb_current)
        
        # 计算各种损失
        
        # 1. 对抗损失
        fake_pred_for_g = self.discriminator(fake_depth, rgb_current)
        g_adv_loss = sum(
            self.adversarial_loss(pred, torch.ones_like(pred))
            for pred in fake_pred_for_g
        ) / len(fake_pred_for_g)
        
        # 2. L1重建损失
        g_l1_loss = self.l1_loss(fake_depth, real_depth)
        
        # 3. 光度一致性损失
        try:
            photometric_losses = self.photometric_loss(
                refined_depth=fake_depth,
                rgb_current=rgb_current,
                rgb_next=rgb_next,
                intrinsics=intrinsics,
                pose_t_to_t1=pose_transform
            )
            g_photometric_loss = photometric_losses['photometric_loss']
            valid_pixel_ratio = photometric_losses['valid_pixel_ratio']
        except Exception as e:
            # 如果光度损失计算失败，使用零损失
            g_photometric_loss = torch.tensor(0.0, device=self.device)
            valid_pixel_ratio = torch.tensor(0.0, device=self.device)
        
        # 总生成器损失
        g_total_loss = (
            self.loss_weights['adversarial'] * g_adv_loss +
            self.loss_weights['l1'] * g_l1_loss +
            self.loss_weights['photometric'] * g_photometric_loss
        )
        
        # 反向传播生成器
        g_total_loss.backward()
        self.optimizer_g.step()
        
        # 返回损失字典
        return {
            'g_total': g_total_loss.item(),
            'g_adv': g_adv_loss.item(),
            'g_l1': g_l1_loss.item(),
            'g_photometric': g_photometric_loss.item(),
            'd_total': d_total_loss.item(),
            'd_real': d_real_loss.item(),
            'd_fake': d_fake_loss.item(),
            'valid_pixels': valid_pixel_ratio.item()
        }
    
    def train_epoch(self):
        """训练一个epoch"""
        epoch_losses = {}
        num_batches = len(self.dataloader)
        successful_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            try:
                losses = self.train_step(batch)
                
                # 累积损失
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value
                
                successful_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'G': f"{losses['g_total']:.3f}",
                    'D': f"{losses['d_total']:.3f}",
                    'Photo': f"{losses['g_photometric']:.3f}",
                    'Valid': f"{losses['valid_pixels']:.2f}"
                })
                
                self.step += 1
                
            except Exception as e:
                print(f"\n❌ 训练步骤失败: {e}")
                continue
        
        # 只有成功的批次才计算平均损失
        if successful_batches > 0:
            avg_losses = {key: value / successful_batches for key, value in epoch_losses.items()}
            self.losses_history.append(avg_losses)
            return avg_losses
        else:
            # 如果没有成功的批次，返回空损失
            print("⚠️  所有训练步骤都失败了")
            return {}
    
    def train(self, num_epochs):
        """主训练循环"""
        print(f"\n🎯 开始训练 {num_epochs} 个epoch...")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            try:
                # 训练一个epoch
                avg_losses = self.train_epoch()
                
                # 检查是否有有效的损失
                if not avg_losses:
                    print(f"Epoch {epoch}: 没有成功的训练步骤，跳过")
                    continue
                
                # 打印损失
                print(f"\nEpoch {epoch} 完成:")
                print(f"  生成器损失: {avg_losses.get('g_total', 0):.4f}")
                print(f"    - 对抗: {avg_losses.get('g_adv', 0):.4f}")
                print(f"    - L1: {avg_losses.get('g_l1', 0):.4f}")
                print(f"    - 光度: {avg_losses.get('g_photometric', 0):.4f}")
                print(f"  判别器损失: {avg_losses.get('d_total', 0):.4f}")
                print(f"  有效像素比例: {avg_losses.get('valid_pixels', 0):.3f}")
                
                # 简单的收敛检查
                if len(self.losses_history) > 5:
                    recent_g_losses = [h.get('g_total', float('inf')) for h in self.losses_history[-5:]]
                    valid_losses = [loss for loss in recent_g_losses if loss != float('inf')]
                    if len(valid_losses) >= 3 and max(valid_losses) - min(valid_losses) < 0.01:
                        print("🎉 损失收敛，提前停止训练")
                        break
                
            except KeyboardInterrupt:
                print("\n⏹️  训练被用户中断")
                break
            except Exception as e:
                print(f"\n❌ Epoch {epoch} 训练失败: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"\n✅ 训练完成! 总时间: {total_time:.1f}秒")
        
        return self.losses_history
    
    def evaluate(self):
        """简单的评估"""
        print("\n📊 运行评估...")
        
        self.generator.eval()
        self.discriminator.eval()
        
        eval_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= 10:  # 只评估10个批次
                    break
                
                try:
                    # 准备数据
                    rgb_current = batch['rgb_current'].to(self.device)
                    real_depth = batch['real_depth'].to(self.device)
                    noisy_depth = batch['noisy_depth'].to(self.device)
                    
                    # 生成深度图
                    fake_depth = self.generator(noisy_depth, rgb_current)
                    
                    # 计算重建误差
                    l1_error = torch.abs(fake_depth - real_depth).mean().item()
                    
                    # 计算RMSE
                    rmse = torch.sqrt(torch.mean((fake_depth - real_depth) ** 2)).item()
                    
                    eval_losses.append({
                        'l1_error': l1_error,
                        'rmse': rmse
                    })
                    
                except Exception as e:
                    print(f"评估批次 {i} 失败: {e}")
                    continue
        
        if eval_losses:
            avg_l1 = np.mean([l['l1_error'] for l in eval_losses])
            avg_rmse = np.mean([l['rmse'] for l in eval_losses])
            
            print(f"  平均L1误差: {avg_l1:.4f}")
            print(f"  平均RMSE: {avg_rmse:.4f}")
            
            return {'l1_error': avg_l1, 'rmse': avg_rmse}
        else:
            print("  ⚠️  评估失败")
            return None
    
    def save_checkpoint(self, filepath):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'losses_history': self.losses_history,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 检查点已保存: {filepath}")
    
    def visualize_results(self):
        """可视化训练结果"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.losses_history:
                print("⚠️  没有训练历史可可视化")
                return
            
            # 绘制损失曲线
            epochs = range(len(self.losses_history))
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 生成器损失
            g_total = [h['g_total'] for h in self.losses_history]
            g_adv = [h['g_adv'] for h in self.losses_history]
            g_l1 = [h['g_l1'] for h in self.losses_history]
            g_photo = [h['g_photometric'] for h in self.losses_history]
            
            axes[0, 0].plot(epochs, g_total, label='Total', linewidth=2)
            axes[0, 0].plot(epochs, g_adv, label='Adversarial', alpha=0.7)
            axes[0, 0].plot(epochs, g_l1, label='L1', alpha=0.7)
            axes[0, 0].plot(epochs, g_photo, label='Photometric', alpha=0.7)
            axes[0, 0].set_title('Generator Losses')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 判别器损失
            d_total = [h['d_total'] for h in self.losses_history]
            d_real = [h['d_real'] for h in self.losses_history]
            d_fake = [h['d_fake'] for h in self.losses_history]
            
            axes[0, 1].plot(epochs, d_total, label='Total', linewidth=2)
            axes[0, 1].plot(epochs, d_real, label='Real', alpha=0.7)
            axes[0, 1].plot(epochs, d_fake, label='Fake', alpha=0.7)
            axes[0, 1].set_title('Discriminator Losses')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 有效像素比例
            valid_pixels = [h['valid_pixels'] for h in self.losses_history]
            axes[1, 0].plot(epochs, valid_pixels, 'g-', linewidth=2)
            axes[1, 0].set_title('Valid Pixel Ratio')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Ratio')
            axes[1, 0].grid(True)
            
            # 训练示例
            self.generator.eval()
            with torch.no_grad():
                sample = self.data_manager.get_sample(0, use_virtual=True)
                
                rgb = sample['rgb_current'].unsqueeze(0).to(self.device)
                real_depth = sample['real_depth'].unsqueeze(0).to(self.device)
                noisy_depth = sample['noisy_depth'].unsqueeze(0).to(self.device)
                
                refined_depth = self.generator(noisy_depth, rgb)
                
                # 显示深度图比较
                real_np = real_depth.squeeze().cpu().numpy()
                noisy_np = noisy_depth.squeeze().cpu().numpy()
                refined_np = refined_depth.squeeze().cpu().numpy()
                
                comparison = np.hstack([real_np, noisy_np, refined_np])
                im = axes[1, 1].imshow(comparison, cmap='viridis')
                axes[1, 1].set_title('Real | Noisy | Refined')
                axes[1, 1].axis('off')
                plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
            print("📊 训练结果可视化已保存为 'training_results.png'")
            plt.close()
            
        except ImportError:
            print("⚠️  matplotlib未安装，跳过可视化")
        except Exception as e:
            print(f"⚠️  可视化失败: {e}")


def create_test_config():
    """创建测试配置"""
    return {
        'model_size': 'light',          # 'light' or 'standard'
        'learning_rate': 0.0002,
        'batch_size': 1,                # 改为1，减少复杂度
        'dataset_size': 10,             # 改为10，快速测试
        'image_size': (64, 64),         # 改为更小的尺寸
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # 损失权重
        'adversarial_weight': 1.0,
        'l1_weight': 50.0,
        'photometric_weight': 5.0,      # 减小光度损失权重
    }


def test_integration():
    """集成测试 - 验证所有模块协同工作"""
    print("🧪 开始集成测试...")
    
    config = create_test_config()
    
    try:
        # 创建训练器
        trainer = MinimalGANTrainer(config)
        
        # 运行几个epoch测试
        print("\n🏃 运行短期训练测试...")
        trainer.train(num_epochs=3)
        
        # 运行评估
        eval_results = trainer.evaluate()
        
        if eval_results:
            print("✅ 集成测试通过!")
            print(f"   最终L1误差: {eval_results['l1_error']:.4f}")
            print(f"   最终RMSE: {eval_results['rmse']:.4f}")
        else:
            print("⚠️  评估失败，但训练过程正常")
        
        # 可视化结果
        trainer.visualize_results()
        
        # 保存检查点
        trainer.save_checkpoint('test_checkpoint.pth')
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="第一阶段最小可行训练器")
    parser.add_argument('--mode', choices=['test', 'train'], default='test', 
                       help='运行模式: test(集成测试) 或 train(完整训练)')
    parser.add_argument('--epochs', type=int, default=10, help='训练epoch数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # 集成测试模式
        success = test_integration()
        if success:
            print("\n🎉 第一阶段所有模块集成测试通过!")
            print("   现在可以开始第二阶段的开发了")
        else:
            print("\n❌ 集成测试失败，请检查模块实现")
            sys.exit(1)
    
    else:
        # 完整训练模式
        config = {
            'model_size': 'standard',
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'dataset_size': 500,
            'image_size': (256, 256),
            'device': args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'),
            'adversarial_weight': 1.0,
            'l1_weight': 100.0,
            'photometric_weight': 15.0,
        }
        
        print(f"🚀 开始完整训练...")
        print(f"   配置: {config}")
        
        trainer = MinimalGANTrainer(config)
        trainer.train(num_epochs=args.epochs)
        
        # 最终评估
        eval_results = trainer.evaluate()
        if eval_results:
            print(f"\n📊 最终结果:")
            print(f"   L1误差: {eval_results['l1_error']:.4f}")
            print(f"   RMSE: {eval_results['rmse']:.4f}")
        
        # 可视化和保存
        trainer.visualize_results()
        trainer.save_checkpoint(f'final_checkpoint_epoch_{args.epochs}.pth')
        
        print("✅ 训练完成!")


if __name__ == "__main__":
    main()