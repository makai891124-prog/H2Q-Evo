#!/usr/bin/env python3
"""
简化的AGI训练脚本 - 使用本地模型进行训练
"""
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import json
from datetime import datetime
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('SimpleAGI')

class SimpleAGIDataset(Dataset):
    """简化的AGI数据集"""

    def __init__(self, size=100):
        self.size = size
        self.data = []

        # 生成简单的数据
        for i in range(size):
            # 创建简单的序列数据
            input_seq = torch.randn(10, 32)  # 10个时间步，每个32维
            target = torch.randn(10, 32)     # 对应的目标

            self.data.append({
                'input': input_seq,
                'target': target
            })

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

class SimpleAGIModel(nn.Module):
    """简化的AGI模型"""

    def __init__(self, input_dim=32, hidden_dim=64, output_dim=32):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        output = self.output_layer(decoded)
        return output

class SimpleAGITrainer:
    """简化的AGI训练器"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleAGIModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # 训练配置
        self.num_epochs = 10
        self.batch_size = 8

        # 数据集
        self.train_dataset = SimpleAGIDataset(200)
        self.val_dataset = SimpleAGIDataset(50)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # 训练历史
        self.train_losses = []
        self.val_losses = []

        logger.info(f"简化的AGI训练器初始化完成，使用设备: {self.device}")

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0

        for batch in self.train_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def train(self):
        """训练模型"""
        logger.info("开始AGI模型训练...")

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            # 训练
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
                logger.info(f"保存最佳模型 (验证损失: {val_loss:.4f})")

        logger.info("训练完成！")
        return self.train_losses, self.val_losses

    def save_model(self, filename):
        """保存模型"""
        model_path = Path("checkpoints") / filename
        model_path.parent.mkdir(exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epoch': len(self.train_losses)
        }, model_path)

        logger.info(f"模型已保存到: {model_path}")

    def save_training_report(self):
        """保存训练报告"""
        report = {
            'training_summary': {
                'total_epochs': len(self.train_losses),
                'final_train_loss': self.train_losses[-1] if self.train_losses else None,
                'final_val_loss': self.val_losses[-1] if self.val_losses else None,
                'best_val_loss': min(self.val_losses) if self.val_losses else None,
                'device': str(self.device),
                'model_type': 'SimpleAGI-LSTM'
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            },
            'timestamp': datetime.now().isoformat(),
            'algorithm_used': 'simplified_agi_with_lstm'
        }

        report_path = Path("reports") / "training_report.json"
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"训练报告已保存到: {report_path}")
        return report

def main():
    """主函数"""
    print("🚀 开始简化的AGI训练实验")
    print("=" * 50)

    try:
        # 初始化训练器
        trainer = SimpleAGITrainer()

        # 开始训练
        train_losses, val_losses = trainer.train()

        # 保存训练报告
        report = trainer.save_training_report()

        print("\n" + "=" * 50)
        print("🎉 训练完成！")
        print(f"📊 最终训练损失: {train_losses[-1]:.4f}")
        print(f"📊 最终验证损失: {val_losses[-1]:.4f}")
        print(f"📊 最佳验证损失: {min(val_losses):.4f}")
        print(f"💾 模型已保存到: checkpoints/")
        print(f"📄 报告已保存到: reports/training_report.json")

        return True

    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)