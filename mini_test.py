#!/usr/bin/env python3
"""迷你AGI系统测试"""

import torch
import torch.nn as nn
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MINI-TEST')

class MiniEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32*16, 128)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))

class MiniSystem:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.encoder = MiniEncoder().to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters())

    async def run_test(self):
        logger.info('🎯 开始迷你AGI系统测试')

        # 测试数据
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32).to(self.device)

        # 前向传播
        output = self.encoder(x)
        logger.info(f'✅ 前向传播成功，输出形状: {output.shape}')

        # 训练步骤
        target = torch.randn(batch_size, 128).to(self.device)
        loss_fn = nn.MSELoss()

        for step in range(3):
            self.optimizer.zero_grad()
            output = self.encoder(x)
            loss = loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            logger.info(f'📊 步骤 {step}, 损失: {loss.item():.4f}')

        logger.info('🎯 迷你AGI系统测试完成')

async def main():
    system = MiniSystem()
    await system.run_test()

if __name__ == "__main__":
    asyncio.run(main())