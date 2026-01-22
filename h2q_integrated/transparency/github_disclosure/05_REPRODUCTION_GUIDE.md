# 如何重现我们的结果

## 前置条件

- Python 3.8+
- PyTorch 2.0+
- Hugging Face datasets
- Unix/Linux 或 macOS

## 步骤1: 克隆仓库

```bash
git clone https://github.com/H2Q-AGI/H2Q-Evo.git
cd H2Q-Evo
```

## 步骤2: 安装依赖

```bash
pip install torch transformers datasets numpy tqdm
```

## 步骤3: 下载数据

```bash
cd h2q_project/h2q/agi
python3 -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1')"
```

## 步骤4: 运行训练

```bash
PYTHONPATH=. python3 real_agi_training.py \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 6e-4 \
  --warmup_steps 2000
```

## 步骤5: 验证结果

```bash
# 检查日志
tail -100 real_logs/training_*.log

# 验证Perplexity
# 目标: 最终Perplexity 应接近 2.95
```

## 预期结果

- 训练时间: ~5小时 (取决于硬件)
- 最终Loss: ~1.09
- 最终Perplexity: ~2.95
- 总处理tokens: ~104M

## 故障排除

### 问题: 内存不足
**解决**: 降低batch_size或max_tokens

### 问题: 数据下载缓慢
**解决**: 从Hugging Face镜像下载

### 问题: 结果不匹配
**检查**:
1. PyTorch版本
2. 随机种子
3. 数据预处理步骤

## 联系支持

如遇问题: https://github.com/H2Q-AGI/H2Q-Evo/issues
