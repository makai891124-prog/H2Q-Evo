#!/usr/bin/env python3
"""
H2Q-Evo 本地模型推理演示
展示训练后的模型能力
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import json

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
H2Q_PROJECT = PROJECT_ROOT / "h2q_project"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(H2Q_PROJECT))


class LocalInferenceModel:
    """本地推理模型"""

    def __init__(self, model_path: Path = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab_size = 256  # ASCII字符

        # 尝试加载训练好的模型
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            # 使用默认模型
            self._init_default_model()

        print(f"🧠 本地推理模型已加载 | 设备: {self.device}")

    def _init_default_model(self):
        """初始化默认模型"""
        embed_dim = 256
        n_heads = 8
        n_layers = 6

        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, embed_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    embed_dim, n_heads, batch_first=True, dropout=0.1
                ),
                num_layers=n_layers
            ),
            nn.Linear(embed_dim, self.vocab_size)
        ).to(self.device)

    def load_model(self, model_path: Path):
        """加载训练好的模型"""
        try:
            # 初始化模型结构
            self._init_default_model()

            # 加载权重
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✓ 模型权重已加载: {model_path}")
        except Exception as e:
            print(f"⚠️ 模型加载失败，使用默认模型: {e}")
            self._init_default_model()

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """生成文本"""
        self.model.eval()

        # 编码提示
        tokens = [ord(c) % self.vocab_size for c in prompt]
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        generated = prompt

        with torch.no_grad():
            for _ in range(max_length):
                # 获取预测
                outputs = self.model(input_ids)
                next_token_logits = outputs[0, -1, :] / temperature

                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # 添加到序列
                next_char = chr(next_token % 128)  # 限制到ASCII范围
                generated += next_char

                # 更新输入
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

                # 限制长度
                if len(input_ids[0]) >= 512:
                    break

        return generated


def demonstrate_capabilities():
    """演示模型能力"""
    print("\n" + "="*60)
    print("🧠 H2Q-Evo 本地模型推理演示")
    print("="*60)
    print("🛡️ 安全保证：完全离线，无联网")
    print("="*60 + "\n")

    # 加载模型
    model_path = PROJECT_ROOT / "h2q_project" / "h2q_trained_model.pt"
    model = LocalInferenceModel(model_path)

    # 测试提示
    test_prompts = [
        "人工智能是",
        "机器学习",
        "量子计算可以",
        "深度学习"
    ]

    print("📝 生成文本演示:")
    print("-" * 40)

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎯 提示 {i}: {prompt}")
        generated = model.generate_text(prompt, max_length=50, temperature=0.8)
        print(f"🤖 生成: {generated}")
        print("-" * 40)

    # 显示训练统计
    try:
        with open(PROJECT_ROOT / "training_log.json", 'r', encoding='utf-8') as f:
            logs = json.load(f)

        if logs:
            latest_log = logs[-1]
            print("\n📊 最新训练统计:")
            print(f"  📅 轮次: {latest_log['epoch']}")
            print(f"  📉 损失: {latest_log['train_loss']:.4f}")
            if latest_log.get('perplexity'):
                print(f"  🎯 困惑度: {latest_log['perplexity']:.2f}")
            print(f"  ⏱️ 轮次时间: {latest_log['epoch_time']:.2f} 秒")
            print(f"  📊 总时间: {latest_log['total_time']:.2f} 秒")
    except FileNotFoundError:
        print("\n⚠️ 未找到训练日志")

    # 显示进化统计
    try:
        with open(PROJECT_ROOT / "evolution_stats.json", 'r', encoding='utf-8') as f:
            stats = json.load(f)

        print("\n🧬 进化统计:")
        print(f"  🔬 总进化次数: {stats['total_evolutions']}")
        print(f"  ✅ 成功进化: {stats['successful_evolutions']}")
        print(f"  ❌ 失败进化: {stats['failed_evolutions']}")
        print(f"  📊 平均改进: {stats['average_improvement']:.4f}")
    except FileNotFoundError:
        print("\n⚠️ 未找到进化统计")

    print("\n🎉 演示完成！")
    print("💡 提示：模型已通过本地训练进化，可以安全离线使用")


def interactive_mode():
    """交互模式"""
    print("\n" + "="*60)
    print("💬 H2Q-Evo 交互式对话")
    print("="*60)
    print("🛡️ 安全保证：完全离线，无联网")
    print("输入 'quit' 退出")
    print("="*60 + "\n")

    # 加载模型
    model_path = PROJECT_ROOT / "h2q_project" / "h2q_trained_model.pt"
    model = LocalInferenceModel(model_path)

    while True:
        try:
            user_input = input("👤 您: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            if user_input:
                print("🤖 AI: ", end="", flush=True)
                response = model.generate_text(user_input, max_length=100, temperature=0.7)
                print(response)
                print()

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="H2Q-Evo 本地模型推理演示")
    parser.add_argument("--mode", choices=["demo", "interactive"], default="demo",
                       help="运行模式：demo(演示) 或 interactive(交互)")
    parser.add_argument("--model", type=str,
                       help="模型文件路径（默认为自动查找）")

    args = parser.parse_args()

    if args.mode == "demo":
        demonstrate_capabilities()
    elif args.mode == "interactive":
        interactive_mode()


if __name__ == "__main__":
    main()