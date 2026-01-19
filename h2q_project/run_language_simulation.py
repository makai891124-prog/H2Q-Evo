# run_language_simulation.py

import torch
import os
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from h2q.system import AutonomousSystem

# --- [中国区加速配置] ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def run_simulation():
    print("🚀 [H2Q] 启动语言认知模拟实验...")

    # 1. 准备“感官” (Tokenizer)
    print("📖 正在加载 GPT-2 分词器...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # 2. 初始化系统 (256维射影空间)
    # 注意：这里我们用 256 维，因为我们的记忆晶体是 256 维的
    system = AutonomousSystem(context_dim=256, action_dim=256)
    
    # 3. 注入知识 (加载晶体)
    crystal_path = "h2q_memory.pt"
    if os.path.exists(crystal_path):
        system.load_knowledge(crystal_path)
    else:
        print("❌ 错误：未找到记忆晶体，请先运行 tools/prism_converter.py")
        return

    # 4. 准备测试文本
    # 这是一段包含简单词汇和复杂概念的文本
    text = "The cat sat on the mat. However, the quantum entanglement implies a spooky action at a distance."
    print(f"\n📄 输入文本: \"{text}\"")
    
    # 将文本转换为索引 (Tokens)
    inputs = tokenizer(text, return_tensors="pt")["input_ids"] # [1, Seq_Len]
    tokens = [tokenizer.decode([t]) for t in inputs[0]] # 用于后续绘图标签
    
    print(f"🔢 Token 序列长度: {inputs.shape[1]}")

    # 5. 认知循环 (Cognitive Loop)
    eta_history = []
    
    print("\n🧠 开始逐词阅读与认知...")
    # 我们模拟一个自回归过程：系统看到前 N 个词，思考第 N+1 个词
    # 为了简化，我们直接把当前词作为“行动”，计算它带来的认知偏转
    
    # 初始化一个随机的认知上下文 (模拟“大脑一片空白”)
    current_context = torch.randn(1, 256)
    current_context = torch.nn.functional.normalize(current_context, p=2, dim=-1)

    for i in range(inputs.shape[1]):
        # 获取当前看到的词 (作为候选行动)
        # 在真实生成中，这里会有多个候选，现在我们强制它“阅读”这个词
        current_token_idx = inputs[:, i:i+1] # [1, 1]
        
        # 执行一步 (这里我们手动调用 DDE 的 forward 来获取元数据)
        # 我们传入 current_token_idx 作为 candidate_actions
        # DDE 会去晶体里查这个词对应的 256维 几何向量
        with torch.no_grad():
            _, metadata = system.dde(current_context, current_token_idx)
        
        # 获取本次的谱位移 (认知偏转角)
        eta = metadata['chosen_eta'].item()
        eta_history.append(eta)
        
        # 更新上下文：简单的移动平均，模拟“记忆残留”
        # 新上下文 = 旧上下文 * 0.8 + 新词向量 * 0.2
        # (注意：这里需要从 DDE 内部获取新词向量，为简化演示，我们略过这一步的精确实现，
        # 仅记录 eta 的变化，因为 eta 本身反映了新词与旧状态的冲突程度)
        
        print(f"   Step {i:02d}: Token = '{tokens[i]:<10}' | η (偏转角) = {eta:.4f} rad")

    # 6. 可视化结果
    print("\n📊 正在绘制认知谱图...")
    plt.figure(figsize=(12, 6))
    plt.plot(eta_history, marker='o', linestyle='-', color='#00ff00', label='Spectral Shift (η)')
    
    # 设置 X 轴标签为单词
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.title(f"Cognitive Spectral Shift Analysis\nInput: {text}")
    plt.ylabel("Cognitive Deflection (η) [Radians]")
    plt.xlabel("Token Stream")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()