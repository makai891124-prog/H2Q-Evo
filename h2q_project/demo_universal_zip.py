# demo_universal_zip.py

import torch
import os
import sys
from h2q.hierarchical_system import H2Q_Hierarchical_System
from h2q.hierarchical_decoder import ConceptDecoder

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SPELLING_WEIGHTS = "h2q_model_knot.pth"
HIERARCHY_WEIGHTS = "h2q_model_hierarchy.pth"
DECODER_WEIGHTS = "h2q_model_decoder.pth"

def load_system():
    print(f"🚀 [H2Q-Universal] 正在加载通用神经压缩系统...")
    
    encoder = H2Q_Hierarchical_System(vocab_size=257, dim=256, spelling_weights_path=SPELLING_WEIGHTS)
    decoder = ConceptDecoder(dim=256, vocab_size=257, stride=8)
    
    # 加载权重
    if os.path.exists(HIERARCHY_WEIGHTS):
        encoder.load_state_dict(torch.load(HIERARCHY_WEIGHTS), strict=False)
    if os.path.exists(DECODER_WEIGHTS):
        decoder.load_state_dict(torch.load(DECODER_WEIGHTS))
        
    encoder.to(DEVICE).eval()
    decoder.to(DEVICE).eval()
    return encoder, decoder

def compress_and_restore(text, encoder, decoder):
    print(f"\n📄 [输入] {text}")
    original_bytes = list(text.encode('utf-8'))
    original_size = len(original_bytes)
    print(f"   原始大小: {original_size} bytes")
    
    # 补齐
    pad_len = (8 - (original_size % 8)) % 8
    padded_bytes = original_bytes + [0] * pad_len
    inputs = torch.tensor([padded_bytes], dtype=torch.long).to(DEVICE)
    
    # 1. 压缩 (Encoding)
    with torch.no_grad():
        _, concept_stream = encoder(inputs)
    
    num_concepts = concept_stream.shape[1]
    print(f"🗜️ [压缩] 生成 {num_concepts} 个概念纽结 (压缩比 8:1)")
    
    # 2. 解压 (Decoding)
    with torch.no_grad():
        recon_logits = decoder(concept_stream)
        pred_bytes = torch.argmax(recon_logits, dim=-1)[0].cpu().tolist()
    
    # 3. 还原
    pred_bytes = pred_bytes[:original_size]
    try:
        recon_text = bytes(pred_bytes).decode('utf-8', errors='ignore')
    except:
        recon_text = "[解码错误]"
        
    print(f"📂 [还原] {recon_text}")
    
    # 验证
    if text == recon_text:
        print("✅ 完美还原 (Lossless Reconstruction)")
    else:
        diff = sum(1 for a, b in zip(text, recon_text) if a != b)
        print(f"⚠️ 存在差异 (字符误差: {diff})")
    print("-" * 50)

def main():
    encoder, decoder = load_system()
    
    print("\n✨ H2Q 通用压缩终端已就绪 ✨")
    print("请输入任意文本（中文、英文、代码）。输入 'q' 退出。")
    
    # 预设测试用例
    test_cases = [
        "H2Q架构：基于射影几何的通用智能压缩协议。",
        "def fractal_recursion(x): return x * 2",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    print("\n--- 自动基准测试 ---")
    for case in test_cases:
        compress_and_restore(case, encoder, decoder)
        
    print("\n--- 交互模式 ---")
    while True:
        try:
            user_input = input("H2Q> ")
            if user_input.lower() == 'q':
                break
            if not user_input:
                continue
            compress_and_restore(user_input, encoder, decoder)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()