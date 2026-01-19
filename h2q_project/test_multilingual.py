# test_multilingual.py

import torch
import os
from h2q.hierarchical_system import H2Q_Hierarchical_System
from h2q.hierarchical_decoder import ConceptDecoder

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SPELLING_WEIGHTS = "h2q_model_knot.pth"
HIERARCHY_WEIGHTS = "h2q_model_hierarchy.pth"
DECODER_WEIGHTS = "h2q_model_decoder.pth"

def run_test():
    print(f"🚀 [H2Q] 启动多模态/多语言压力测试...")
    
    # 加载系统 (复用之前的权重)
    encoder = H2Q_Hierarchical_System(vocab_size=257, dim=256, spelling_weights_path=SPELLING_WEIGHTS)
    decoder = ConceptDecoder(dim=256, vocab_size=257, stride=8)
    
    if os.path.exists(HIERARCHY_WEIGHTS): encoder.load_state_dict(torch.load(HIERARCHY_WEIGHTS), strict=False)
    if os.path.exists(DECODER_WEIGHTS): decoder.load_state_dict(torch.load(DECODER_WEIGHTS))
    
    encoder.to(DEVICE).eval()
    decoder.to(DEVICE).eval()
    
    # 测试用例
    test_cases = [
        ("中文测试", "H2Q架构能否理解汉字的字节流拓扑结构？这是一个关键的测试。"),
        ("代码测试", "def hello_world():\n    print('H2Q is running!')\n    return True"),
        ("混合测试", "The price is 100¥. 价格是一百元。")
    ]
    
    for name, text in test_cases:
        print(f"\n🧪 [{name}]")
        print(f"   原文: {text}")
        
        bytes_data = list(text.encode('utf-8'))
        original_size = len(bytes_data)
        pad_len = (8 - (original_size % 8)) % 8
        bytes_data += [0] * pad_len
        
        inputs = torch.tensor([bytes_data], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            _, concept_stream = encoder(inputs)
            recon_logits = decoder(concept_stream)
            pred_bytes = torch.argmax(recon_logits, dim=-1)[0].cpu().tolist()
            
        pred_bytes = pred_bytes[:original_size]
        
        try:
            recon_text = bytes(pred_bytes).decode('utf-8', errors='ignore')
        except:
            recon_text = "[解码失败]"
            
        print(f"   还原: {recon_text}")
        
        # 计算字节级准确率
        matches = sum(1 for a, b in zip(list(text.encode('utf-8')), pred_bytes) if a == b)
        acc = matches / original_size
        print(f"   ✅ 字节准确率: {acc*100:.2f}%")

if __name__ == "__main__":
    run_test()