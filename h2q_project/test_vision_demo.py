# test_vision_demo.py

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from h2q.knot_kernel import H2Q_Knot_Kernel
from h2q.hierarchical_decoder import ConceptDecoder
from tools.vision_loader import get_vision_dataloader

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
VISION_KERNEL_PATH = "h2q_vision_kernel.pth"
VISION_DECODER_PATH = "h2q_vision_decoder.pth"
VOCAB_SIZE = 257

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def test_cifar_reconstruction():
    print(f"🖼️ [H2Q-Vision] 启动 CIFAR-10 还原测试...")
    
    # 1. 加载模型
    kernel = H2Q_Knot_Kernel(max_dim=256, vocab_size=VOCAB_SIZE, depth=6).to(DEVICE)
    decoder = ConceptDecoder(dim=256, vocab_size=VOCAB_SIZE, stride=1).to(DEVICE)
    
    if os.path.exists(VISION_KERNEL_PATH):
        kernel.load_state_dict(torch.load(VISION_KERNEL_PATH))
        decoder.load_state_dict(torch.load(VISION_DECODER_PATH))
        print("✅ 权重加载成功")
    else:
        print("❌ 权重未找到，请先训练！")
        return
    
    kernel.eval()
    decoder.eval()
    
    # 2. 加载测试集 (Test Split)
    loader = get_vision_dataloader(split="test", batch_size=1)
    
    # 随机抽取一张
    data_iter = iter(loader)
    original_bytes = next(data_iter).to(DEVICE) # [1, 3072]
    
    # 3. H2Q 处理
    with torch.no_grad():
        features, _ = kernel(original_bytes, return_features=True)
        logits = decoder(features)
        pred_bytes = torch.argmax(logits, dim=-1) # [1, 3072]
        
    # 4. 数据后处理
    orig_np = original_bytes.cpu().numpy().reshape(32, 32, 3).astype(np.uint8)
    pred_np = pred_bytes.cpu().numpy().reshape(32, 32, 3).astype(np.uint8)
    
    # 5. 计算指标
    psnr = calculate_psnr(orig_np, pred_np)
    print(f"📊 还原 PSNR: {psnr:.2f} dB")
    
    # 6. 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original (CIFAR-10)")
    axes[0].axis('off')
    
    axes[1].imshow(pred_np)
    axes[1].set_title(f"H2Q Restored\nPSNR: {psnr:.2f} dB")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("vision_result.png")
    print("✅ 对比图已保存为: vision_result.png")
    plt.show()

if __name__ == "__main__":
    test_cifar_reconstruction()