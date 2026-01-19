import torch
import time

def check_compute():
    print(">>> 正在审计本地算力真实性...")
    
    # 1. 检查设备
    if not torch.backends.mps.is_available():
        print("❌ 警告：MPS (Metal) 加速不可用！模型可能在 CPU 上龟速爬行。")
    else:
        print("✅ MPS (Apple Silicon NPU) 加速已激活。")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 2. 压力测试
    print(f">>> 正在 {device} 上进行矩阵乘法压力测试...")
    try:
        # 创建两个较大的张量
        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)
        
        start = time.time()
        # 执行计算
        c = torch.matmul(a, b)
        # 强制同步 (MPS 是异步的，必须同步才能测准时间)
        if device.type == 'mps':
            torch.mps.synchronize()
        end = time.time()
        
        print(f"✅ 计算完成！耗时: {end - start:.4f} 秒")
        print(f"   结果张量均值: {c.mean().item():.4f}")
        print(">>> 结论：本地算力通道是畅通的。")
        
    except Exception as e:
        print(f"❌ 算力调用失败: {e}")

if __name__ == "__main__":
    check_compute()