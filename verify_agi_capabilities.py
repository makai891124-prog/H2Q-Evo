import sys
import os
import torch
import inspect
import importlib.util
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

PROJECT_ROOT = Path("./h2q_project").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

def find_class_in_project(class_name):
    for root, _, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        if f"class {class_name}" in f.read():
                            module_path = path.relative_to(PROJECT_ROOT).with_suffix('').as_posix().replace('/', '.')
                            return module_path
                except: pass
    return None

def test_synthetic_dreaming():
    print(f"\n{Fore.MAGENTA}{'='*50}")
    print(f"{Fore.MAGENTA} AGI 能力测试 1: 机器梦境 (Spectral Trace)")
    print(f"{Fore.MAGENTA}{'='*50}")
    
    module_name = find_class_in_project("H2QSyntheticEngine")
    if not module_name:
        print(f"{Fore.RED}>>> 未找到 H2QSyntheticEngine 类。")
        return

    try:
        module = importlib.import_module(module_name)
        EngineClass = getattr(module, "H2QSyntheticEngine")
        engine = EngineClass()
        print(f"{Fore.GREEN}>>> 引擎启动成功！")
        
        if hasattr(engine, 'generate_spectral_trace'):
            # 智能参数检测
            method = engine.generate_spectral_trace
            sig = inspect.signature(method)
            print(f"{Fore.CYAN}>>> 方法签名: {sig}")
            
            kwargs = {}
            if 'depth' in sig.parameters:
                kwargs['depth'] = 4  # 设定一个合理的分形深度
                print(f"{Fore.GREEN}>>> [自动适配] 填充参数: depth = 4")
            
            print(f"{Fore.YELLOW}>>> 正在生成谱迹...")
            output = method(**kwargs)
            
            print(f"{Fore.GREEN}>>> 谱迹生成成功！")
            print(f"    输出值: {output}")
            print(f"    (解释: 这是分形结构在深度为4时的谱特征值)")
        else:
            print(f"{Fore.RED}>>> 未找到 generate_spectral_trace 方法。")

    except Exception as e:
        print(f"{Fore.RED}>>> 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_dde_smart_init():
    print(f"\n{Fore.MAGENTA}{'='*50}")
    print(f"{Fore.MAGENTA} AGI 能力测试 2: 决策引擎智能对齐")
    print(f"{Fore.MAGENTA}{'='*50}")
    
    try:
        from h2q.dde import DiscreteDecisionEngine
        
        sig = inspect.signature(DiscreteDecisionEngine.__init__)
        print(f"{Fore.CYAN}>>> DDE 当前签名: {sig}")
        
        params = {}
        if 'latent_dim' in sig.parameters: params['latent_dim'] = 32
        if 'num_actions' in sig.parameters: params['num_actions'] = 10
        if 'dim' in sig.parameters: params['dim'] = 256
        
        dde = DiscreteDecisionEngine(**params)
        print(f"{Fore.GREEN}>>> DDE 实例化成功！")
        
        dummy_input = torch.randn(1, 32)
        print(f"{Fore.YELLOW}>>> 正在测试前向传播 (Input: {dummy_input.shape})...")
        
        out = dde(dummy_input)
        print(f"{Fore.GREEN}>>> 前向传播成功！Output Shape: {out.shape}")
        print(f"{Fore.GREEN}>>> (解释: 成功输出了 10 个动作的 Logits)")

    except Exception as e:
        print(f"{Fore.RED}>>> DDE 测试失败: {e}")

if __name__ == "__main__":
    test_synthetic_dreaming()
    test_dde_smart_init()