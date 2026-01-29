#!/usr/bin/env python3
"""
简单的Ollama桥接测试
"""

from ollama_bridge import OllamaBridge, OllamaConfig

def test_ollama_direct():
    """直接测试Ollama桥接"""
    config = OllamaConfig(
        host="http://localhost:11434",
        model_name="deepseek-coder-v2:236b"
    )

    bridge = OllamaBridge(config)

    # 检查状态
    print("检查Ollama状态...")
    status = bridge.check_ollama_status()
    print(f"Ollama状态: {status}")

    if status:
        # 测试简单推理
        print("\n测试简单推理...")
        result = bridge.hot_start_inference(
            model_name="deepseek-coder-v2:236b",
            prompt="Hello, please respond with 'Hello World'",
            max_tokens=50
        )

        print(f"推理结果: {result}")
        print(f"响应内容: '{result.get('response', 'NO RESPONSE')}'")
        print(f"响应长度: {len(result.get('response', ''))}")

if __name__ == "__main__":
    test_ollama_direct()