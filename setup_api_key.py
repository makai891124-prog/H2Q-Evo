#!/usr/bin/env python3
"""
H2Q API Key 安全配置工具

此脚本帮助您安全地配置 API Key，确保不会被提交到 Git。

使用方法:
    python3 setup_api_key.py

安全措施:
1. API Key 存储在 .env 文件中
2. .env 已在 .gitignore 中排除
3. 运行时从环境变量加载
"""

import os
import sys
from pathlib import Path
from getpass import getpass

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / '.env'
GITIGNORE_FILE = PROJECT_ROOT / '.gitignore'


def check_gitignore():
    """确保 .env 在 .gitignore 中."""
    if not GITIGNORE_FILE.exists():
        print("⚠️ .gitignore 文件不存在，创建中...")
        with open(GITIGNORE_FILE, 'w') as f:
            f.write(".env\n.env.local\n")
        return True
    
    with open(GITIGNORE_FILE, 'r') as f:
        content = f.read()
    
    if '.env' in content:
        print("✓ .gitignore 已包含 .env")
        return True
    else:
        print("⚠️ 添加 .env 到 .gitignore...")
        with open(GITIGNORE_FILE, 'a') as f:
            f.write("\n# 环境变量文件\n.env\n.env.local\n")
        return True


def mask_key(key: str) -> str:
    """遮蔽 API Key 显示."""
    if len(key) <= 8:
        return '*' * len(key)
    return key[:4] + '*' * (len(key) - 8) + key[-4:]


def setup_gemini_key():
    """设置 Gemini API Key."""
    print()
    print("=" * 60)
    print("       H2Q API Key 安全配置工具")
    print("=" * 60)
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                      终 极 目 标                           ║")
    print("║                                                            ║")
    print("║        训练本地可用的实时AGI系统                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # 检查 .gitignore
    check_gitignore()
    
    # 检查现有配置
    existing_key = None
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            for line in f:
                if line.startswith('GEMINI_API_KEY='):
                    existing_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                    break
    
    if existing_key and existing_key != 'your-gemini-api-key-here':
        print(f"✓ 已检测到 Gemini API Key: {mask_key(existing_key)}")
        choice = input("\n是否要更新 API Key? (y/N): ").strip().lower()
        if choice != 'y':
            print("保持现有配置。")
            return existing_key
    
    # 获取 API Key
    print("\n请输入您的 Gemini API Key:")
    print("(获取地址: https://aistudio.google.com/app/apikey)")
    print()
    
    api_key = getpass("API Key (输入时不显示): ").strip()
    
    if not api_key:
        print("✗ API Key 不能为空")
        return None
    
    # 确认
    print(f"\n您输入的 API Key: {mask_key(api_key)}")
    confirm = input("确认保存? (Y/n): ").strip().lower()
    
    if confirm == 'n':
        print("已取消。")
        return None
    
    # 读取现有 .env 内容
    env_content = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_content[key.strip()] = value.strip()
    
    # 更新 GEMINI_API_KEY
    env_content['GEMINI_API_KEY'] = api_key
    
    # 写入 .env
    with open(ENV_FILE, 'w') as f:
        f.write("# H2Q-Evo 环境配置\n")
        f.write("# 此文件不会被提交到 Git\n")
        f.write(f"# 生成时间: {__import__('datetime').datetime.now().isoformat()}\n\n")
        
        for key, value in env_content.items():
            f.write(f"{key}={value}\n")
    
    # 设置文件权限（仅所有者可读写）
    try:
        os.chmod(ENV_FILE, 0o600)
        print("✓ 已设置 .env 文件权限为 600（仅所有者可读写）")
    except:
        pass
    
    print(f"\n✓ API Key 已安全保存到 {ENV_FILE}")
    print("✓ 此文件不会被提交到 Git")
    
    return api_key


def verify_setup():
    """验证配置是否正确."""
    print("\n" + "-" * 60)
    print("验证配置...")
    
    # 检查 .env 文件
    if not ENV_FILE.exists():
        print("✗ .env 文件不存在")
        return False
    
    # 检查 API Key
    api_key = None
    with open(ENV_FILE, 'r') as f:
        for line in f:
            if line.startswith('GEMINI_API_KEY='):
                api_key = line.split('=', 1)[1].strip()
                break
    
    if not api_key or api_key == 'your-gemini-api-key-here':
        print("✗ GEMINI_API_KEY 未设置")
        return False
    
    print(f"✓ GEMINI_API_KEY: {mask_key(api_key)}")
    
    # 检查 .gitignore
    if GITIGNORE_FILE.exists():
        with open(GITIGNORE_FILE, 'r') as f:
            if '.env' in f.read():
                print("✓ .env 已在 .gitignore 中（安全）")
            else:
                print("⚠️ .env 未在 .gitignore 中（不安全！）")
                return False
    
    # 测试 API 连接
    print("\n测试 Gemini API 连接...")
    try:
        # 设置环境变量
        os.environ['GEMINI_API_KEY'] = api_key
        
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Reply with just 'OK' if you can read this."
        )
        
        if response and response.text:
            print(f"✓ API 连接成功: {response.text.strip()[:50]}")
            return True
        else:
            print("✗ API 响应为空")
            return False
            
    except Exception as e:
        print(f"✗ API 测试失败: {e}")
        return False


def main():
    """主函数."""
    # 设置 API Key
    api_key = setup_gemini_key()
    
    if api_key:
        # 验证配置
        success = verify_setup()
        
        print("\n" + "=" * 60)
        if success:
            print("✓ 配置完成！现在可以运行验证系统:")
            print()
            print("  # 运行 Gemini 验证器")
            print("  PYTHONPATH=. python3 h2q_project/h2q/agi/gemini_verifier.py")
            print()
            print("  # 运行监督学习系统（带验证）")
            print("  PYTHONPATH=h2q_project/h2q/agi python3 h2q_project/h2q/agi/supervised_learning_system.py")
        else:
            print("⚠️ 配置未完成，请检查上述错误")
        print("=" * 60)
    else:
        print("\n配置已取消。")


if __name__ == "__main__":
    main()
