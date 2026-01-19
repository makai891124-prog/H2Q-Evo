import sys
import os
import site

print("="*60)
print("🕵️‍♂️ DOCKER 环境深度诊断报告")
print("="*60)

# 1. 检查 sys.path (Python 搜索路径)
print(f"\n[1] Python 搜索路径 (sys.path):")
for i, p in enumerate(sys.path):
    print(f"  {i}: {p}")

# 2. 检查 site-packages 位置
print(f"\n[2] 系统库安装位置 (site-packages):")
for p in site.getsitepackages():
    print(f"  - {p}")

# 3. 影子杀手：检查当前目录下是否有捣乱的文件
print(f"\n[3] 影子文件检测 (Shadowing Check):")
cwd = os.getcwd()
print(f"  当前工作目录: {cwd}")
suspicious = ["transformers", "torch", "numpy", "fastapi"]
found_shadow = False
for name in suspicious:
    # 检查文件
    if os.path.exists(os.path.join(cwd, f"{name}.py")):
        print(f"  🚨 发现危险文件: {os.path.join(cwd, name + '.py')} <--- 罪魁祸首可能就是它！")
        found_shadow = True
    # 检查文件夹
    if os.path.exists(os.path.join(cwd, name)):
        # 检查是否是包
        if os.path.exists(os.path.join(cwd, name, "__init__.py")):
             print(f"  🚨 发现危险包目录: {os.path.join(cwd, name)} <--- 它覆盖了系统库！")
             found_shadow = True

if not found_shadow:
    print("  ✅ 未发现明显的影子文件。")

# 4. 终极导入测试
print(f"\n[4] 尝试导入 'transformers':")
try:
    import transformers
    print(f"  ✅ 成功导入！")
    print(f"  📂 文件位置: {transformers.__file__}")
    print(f"  🔢 版本: {transformers.__version__}")
except ImportError as e:
    print(f"  ❌ 导入失败: {e}")
    print("  -> 结论: 库确实没装上，或者路径配置完全错误。")
except Exception as e:
    print(f"  ❌ 发生异常: {e}")
    print(f"  📂 错误发生时的 __file__: {getattr(transformers, '__file__', 'Unknown') if 'transformers' in locals() else 'N/A'}")

print("="*60)