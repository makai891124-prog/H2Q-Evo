import os
import sys
from pathlib import Path
from openai import OpenAI

# --- 配置部分 ---
# 使用环境变量获取 API Key
# 支持多个 LLM 提供商：DeepSeek, OpenAI, Claude 等

def get_api_config():
    """从环境变量获取 API 配置，提示用户输入"""
    
    # 尝试从环境变量获取
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    
    # 如果没有配置，提示用户
    if not api_key:
        print("=" * 60)
        print("ERROR: LLM API Key not found!")
        print("=" * 60)
        print("\n请设置以下环境变量之一：")
        print("\n方式 1：DeepSeek API")
        print("  export LLM_API_KEY='your-deepseek-api-key'")
        print("  export LLM_BASE_URL='https://api.deepseek.com/v1'")
        print("  export LLM_MODEL='deepseek-chat'")
        print("\n方式 2：OpenAI API")
        print("  export LLM_API_KEY='your-openai-api-key'")
        print("  export LLM_BASE_URL='https://api.openai.com/v1'")
        print("  export LLM_MODEL='gpt-3.5-turbo'")
        print("\n方式 3：其他 OpenAI 兼容的提供商")
        print("  export LLM_API_KEY='your-api-key'")
        print("  export LLM_BASE_URL='your-base-url'")
        print("  export LLM_MODEL='your-model'")
        print("\n获取免费 API Key：")
        print("  • DeepSeek: https://platform.deepseek.com/")
        print("  • OpenAI: https://platform.openai.com/")
        print("=" * 60)
        raise ValueError("LLM_API_KEY not set. Please set environment variables first.")
    
    return api_key, base_url, model

# 获取配置并初始化客户端
API_KEY, BASE_URL, MODEL = get_api_config()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class ProjectAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).resolve()
        self.ignore_dirs = {'.git', '__pycache__', 'venv', '.venv', 'node_modules'}

    def get_all_py_files(self):
        """递归获取目录下所有 .py 文件"""
        py_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # 过滤掉不需要扫描的目录
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            for file in files:
                if file.endswith('.py'):
                    py_files.append(Path(root) / file)
        return py_files

    def read_codes(self):
        """读取所有代码并格式化"""
        combined_content = []
        files = self.get_all_py_files()
        
        if not files:
            return None

        for file_path in files:
            relative_path = file_path.relative_to(self.root_dir)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    combined_content.append(f"--- File: {relative_path} ---\n{code}\n")
            except Exception as e:
                combined_content.append(f"--- File: {relative_path} ---\n[读取失败: {e}]\n")
        
        return "\n".join(combined_content)

    def analyze(self):
        """调用大模型进行分析"""
        print(f"正在读取目录: {self.root_dir} ...")
        all_code = self.read_codes()

        if not all_code:
            print("未发现任何 .py 文件。")
            return

        # 检查代码量是否过大（简单估算）
        if len(all_code) > 100000: # 约 100k 字符
            print("警告：代码量较大，可能会超过模型上下文限制。正在尝试发送...")

        prompt = f"""
你是一个资深的 Python 架构师。请分析以下项目的代码，并完成以下任务：
1. **项目结构梳理**：简述该项目的模块划分和目录结构。
2. **核心逻辑分析**：解释项目的主要功能实现流程、关键类和函数的作用。
3. **代码质量评价**：从可读性、健壮性、设计模式使用、潜在 Bug 等角度进行评价。
4. **改进建议**：给出具体的优化建议。

以下是项目代码内容：
{all_code}
"""

        print("正在请求大模型分析中，请稍候...")
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的代码审计和架构分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API 请求失败: {e}"

def main():
    # 可以通过命令行参数传入路径，否则默认当前目录
    target_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    if API_KEY == "你的API_KEY":
        print("错误：请先在脚本中配置你的 API_KEY")
        return

    analyzer = ProjectAnalyzer(target_path)
    result = analyzer.analyze()
    
    if result:
        print("\n" + "="*50)
        print("项目分析报告")
        print("="*50 + "\n")
        print(result)
        
        # 将结果保存到本地文件
        output_file = "project_analysis_report.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n报告已保存至: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()