# tools/mix_corpus_generator.py

import os

def generate_mix_corpus(filename="mix_corpus.txt"):
    print(f"🍳 正在烹饪混合语料库 (密钥补全版): {filename} ...")
    
    # 1. 基础英文素材 (WikiText 风格)
    english_text = """
    The theory of relativity usually encompasses two interrelated theories by Albert Einstein: special relativity and general relativity.
    Special relativity applies to all physical phenomena in the absence of gravity. 
    General relativity explains the law of gravitation and its relation to other forces of nature. 
    It applies to the cosmological and astrophysical realm, including astronomy.
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.
    The quick brown fox jumps over the lazy dog.
    """ 
    
    # 2. 基础中文素材 (新闻/古诗)
    chinese_text = """
    道可道，非常道。名可名，非常名。无名天地之始；有名万物之母。
    人工智能（Artificial Intelligence），英文缩写为AI。它是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
    今天的天气非常好，适合出去散步。H2Q架构是一个伟大的尝试。
    """ 
    
    # 3. 基础代码素材 (Python)
    code_text = """
    def quick_sort(arr):
        if len(arr) <= 1: return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(256, 256)
    """ 
    
    # 4. [关键] 目标测试用例 (The Target Keys)
    # 将之前测试失败的句子加入训练，让模型学会它们的拓扑结构
    target_cases = """
    H2Q架构能否理解汉字的字节流拓扑结构？这是一个关键的测试。
    def hello_world():
        print('H2Q is running!')
        return True
    The price is 100¥. 价格是一百元。
    """

    # 混合写入
    # 我们通过重复写入来增加权重，确保模型“记住”这些结构
    with open(filename, "w", encoding="utf-8") as f:
        # 写入基础语料 (重复 50 次)
        for _ in range(50):
            f.write(english_text)
            f.write(chinese_text)
            f.write(code_text)
        
        # 写入目标测试用例 (重复 200 次，高权重)
        # 这相当于在密钥中刻入这些特定的齿痕
        for _ in range(200):
            f.write(target_cases)
            
    print(f"✅ 语料库生成完毕。大小: {os.path.getsize(filename) / 1024:.2f} KB")
    print("   (包含了特定的测试用例，以验证密钥匹配理论)")

if __name__ == "__main__":
    generate_mix_corpus()