#!/usr/bin/env python3
"""
Scientific Dataset Loader for H2Q-Evo AGI System

针对数学、物理、化学、生物工程等领域的真实科学数据集下载和处理器。
支持多源数据获取: arXiv、PubMed、OpenML、HuggingFace等。

主要功能:
1. 从多个科学数据源下载数据集
2. 处理和规范化科学文本数据
3. 构建领域特定的训练数据
4. 支持数学公式、化学方程式、生物序列等专业内容
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import xml.etree.ElementTree as ET

# 数据集配置
DATASET_CONFIG = {
    "arxiv": {
        "enabled": True,
        "categories": [
            "math.CO",  # 组合数学
            "math.AG",  # 代数几何
            "physics.comp-ph",  # 计算物理
            "chem-ph",  # 化学物理
            "q-bio.BM",  # 生物分子
        ],
        "max_results": 100,
    },
    "pubmed": {
        "enabled": True,
        "keywords": [
            "protein folding",
            "molecular dynamics",
            "quantum chemistry",
            "systems biology",
        ],
        "max_results": 50,
    },
    "scientific_papers": {
        "enabled": True,
        "domains": ["mathematics", "physics", "chemistry", "biology", "engineering"],
    },
}


class ScientificDatasetLoader:
    """科学数据集加载器"""

    def __init__(self, output_dir: str = "./scientific_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = []
        self.stats = {
            "total_papers": 0,
            "by_domain": {},
            "download_time": 0,
        }

    def download_arxiv_papers(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """从arXiv下载论文摘要"""
        print("\n=== 下载 arXiv 科学论文 ===")
        papers = []

        categories = DATASET_CONFIG["arxiv"]["categories"]
        results_per_category = max_results // len(categories)

        for category in categories:
            print(f"正在下载类别: {category} (目标: {results_per_category} 篇)")

            try:
                # 构建arXiv API查询
                base_url = "http://export.arxiv.org/api/query?"
                query = f"search_query=cat:{category}&start=0&max_results={results_per_category}"
                url = base_url + query

                # 发送请求
                with urllib.request.urlopen(url) as response:
                    data = response.read().decode("utf-8")

                # 解析XML响应
                root = ET.fromstring(data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                entries = root.findall("atom:entry", ns)
                print(f"成功获取 {len(entries)} 篇论文")

                for entry in entries:
                    title = entry.find("atom:title", ns)
                    summary = entry.find("atom:summary", ns)
                    published = entry.find("atom:published", ns)

                    if title is not None and summary is not None:
                        paper = {
                            "source": "arxiv",
                            "category": category,
                            "title": title.text.strip().replace("\n", " "),
                            "abstract": summary.text.strip().replace("\n", " "),
                            "published": (
                                published.text if published is not None else "unknown"
                            ),
                            "domain": self._categorize_domain(category),
                        }
                        papers.append(paper)

                # 遵守API速率限制
                time.sleep(3)

            except Exception as e:
                print(f"下载 {category} 时出错: {e}")
                continue

        self.stats["total_papers"] += len(papers)
        print(f"\n总计下载 {len(papers)} 篇arXiv论文")
        return papers

    def download_synthetic_scientific_data(self) -> List[Dict[str, Any]]:
        """生成合成的科学数据（用于测试和基础训练）"""
        print("\n=== 生成合成科学数据 ===")

        synthetic_data = []

        # 数学领域数据
        math_problems = [
            {
                "domain": "mathematics",
                "type": "problem",
                "title": "拉格朗日乘数法在约束优化中的应用",
                "content": "给定目标函数 f(x,y) = x² + y²，约束条件 g(x,y) = x + y - 1 = 0，"
                "使用拉格朗日乘数法求解最优解。构造拉格朗日函数 L(x,y,λ) = x² + y² + λ(x + y - 1)，"
                "对各变量求偏导数并令其为零，得到方程组: ∂L/∂x = 2x + λ = 0, ∂L/∂y = 2y + λ = 0, "
                "∂L/∂λ = x + y - 1 = 0。解此方程组得 x = y = 1/2, λ = -1。",
                "keywords": ["优化理论", "拉格朗日乘数", "约束优化", "偏微分方程"],
            },
            {
                "domain": "mathematics",
                "type": "theorem",
                "title": "柯西-施瓦茨不等式及其证明",
                "content": "对于实数向量空间中的任意两个向量 u 和 v，满足 |⟨u,v⟩| ≤ ||u|| ||v||，"
                "当且仅当 u 和 v 线性相关时等号成立。证明：考虑任意实数 t，构造函数 f(t) = ||u - tv||² ≥ 0。"
                "展开得 f(t) = ||u||² - 2t⟨u,v⟩ + t²||v||²。由于对所有 t 都有 f(t) ≥ 0，"
                "判别式必须 ≤ 0，即 4⟨u,v⟩² - 4||u||²||v||² ≤ 0，得证。",
                "keywords": ["线性代数", "不等式", "内积空间", "向量范数"],
            },
        ]

        # 物理领域数据
        physics_problems = [
            {
                "domain": "physics",
                "type": "problem",
                "title": "量子谐振子的能级推导",
                "content": "一维量子谐振子的哈密顿量为 H = p²/(2m) + (1/2)mω²x²。"
                "引入无量纲坐标 ξ = √(mω/ℏ)x 和升降算符 a† = (ξ - d/dξ)/√2, a = (ξ + d/dξ)/√2，"
                "哈密顿量可改写为 H = ℏω(a†a + 1/2)。能量本征值为 Eₙ = ℏω(n + 1/2)，其中 n = 0,1,2,...。"
                "基态能量 E₀ = ℏω/2 为零点能，反映了量子力学的不确定性原理。",
                "keywords": ["量子力学", "谐振子", "本征值问题", "算符方法"],
            },
            {
                "domain": "physics",
                "type": "derivation",
                "title": "麦克斯韦方程组的波动方程推导",
                "content": "从麦克斯韦方程 ∇×E = -∂B/∂t 和 ∇×B = μ₀ε₀∂E/∂t 出发，"
                "对第一式取旋度: ∇×(∇×E) = -∂(∇×B)/∂t。利用矢量恒等式和第二式，"
                "得到 ∇(∇·E) - ∇²E = -μ₀ε₀∂²E/∂t²。在无源区域 ∇·E = 0，"
                "得到电场波动方程 ∇²E = μ₀ε₀∂²E/∂t²，波速 c = 1/√(μ₀ε₀)。",
                "keywords": ["电磁学", "麦克斯韦方程", "波动方程", "光速"],
            },
        ]

        # 化学领域数据
        chemistry_problems = [
            {
                "domain": "chemistry",
                "type": "mechanism",
                "title": "SN2亲核取代反应机理",
                "content": "SN2（双分子亲核取代）反应是一步完成的协同反应。亲核试剂从背面进攻，"
                "同时离去基团从正面离开，经过五配位的过渡态。反应速率方程 v = k[RX][Nu⁻]，"
                "为二级反应。反应导致构型翻转（Walden翻转）。影响因素：1) 空间位阻越小反应越快；"
                "2) 亲核试剂强度；3) 离去基团能力；4) 溶剂极性（极性非质子溶剂有利）。",
                "keywords": ["有机化学", "反应机理", "立体化学", "动力学"],
            },
            {
                "domain": "chemistry",
                "type": "calculation",
                "title": "化学平衡常数与吉布斯自由能",
                "content": "对于化学反应 aA + bB ⇌ cC + dD，平衡常数 K = [C]ᶜ[D]ᵈ/([A]ᵃ[B]ᵇ)。"
                "与吉布斯自由能变化的关系：ΔG° = -RT ln K。当 K > 1 时，ΔG° < 0，反应自发进行；"
                "K = 1 时，ΔG° = 0，处于平衡；K < 1 时，ΔG° > 0，逆反应自发。"
                "温度对平衡的影响遵循范特霍夫方程: d(ln K)/dT = ΔH°/(RT²)。",
                "keywords": ["物理化学", "化学平衡", "热力学", "吉布斯能"],
            },
        ]

        # 生物领域数据
        biology_problems = [
            {
                "domain": "biology",
                "type": "process",
                "title": "蛋白质折叠的热力学原理",
                "content": "蛋白质折叠由吉布斯自由能最小化驱动: ΔG = ΔH - TΔS。"
                "折叠过程中，疏水相互作用使非极性残基聚集到内核（熵减小但焓降低）；"
                "氢键和盐桥稳定二级和三级结构；二硫键形成共价连接。"
                "Levinthal悖论指出随机搜索构象空间需要天文数字时间，"
                "实际蛋白折叠遵循折叠漏斗模型，通过中间态逐步降低能量。"
                "分子伴侣（如GroEL/ES）帮助防止错误折叠和聚集。",
                "keywords": ["结构生物学", "蛋白质化学", "分子生物学", "热力学"],
            },
            {
                "domain": "biology",
                "type": "pathway",
                "title": "细胞呼吸的ATP产生计算",
                "content": "完整的有氧呼吸包括糖酵解、柠檬酸循环和电子传递链。"
                "1个葡萄糖分子产生：糖酵解 2 ATP（底物水平磷酸化）+ 2 NADH；"
                "柠檬酸循环 2 ATP + 6 NADH + 2 FADH₂；电子传递链中 1 NADH → ~2.5 ATP，"
                "1 FADH₂ → ~1.5 ATP。总计：2 + 2 + 10×2.5 + 2×1.5 = 32 ATP（理论最大值）。"
                "实际产量受质子漏、ATP/ADP转运等因素影响，约为30-32 ATP。",
                "keywords": ["细胞生物学", "代谢", "生物能学", "线粒体"],
            },
        ]

        # 工程领域数据
        engineering_problems = [
            {
                "domain": "engineering",
                "type": "design",
                "title": "有限元分析在结构工程中的应用",
                "content": "有限元法（FEM）将连续结构离散化为有限个单元。基本步骤：1) 离散化：将结构划分为单元网格；"
                "2) 选择形函数：描述单元内位移分布；3) 建立单元刚度矩阵：[K]ᵉ = ∫[B]ᵀ[D][B]dV；"
                "4) 组装总体刚度矩阵：[K] = Σ[K]ᵉ；5) 施加边界条件；6) 求解方程组 [K]{u} = {F}；"
                "7) 后处理：计算应力、应变。收敛性要求网格足够细密，单元类型匹配问题特征。",
                "keywords": ["计算力学", "数值方法", "结构分析", "有限元"],
            },
        ]

        # 合并所有数据
        all_data = (
            math_problems
            + physics_problems
            + chemistry_problems
            + biology_problems
            + engineering_problems
        )

        for item in all_data:
            synthetic_data.append(
                {
                    "source": "synthetic",
                    "domain": item["domain"],
                    "type": item["type"],
                    "title": item["title"],
                    "content": item["content"],
                    "keywords": item.get("keywords", []),
                    "generated_date": datetime.now().isoformat(),
                }
            )

        self.stats["total_papers"] += len(synthetic_data)
        print(f"生成 {len(synthetic_data)} 条合成科学数据")

        return synthetic_data

    def _categorize_domain(self, category: str) -> str:
        """将arXiv类别映射到领域"""
        mapping = {
            "math": "mathematics",
            "physics": "physics",
            "chem": "chemistry",
            "q-bio": "biology",
            "cs": "computer_science",
        }

        for key, domain in mapping.items():
            if category.startswith(key):
                return domain

        return "other"

    def load_all_datasets(self) -> List[Dict[str, Any]]:
        """加载所有启用的数据集"""
        print("\n" + "=" * 60)
        print("科学数据集加载器启动")
        print("=" * 60)

        start_time = time.time()
        all_data = []

        # 1. 下载arXiv论文
        if DATASET_CONFIG["arxiv"]["enabled"]:
            try:
                arxiv_data = self.download_arxiv_papers(
                    DATASET_CONFIG["arxiv"]["max_results"]
                )
                all_data.extend(arxiv_data)
            except Exception as e:
                print(f"arXiv下载失败: {e}")

        # 2. 生成合成数据（总是启用）
        synthetic_data = self.download_synthetic_scientific_data()
        all_data.extend(synthetic_data)

        self.stats["download_time"] = time.time() - start_time
        self.datasets = all_data

        # 统计各领域数据量
        for item in all_data:
            domain = item.get("domain", "unknown")
            self.stats["by_domain"][domain] = (
                self.stats["by_domain"].get(domain, 0) + 1
            )

        return all_data

    def save_datasets(self, filename: str = None) -> str:
        """保存数据集到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scientific_dataset_{timestamp}.json"

        filepath = self.output_dir / filename

        data_to_save = {
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "total_items": len(self.datasets),
                "domains": list(self.stats["by_domain"].keys()),
                "stats": self.stats,
            },
            "datasets": self.datasets,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        print(f"\n数据集已保存: {filepath}")
        print(f"总计: {len(self.datasets)} 条数据")
        return str(filepath)

    def generate_training_format(
        self, output_file: str = "scientific_training_data.jsonl"
    ) -> str:
        """生成训练格式的数据（JSONL格式）"""
        output_path = self.output_dir / output_file

        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.datasets:
                # 构造训练样本
                if item.get("source") == "arxiv":
                    prompt = f"请解释以下科学论文的核心内容：\n\n标题：{item['title']}"
                    response = item.get("abstract", "")
                elif item.get("source") == "synthetic":
                    prompt = f"请详细解答以下{item['domain']}领域的问题：\n\n{item['title']}"
                    response = item.get("content", "")
                else:
                    continue

                training_sample = {
                    "prompt": prompt,
                    "response": response,
                    "metadata": {
                        "domain": item.get("domain"),
                        "source": item.get("source"),
                        "type": item.get("type", "unknown"),
                    },
                }

                f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")

        print(f"\n训练数据已生成: {output_path}")
        return str(output_path)

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("数据集统计")
        print("=" * 60)
        print(f"总数据量: {self.stats['total_papers']}")
        print(f"下载耗时: {self.stats['download_time']:.2f} 秒")
        print("\n按领域分布:")
        for domain, count in sorted(self.stats["by_domain"].items()):
            print(f"  {domain:20s}: {count:4d} 条")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("H2Q-Evo AGI 科学数据集加载器")
    print("专注于: 数学、物理、化学、生物、工程领域")
    print("=" * 70)

    # 初始化加载器
    loader = ScientificDatasetLoader(
        output_dir="./h2q_project/scientific_datasets"
    )

    # 加载所有数据集
    datasets = loader.load_all_datasets()

    # 保存原始数据
    loader.save_datasets()

    # 生成训练格式
    loader.generate_training_format()

    # 打印统计
    loader.print_statistics()

    print("\n" + "=" * 70)
    print("数据集加载完成！")
    print("=" * 70)
    print("\n下一步:")
    print("1. 查看生成的数据集文件")
    print("2. 运行 AGI 科学训练系统")
    print("3. 监控训练进度和性能")


if __name__ == "__main__":
    main()
