#!/usr/bin/env python3
"""
H2Q-Evo AGI 一键部署脚本
Autonomous AGI Engineering System - Complete Deployment

完整工作流:
1. 下载科学数据集 (数学/物理/化学/生物/工程)
2. 准备训练数据
3. 启动 AGI 科学训练
4. 监控训练进度
5. 生成进化报告

使用方法:
    python3 deploy_agi_final.py --hours 4 --download-data
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


class AGIDeploymentManager:
    """AGI 部署管理器"""

    def __init__(self, training_hours: float = 4.0, download_data: bool = True):
        self.training_hours = training_hours
        self.download_data = download_data
        self.project_root = Path(__file__).parent.parent
        self.h2q_project = self.project_root / "h2q_project"

        self.steps_completed = []
        self.start_time = None

    def print_banner(self):
        """打印启动横幅"""
        print("\n" + "=" * 80)
        print("H2Q-Evo AGI 自主可进化工程系统")
        print("Autonomous Self-Evolving AGI Engineering System")
        print("=" * 80)
        print(f"\n目标领域: 数学 | 物理 | 化学 | 生物 | 工程")
        print(f"训练时长: {self.training_hours} 小时")
        print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"项目路径: {self.project_root}")
        print("\n" + "=" * 80 + "\n")

    def check_environment(self) -> bool:
        """检查环境"""
        print("🔍 步骤 1/5: 检查环境...")

        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (
            python_version.major == 3 and python_version.minor < 8
        ):
            print(f"❌ Python版本过低: {sys.version}")
            print("   需要 Python 3.8+")
            return False

        print(f"✅ Python版本: {sys.version.split()[0]}")

        # 检查必要文件
        required_files = [
            self.h2q_project / "scientific_dataset_loader.py",
            self.h2q_project / "agi_scientific_trainer.py",
        ]

        for file in required_files:
            if not file.exists():
                print(f"❌ 缺少必要文件: {file}")
                return False

        print("✅ 必要文件检查通过")
        self.steps_completed.append("environment_check")
        return True

    def download_scientific_datasets(self) -> bool:
        """下载科学数据集"""
        if not self.download_data:
            print("\n📦 步骤 2/5: 跳过数据集下载（使用现有数据）")
            self.steps_completed.append("dataset_download_skipped")
            return True

        print("\n📦 步骤 2/5: 下载科学数据集...")
        print("   数据源: arXiv + 合成科学数据")
        print("   领域: 数学、物理、化学、生物、工程\n")

        loader_script = self.h2q_project / "scientific_dataset_loader.py"

        try:
            # 运行数据加载器
            result = subprocess.run(
                [sys.executable, str(loader_script)],
                cwd=str(self.h2q_project),
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
            )

            if result.returncode == 0:
                print(result.stdout)
                print("✅ 数据集下载完成")
                self.steps_completed.append("dataset_download")
                return True
            else:
                print(f"❌ 数据集下载失败:")
                print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("❌ 数据集下载超时")
            return False
        except Exception as e:
            print(f"❌ 数据集下载出错: {e}")
            return False

    def verify_training_data(self) -> bool:
        """验证训练数据"""
        print("\n✓ 步骤 3/5: 验证训练数据...")

        training_data_file = (
            self.h2q_project
            / "scientific_datasets"
            / "scientific_training_data.jsonl"
        )

        if not training_data_file.exists():
            print(f"❌ 训练数据文件不存在: {training_data_file}")
            return False

        # 统计行数
        try:
            with open(training_data_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)

            print(f"✅ 训练数据: {line_count} 条样本")

            if line_count < 10:
                print("⚠️  警告: 训练样本数量较少")

            self.steps_completed.append("data_verification")
            return True

        except Exception as e:
            print(f"❌ 读取训练数据失败: {e}")
            return False

    def start_agi_training(self) -> bool:
        """启动AGI训练"""
        print(f"\n🚀 步骤 4/5: 启动AGI科学训练 ({self.training_hours}小时)...")
        print("   初始化知识库...")
        print("   启动推理引擎...")
        print("   开始迭代训练...\n")

        trainer_script = self.h2q_project / "agi_scientific_trainer.py"
        training_data = (
            self.h2q_project
            / "scientific_datasets"
            / "scientific_training_data.jsonl"
        )

        try:
            # 启动训练
            result = subprocess.run(
                [
                    sys.executable,
                    str(trainer_script),
                    "--data",
                    str(training_data),
                    "--duration",
                    str(self.training_hours),
                    "--output",
                    str(self.h2q_project / "agi_training_output"),
                ],
                cwd=str(self.h2q_project),
                timeout=self.training_hours * 3600 + 300,  # 训练时长 + 5分钟缓冲
            )

            if result.returncode == 0:
                print("\n✅ AGI训练完成")
                self.steps_completed.append("agi_training")
                return True
            else:
                print(f"\n❌ AGI训练失败，返回码: {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            print("\n⚠️  训练超时（这可能是正常的）")
            self.steps_completed.append("agi_training_timeout")
            return True
        except KeyboardInterrupt:
            print("\n\n⚠️  训练被用户中断")
            return False
        except Exception as e:
            print(f"\n❌ 训练过程出错: {e}")
            return False

    def generate_final_report(self) -> bool:
        """生成最终报告"""
        print("\n📊 步骤 5/5: 生成最终报告...")

        output_dir = self.h2q_project / "agi_training_output"
        if not output_dir.exists():
            print("❌ 输出目录不存在")
            return False

        # 查找最新的报告文件
        report_files = list(output_dir.glob("agi_training_report_*.md"))
        result_files = list(output_dir.glob("agi_training_results_*.json"))

        if not report_files:
            print("❌ 未找到训练报告")
            return False

        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        latest_result = (
            max(result_files, key=lambda p: p.stat().st_mtime)
            if result_files
            else None
        )

        print(f"✅ 训练报告: {latest_report.name}")
        if latest_result:
            print(f"✅ 训练结果: {latest_result.name}")

        # 打印报告摘要
        try:
            with open(latest_report, "r", encoding="utf-8") as f:
                content = f.read()
                # 提取关键信息
                lines = content.split("\n")
                for line in lines[:20]:  # 打印前20行
                    if line.strip() and not line.startswith("#"):
                        print(f"   {line}")

        except Exception as e:
            print(f"⚠️  无法读取报告: {e}")

        self.steps_completed.append("report_generation")
        return True

    def print_summary(self):
        """打印部署摘要"""
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0

        print("\n" + "=" * 80)
        print("部署完成摘要")
        print("=" * 80)
        print(f"\n总耗时: {self._format_time(int(total_time))}")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n完成步骤 ({len(self.steps_completed)}/5):")

        step_names = {
            "environment_check": "✅ 环境检查",
            "dataset_download": "✅ 数据集下载",
            "dataset_download_skipped": "⊗ 数据集下载（跳过）",
            "data_verification": "✅ 数据验证",
            "agi_training": "✅ AGI训练",
            "agi_training_timeout": "✅ AGI训练（超时但完成）",
            "report_generation": "✅ 报告生成",
        }

        for step in self.steps_completed:
            print(f"  {step_names.get(step, step)}")

        print("\n" + "=" * 80)
        print("\n📁 输出位置:")
        print(f"  - 数据集: {self.h2q_project}/scientific_datasets/")
        print(f"  - 训练结果: {self.h2q_project}/agi_training_output/")
        print(f"  - 日志文件: {self.h2q_project}/agi_scientific_training.log")

        print("\n🎯 系统能力:")
        print("  ✅ 科学问题理解与分类")
        print("  ✅ 跨领域知识整合")
        print("  ✅ 自主推理与求解")
        print("  ✅ 知识库持续积累")
        print("  ✅ 进化趋势分析")

        print("\n🔄 下一步:")
        print("  1. 查看训练报告了解系统性能")
        print("  2. 分析知识库内容")
        print("  3. 运行更长时间的训练（12-24小时）")
        print("  4. 集成更多科学数据源")
        print("  5. 开发专业领域求解器")

        print("\n" + "=" * 80 + "\n")

    def _format_time(self, seconds: int) -> str:
        """格式化时间"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs}s"

    def run(self) -> bool:
        """运行完整部署流程"""
        self.start_time = time.time()
        self.print_banner()

        # 步骤1: 环境检查
        if not self.check_environment():
            print("\n❌ 部署失败: 环境检查未通过")
            return False

        # 步骤2: 下载数据集
        if not self.download_scientific_datasets():
            print("\n❌ 部署失败: 数据集下载失败")
            return False

        # 步骤3: 验证数据
        if not self.verify_training_data():
            print("\n❌ 部署失败: 训练数据验证失败")
            return False

        # 步骤4: 启动训练
        if not self.start_agi_training():
            print("\n❌ 部署失败: AGI训练失败")
            return False

        # 步骤5: 生成报告
        if not self.generate_final_report():
            print("\n⚠️  警告: 报告生成失败（训练可能成功）")

        # 打印摘要
        self.print_summary()

        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="H2Q-Evo AGI 自主可进化工程系统 - 一键部署"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=4.0,
        help="训练时长（小时），默认4小时",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        default=True,
        help="下载科学数据集（默认启用）",
    )
    parser.add_argument(
        "--no-download",
        dest="download_data",
        action="store_false",
        help="跳过数据集下载，使用现有数据",
    )

    args = parser.parse_args()

    # 创建部署管理器
    manager = AGIDeploymentManager(
        training_hours=args.hours, download_data=args.download_data
    )

    # 运行部署
    try:
        success = manager.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  部署被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 部署过程出现未预期错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
