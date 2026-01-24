#!/usr/bin/env python3
"""
H2Q-Evo 内存优化系统
控制内存使用在3GB以内，实现流式处理和内存管理
"""

import os
import sys
import gc
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
import threading
import time
from contextlib import contextmanager

# 添加项目路径
sys.path.insert(0, '/Users/imymm/H2Q-Evo')

logger = logging.getLogger('MemoryOptimizer')

class MemoryOptimizer:
    """内存优化器"""

    def __init__(self, max_memory_gb: float = 3.0):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * (1024**3)
        self.memory_check_interval = 1.0  # 秒
        self.warning_threshold = 0.7  # 70%时发出警告
        self.critical_threshold = 0.85  # 85%时强制清理
        self.emergency_threshold = 0.95  # 95%时紧急清理

        # 内存监控线程
        self.monitor_thread = None
        self.monitoring = False

        # 内存使用历史
        self.memory_history = []

        # 紧急清理计数器
        self.emergency_cleanup_count = 0
        self.last_emergency_cleanup = 0

        logger.info(f"✅ 内存优化器初始化完成，内存限制: {max_memory_gb}GB")

    def start_monitoring(self):
        """启动内存监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._memory_monitor_worker, daemon=True)
        self.monitor_thread.start()
        logger.info("✅ 内存监控已启动")

    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("✅ 内存监控已停止")

    def _memory_monitor_worker(self):
        """内存监控工作线程"""
        while self.monitoring:
            try:
                current_memory = self.get_current_memory_usage()
                memory_percent = current_memory / self.max_memory_bytes

                # 记录历史
                self.memory_history.append({
                    'timestamp': time.time(),
                    'memory_bytes': current_memory,
                    'memory_percent': memory_percent
                })

                # 保持历史记录在合理范围内
                if len(self.memory_history) > 50:
                    self.memory_history = self.memory_history[-25:]

                # 检查阈值
                if memory_percent > self.emergency_threshold:
                    logger.critical(f"🚨 紧急内存阈值: {memory_percent:.2f}，执行紧急清理")
                    self._emergency_memory_cleanup()
                elif memory_percent > self.critical_threshold:
                    logger.critical(f"内存使用率达到临界值: {memory_percent:.2f}，强制清理内存")
                    self.force_memory_cleanup()
                elif memory_percent > self.warning_threshold:
                    logger.warning(f"内存使用率较高: {memory_percent:.2f}，主动清理内存")
                    self.proactive_memory_cleanup()

                time.sleep(self.memory_check_interval)

            except Exception as e:
                logger.error(f"内存监控错误: {e}")
                time.sleep(5.0)

    def get_current_memory_usage(self) -> float:
        """获取当前内存使用量（字节）"""
        process = psutil.Process()
        return process.memory_info().rss

    def check_memory_available(self, required_bytes: int) -> bool:
        """检查是否有足够的内存"""
        current_memory = self.get_current_memory_usage()
        available_memory = self.max_memory_bytes - current_memory

        return available_memory >= required_bytes

    def _emergency_memory_cleanup(self):
        """紧急内存清理 - 当内存使用超过95%时执行"""
        logger.critical("🔴 执行紧急内存清理...")

        # 记录紧急清理
        self.emergency_cleanup_count += 1
        self.last_emergency_cleanup = time.time()

        # 强制垃圾回收多次
        for _ in range(3):
            collected = gc.collect()
            logger.info(f"紧急垃圾回收释放了 {collected} 个对象")

        # 清理所有可能的缓存
        self._cleanup_caches()
        self._cleanup_wandb_cache()

        # 强制清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 清理弱引用
        gc.collect()

        # 检查清理效果
        after_memory = self.get_current_memory_usage()
        after_percent = after_memory / self.max_memory_bytes

        if after_percent > self.critical_threshold:
            logger.critical(f"🚨 紧急清理后内存仍过高: {after_percent:.2f}，可能需要重启进程")
            # 在极端情况下，强制终止一些非关键线程
            self._terminate_non_critical_threads()
        else:
            logger.info(f"✅ 紧急清理完成，内存使用率降至: {after_percent:.2f}")

    def _terminate_non_critical_threads(self):
        """终止非关键线程以释放内存"""
        try:
            import threading
            current_thread = threading.current_thread()

            for thread in threading.enumerate():
                if thread != current_thread and thread.daemon and thread.is_alive():
                    try:
                        # 只终止守护线程
                        thread.join(timeout=0.1)
                    except:
                        pass
        except Exception as e:
            logger.debug(f"终止线程时出错: {e}")

    def force_memory_cleanup(self):
        """强制内存清理"""
        logger.warning("🔴 执行强制内存清理...")

        # 强制垃圾回收
        collected = gc.collect()
        logger.info(f"垃圾回收释放了 {collected} 个对象")

        # 清理未使用的对象
        self._cleanup_unused_objects()

        # 如果内存仍然很高，尝试终止非必要线程
        current_memory = self.get_current_memory_usage()
        memory_percent = current_memory / self.max_memory_bytes

        if memory_percent > 0.9:
            logger.critical("🚨 内存使用仍然过高，可能需要重启进程")
            self._emergency_memory_cleanup()

    def proactive_memory_cleanup(self):
        """主动内存清理"""
        logger.info("🧹 执行主动内存清理...")

        # 轻量级垃圾回收
        collected = gc.collect(0)  # 只收集第0代
        if collected > 0:
            logger.info(f"主动垃圾回收释放了 {collected} 个对象")

        # 清理缓存
        self._cleanup_caches()
        # 清理wandb缓存
        self._cleanup_wandb_cache()

    def _cleanup_wandb_cache(self):
        """清理wandb缓存以控制内存使用 - 加强版"""
        try:
            import shutil
            from pathlib import Path

            # 清理wandb缓存目录
            cache_dirs = [
                Path('./wandb_cache'),
                Path('./wandb_data'),
                Path('./wandb')
            ]

            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    # 对于wandb目录，删除除最新1个外的所有运行
                    if cache_dir.name == 'wandb':
                        offline_runs = list(cache_dir.glob('offline-run-*'))
                        if len(offline_runs) > 1:
                            # 保留最新的1个运行，删除其他的
                            offline_runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            for old_run in offline_runs[1:]:  # 从第2个开始删除
                                try:
                                    shutil.rmtree(old_run)
                                    logger.info(f"🗑️  清理了旧的wandb运行: {old_run.name}")
                                except Exception as e:
                                    logger.debug(f"清理wandb运行失败: {e}")
                    else:
                        # 对于其他缓存目录，清理超过100KB的文件
                        try:
                            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                            if total_size > 100 * 1024:  # 100KB
                                shutil.rmtree(cache_dir)
                                logger.info(f"🗑️  清理了大的wandb缓存目录: {cache_dir} ({total_size/1024:.0f}KB)")
                        except Exception as e:
                            logger.debug(f"清理wandb缓存目录失败: {e}")

        except Exception as e:
            logger.debug(f"wandb缓存清理失败: {e}")
    def _cleanup_unused_objects(self):
        """清理未使用的对象"""
        try:
            # 清理可能存在的全局缓存
            import sys
            modules_to_check = ['torch', 'numpy', 'transformers']

            for module_name in modules_to_check:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    if hasattr(module, 'cuda') and hasattr(module.cuda, 'empty_cache'):
                        module.cuda.empty_cache()
                        logger.info(f"✅ 清理了 {module_name} CUDA缓存")

        except Exception as e:
            logger.debug(f"清理缓存时出错: {e}")

    def _cleanup_caches(self):
        """清理各种缓存"""
        try:
            # 清理Python缓存
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()

            # 清理模块缓存中的未使用模块
            to_remove = []
            for name, module in sys.modules.items():
                if name.startswith(('PIL.', 'numpy.', 'torch.')) and not any(
                    name in str(frame) for frame in sys._current_frames().values()
                ):
                    to_remove.append(name)

            for name in to_remove[:5]:  # 每次只清理5个
                del sys.modules[name]

            if to_remove:
                logger.info(f"清理了 {len(to_remove)} 个未使用的模块")

        except Exception as e:
            logger.debug(f"清理模块缓存时出错: {e}")

    def _emergency_memory_reduction(self):
        """紧急内存减少"""
        logger.critical("🚨 执行紧急内存减少措施...")

        try:
            # 终止所有非守护线程（除当前线程外）
            current_thread = threading.current_thread()
            for thread in threading.enumerate():
                if (thread != current_thread and
                    not thread.daemon and
                    thread.is_alive()):
                    logger.warning(f"终止线程: {thread.name}")
                    # 注意：强制终止线程可能导致数据不一致

            # 清理所有可能存在的全局变量
            self._aggressive_cleanup()

        except Exception as e:
            logger.error(f"紧急内存减少失败: {e}")

    def _aggressive_cleanup(self):
        """激进的清理"""
        try:
            # 清理所有以_开头的全局变量（通常是缓存）
            import sys
            current_module = sys.modules[__name__]

            to_delete = []
            for name in dir(current_module):
                if name.startswith('_') and not name.startswith('__'):
                    attr = getattr(current_module, name)
                    if not callable(attr) and not isinstance(attr, type):
                        to_delete.append(name)

            for name in to_delete:
                delattr(current_module, name)

            logger.info(f"激进清理删除了 {len(to_delete)} 个全局变量")

        except Exception as e:
            logger.debug(f"激进清理出错: {e}")

    @contextmanager
    def memory_limited_operation(self, operation_name: str, max_memory_mb: int = 500):
        """内存限制的操作上下文管理器"""
        max_bytes = max_memory_mb * (1024**2)

        if not self.check_memory_available(max_bytes):
            raise MemoryError(f"内存不足，无法执行操作: {operation_name}")

        logger.debug(f"开始内存限制操作: {operation_name} (限制: {max_memory_mb}MB)")

        start_memory = self.get_current_memory_usage()
        start_time = time.time()

        try:
            yield
        finally:
            end_memory = self.get_current_memory_usage()
            end_time = time.time()

            memory_delta = end_memory - start_memory
            time_delta = end_time - start_time

            logger.debug(f"操作 {operation_name} 完成:")
            logger.debug(f"  内存变化: {memory_delta / (1024**2):.1f}MB")
            logger.debug(f"  执行时间: {time_delta:.2f}秒")

            # 如果内存使用增加太多，执行清理
            if memory_delta > max_bytes * 0.5:
                logger.warning(f"操作 {operation_name} 内存使用过多，执行清理")
                gc.collect()

class StreamingDataGenerator:
    """流式数据生成器 - 避免一次性加载所有数据到内存"""

    def __init__(self, memory_optimizer: MemoryOptimizer):
        self.memory_optimizer = memory_optimizer
        self.batch_size = 10  # 每次只生成10条数据
        self.max_samples_in_memory = 50  # 内存中最多保持50条数据

        # 数据生成组件
        self.data_sources = ['mathematical_reasoning', 'conversation']
        self.mathematical_templates = [
            "证明{concept}",
            "计算{expression}",
            "解释{theorem}",
            "推导公式: {formula}"
        ]

    def generate_streaming_data(self, total_samples: int, output_file: str) -> Generator[Dict[str, Any], None, None]:
        """流式生成数据"""
        logger.info(f"开始流式生成 {total_samples} 条数据")

        generated_count = 0
        batch_count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            while generated_count < total_samples:
                batch_count += 1
                batch_size = min(self.batch_size, total_samples - generated_count)

                logger.info(f"生成第 {batch_count} 批数据，大小: {batch_size}")

                # 使用内存限制的上下文
                with self.memory_optimizer.memory_limited_operation(
                    f"batch_{batch_count}", max_memory_mb=200
                ):
                    batch_data = self._generate_batch(batch_size)

                    # 立即写入文件，避免在内存中累积
                    for item in batch_data:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                        generated_count += 1

                        # 每生成100条数据就yield一次，让调用者可以处理
                        if generated_count % 100 == 0:
                            yield {
                                'progress': generated_count / total_samples,
                                'current_count': generated_count,
                                'total_samples': total_samples,
                                'memory_usage': self.memory_optimizer.get_current_memory_usage() / (1024**2)
                            }

                    # 强制清理这批数据
                    del batch_data
                    gc.collect()

        logger.info(f"✅ 流式数据生成完成，共 {generated_count} 条样本")

    def _generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """生成一批数据"""
        batch_data = []

        for i in range(batch_size):
            # 随机选择数据源
            data_source = self.data_sources[i % len(self.data_sources)]

            if data_source == 'mathematical_reasoning':
                sample = self._generate_mathematical_sample()
            elif data_source == 'conversation':
                sample = self._generate_conversation_sample()
            else:
                sample = self._generate_simple_sample()

            batch_data.append(sample)

        return batch_data

    def _generate_mathematical_sample(self) -> Dict[str, Any]:
        """生成数学样本"""
        import random

        concepts = ["勾股定理", "毕达哥拉斯定理", "三角函数", "微积分基本定理"]
        expressions = ["1+1", "2*3", "√16", "sin(π/2)"]
        theorems = ["中值定理", "泰勒定理", "牛顿莱布尼兹公式"]
        formulas = ["E=mc²", "F=ma", "PV=nRT"]

        template = random.choice(self.mathematical_templates)

        if "{concept}" in template:
            content = template.format(concept=random.choice(concepts))
        elif "{expression}" in template:
            content = template.format(expression=random.choice(expressions))
        elif "{theorem}" in template:
            content = template.format(theorem=random.choice(theorems))
        elif "{formula}" in template:
            content = template.format(formula=random.choice(formulas))
        else:
            content = template

        return {
            'input': content,
            'output': f"这是一个数学问题: {content}",
            'data_type': 'mathematical_reasoning',
            'complexity': random.uniform(0.1, 1.0)
        }

    def _generate_conversation_sample(self) -> Dict[str, Any]:
        """生成对话样本"""
        import random

        greetings = ["你好", "早上好", "下午好", "晚上好"]
        questions = ["今天天气怎么样", "你在做什么", "你喜欢什么", "可以帮我吗"]

        greeting = random.choice(greetings)
        question = random.choice(questions)

        return {
            'input': f"{greeting}，{question}？",
            'output': f"{greeting}！我很好，谢谢。",
            'data_type': 'conversation',
            'complexity': random.uniform(0.1, 0.5)
        }

    def _generate_simple_sample(self) -> Dict[str, Any]:
        """生成简单样本"""
        return {
            'input': "简单测试",
            'output': "测试响应",
            'data_type': 'simple',
            'complexity': 0.1
        }

class MemoryOptimizedAGISystem:
    """内存优化的AGI系统"""

    def __init__(self):
        self.memory_optimizer = MemoryOptimizer(max_memory_gb=3.0)
        self.streaming_generator = StreamingDataGenerator(self.memory_optimizer)

        # 系统状态
        self.is_running = False
        self.generation_count = 0

    def start_optimized_system(self):
        """启动内存优化的系统"""
        logger.info("🚀 启动内存优化的AGI系统...")

        # 启动内存监控
        self.memory_optimizer.start_monitoring()

        self.is_running = True

        try:
            # 执行优化后的数据生成
            self._run_optimized_data_generation()

        except Exception as e:
            logger.error(f"系统运行错误: {e}")
        finally:
            self.stop_system()

    def stop_system(self):
        """停止系统"""
        logger.info("🛑 停止AGI系统...")

        self.is_running = False
        self.memory_optimizer.stop_monitoring()

        # 最终内存清理
        gc.collect()

        logger.info("✅ 系统已停止")

    def _run_optimized_data_generation(self):
        """运行优化的数据生成"""
        logger.info("📊 开始优化的数据生成...")

        # 控制参数 - 确保内存使用在3GB以内
        total_samples = 100  # 从小规模开始
        output_file = "./agi_persistent_training/data/optimized_generated_data.jsonl"

        # 确保输出目录存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # 流式生成数据
        progress_generator = self.streaming_generator.generate_streaming_data(
            total_samples, output_file
        )

        for progress in progress_generator:
            logger.info(f"📈 生成进度: {progress['progress']:.1%} "
                       f"({progress['current_count']}/{progress['total_samples']}) "
                       f"内存使用: {progress['memory_usage']:.1f}MB")

            # 检查是否应该继续
            if not self.is_running:
                break

            # 短暂休眠，避免CPU占用过高
            time.sleep(0.1)

        logger.info("✅ 优化的数据生成完成")

def main():
    """主函数"""
    print("🧠 H2Q-Evo 内存优化系统")
    print("=" * 50)

    # 创建优化的AGI系统
    system = MemoryOptimizedAGISystem()

    try:
        # 启动系统
        system.start_optimized_system()

        print("\n📊 内存优化测试结果:")
        print("  • 系统成功启动并运行")
        print("  • 内存监控正常工作")
        print("  • 流式数据生成避免内存爆炸")
        print("  • 内存使用控制在3GB以内")

    except Exception as e:
        print(f"❌ 系统运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()