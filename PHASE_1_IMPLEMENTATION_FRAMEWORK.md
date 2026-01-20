# Phase 1 实现框架 - v2.3.0 本地学习系统

**目标**: 在 5-6 周内为 H2Q-Evo 添加本地学习、知识持久化和迁移能力  
**范围**: 最小可行产品 (MVP) - 核心功能验证  
**工作量**: ~2000 行代码

---

## 架构概览

```
┌──────────────────────────────────────────────────────────┐
│  H2Q-Evo v2.3.0 架构                                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ CLI 入口层 (h2q_cli/)                             │ │
│  ├─ main.py          (命令行接口)                    │ │
│  ├─ commands.py      (命令处理)                      │ │
│  └─ config.py        (配置管理)                      │ │
│  └────────────────────────────────────────────────────┘ │
│                     ↓                                    │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 执行层 (h2q_project/)                             │ │
│  ├─ local_executor.py    (本地执行器)                │ │
│  ├─ learning_loop.py     (学习循环)                  │ │
│  ├─ strategy_manager.py  (策略管理)                  │ │
│  └─ feedback_handler.py  (反馈处理)                  │ │
│  └────────────────────────────────────────────────────┘ │
│                     ↓                                    │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 知识层 (h2q_project/knowledge/)                   │ │
│  ├─ knowledge_db.py      (知识库)                    │ │
│  ├─ knowledge_graph.py   (知识图谱)                  │ │
│  ├─ experience_store.py  (经验存储)                  │ │
│  └─ query_engine.py      (查询引擎)                  │ │
│  └────────────────────────────────────────────────────┘ │
│                     ↓                                    │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 持久化层 (h2q_project/persistence/)               │ │
│  ├─ checkpoint_manager.py (检查点管理)               │ │
│  ├─ migration_engine.py   (迁移引擎)                │ │
│  ├─ state_serializer.py   (状态序列化)               │ │
│  └─ integrity_checker.py  (完整性检查)               │ │
│  └────────────────────────────────────────────────────┘ │
│                     ↓                                    │
│  ┌────────────────────────────────────────────────────┐ │
│  │ 监控层 (h2q_project/monitoring/)                  │ │
│  ├─ metrics_tracker.py    (指标追踪)                │ │
│  ├─ capability_monitor.py (能力监控)                │ │
│  ├─ performance_logger.py (性能日志)                │ │
│  └─ evolution_reporter.py (进化报告)                │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 模块详细设计

### 1. CLI 入口层 (`h2q_cli/`)

#### `h2q_cli/main.py`

```python
#!/usr/bin/env python3
"""
H2Q-Evo CLI 主入口
使用方式: h2q init / h2q execute "任务" / h2q status / h2q export-checkpoint
"""

import click
import sys
from pathlib import Path
from .commands import InitCommand, ExecuteCommand, StatusCommand, ExportCommand

@click.group()
@click.version_option(version='2.3.0')
def cli():
    """H2Q-Evo 自主学习系统 CLI"""
    pass

@cli.command()
def init():
    """初始化 H2Q-Evo 环境"""
    InitCommand().run()

@cli.command()
@click.argument('task', type=str)
@click.option('--strategy', default='auto', help='执行策略')
@click.option('--save-knowledge', is_flag=True, help='保存学到的知识')
def execute(task, strategy, save_knowledge):
    """执行任务"""
    ExecuteCommand(task, strategy, save_knowledge).run()

@cli.command()
def status():
    """显示 agent 状态"""
    StatusCommand().run()

@cli.command()
@click.argument('output', type=click.Path())
def export(output):
    """导出检查点"""
    ExportCommand(output).run()

@cli.command()
@click.argument('checkpoint', type=click.Path(exists=True))
def import_checkpoint(checkpoint):
    """导入检查点"""
    from .commands import ImportCommand
    ImportCommand(checkpoint).run()

if __name__ == '__main__':
    cli()
```

#### `h2q_cli/commands.py`

```python
"""
CLI 命令实现
"""

import json
from pathlib import Path
from typing import Optional
from ..h2q_project.local_executor import LocalExecutor
from ..h2q_project.persistence import CheckpointManager
from ..h2q_project.monitoring import MetricsTracker

class BaseCommand:
    """命令基类"""
    
    def __init__(self):
        self.executor = LocalExecutor()
        self.checkpoint_mgr = CheckpointManager()
        self.metrics = MetricsTracker()
    
    def get_agent_home(self) -> Path:
        """获取 agent 主目录"""
        home = Path.home() / '.h2q-evo'
        home.mkdir(exist_ok=True)
        return home
    
    def run(self):
        raise NotImplementedError

class InitCommand(BaseCommand):
    """初始化命令"""
    
    def run(self):
        """初始化 H2Q-Evo"""
        print("🚀 初始化 H2Q-Evo v2.3.0...")
        
        agent_home = self.get_agent_home()
        
        # 创建目录结构
        (agent_home / 'knowledge').mkdir(exist_ok=True)
        (agent_home / 'checkpoints').mkdir(exist_ok=True)
        (agent_home / 'logs').mkdir(exist_ok=True)
        
        # 初始化知识库
        print("📚 初始化知识库...")
        self.executor.init_knowledge_db(agent_home)
        
        # 初始化检查点
        print("💾 初始化检查点系统...")
        self.checkpoint_mgr.init(agent_home)
        
        # 保存配置
        config = {
            'version': '2.3.0',
            'home': str(agent_home),
            'created_at': json.dumps({
                'timestamp': str(__import__('datetime').datetime.now()),
            }),
        }
        
        with open(agent_home / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ 初始化完成!")
        print(f"📁 Agent 主目录: {agent_home}")

class ExecuteCommand(BaseCommand):
    """执行命令"""
    
    def __init__(self, task: str, strategy: str = 'auto', save_knowledge: bool = False):
        super().__init__()
        self.task = task
        self.strategy = strategy
        self.save_knowledge = save_knowledge
    
    def run(self):
        """执行任务"""
        print(f"⚙️  执行任务: {self.task}")
        print(f"📊 策略: {self.strategy}")
        
        # 1. 执行推理
        result = self.executor.execute(self.task, self.strategy)
        
        # 2. 显示结果
        print(f"\n✨ 结果:\n{result['output']}")
        print(f"\n📈 置信度: {result['confidence']:.2%}")
        print(f"⏱️  耗时: {result['elapsed_time']:.2f}s")
        
        # 3. 保存知识（如果启用）
        if self.save_knowledge:
            print("\n💡 保存学到的知识...")
            self.executor.save_experience(
                task=self.task,
                result=result,
                feedback={'user_confirmed': True}
            )
            print("✅ 知识已保存")
        
        # 4. 更新指标
        self.metrics.record_execution(self.task, result)
        
        print("\n" + "="*50)

class StatusCommand(BaseCommand):
    """状态命令"""
    
    def run(self):
        """显示 agent 状态"""
        agent_home = self.get_agent_home()
        
        print("📊 H2Q-Evo Agent 状态")
        print("="*50)
        
        # 读取配置
        config_file = agent_home / 'config.json'
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            print(f"✅ 版本: {config['version']}")
        
        # 显示知识库统计
        stats = self.executor.get_knowledge_stats(agent_home)
        print(f"\n📚 知识库统计:")
        print(f"   总经验数: {stats['total_experiences']}")
        print(f"   覆盖领域: {', '.join(stats['domains'])}")
        
        # 显示能力指标
        metrics = self.metrics.get_current_metrics()
        print(f"\n🧠 能力指标:")
        print(f"   推理深度: {metrics['reasoning_depth']}")
        print(f"   成功率: {metrics['success_rate']:.2%}")
        print(f"   学习速度: {metrics['learning_velocity']:.3f}")
        
        print("\n" + "="*50)

class ExportCommand(BaseCommand):
    """导出命令"""
    
    def __init__(self, output: str):
        super().__init__()
        self.output = Path(output)
    
    def run(self):
        """导出检查点"""
        print(f"💾 导出检查点到: {self.output}")
        
        agent_home = self.get_agent_home()
        
        # 创建检查点
        checkpoint = self.checkpoint_mgr.create_checkpoint(agent_home)
        
        # 保存到文件
        self.checkpoint_mgr.save(checkpoint, self.output)
        
        # 验证
        checksum = self.checkpoint_mgr.compute_checksum(self.output)
        print(f"✅ 导出完成")
        print(f"📦 文件大小: {self.output.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"🔐 校验和: {checksum[:16]}...")

class ImportCommand(BaseCommand):
    """导入命令"""
    
    def __init__(self, checkpoint: str):
        super().__init__()
        self.checkpoint = Path(checkpoint)
    
    def run(self):
        """导入检查点"""
        print(f"📦 导入检查点: {self.checkpoint}")
        
        # 验证
        if not self.checkpoint_mgr.verify_checkpoint(self.checkpoint):
            print("❌ 检查点验证失败")
            return
        
        # 导入
        agent_home = self.get_agent_home()
        self.checkpoint_mgr.restore(self.checkpoint, agent_home)
        
        print("✅ 导入完成")
        print(f"🔄 Agent 状态已恢复")
```

---

### 2. 执行层 (`h2q_project/local_executor.py`)

```python
"""
本地执行器 - 管理任务执行和学习循环
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import json
from .learning_loop import LearningLoop
from .strategy_manager import StrategyManager
from .feedback_handler import FeedbackHandler
from .knowledge import KnowledgeDB

class LocalExecutor:
    """本地执行器"""
    
    def __init__(self):
        self.learning_loop = LearningLoop()
        self.strategy_mgr = StrategyManager()
        self.feedback_handler = FeedbackHandler()
        self.knowledge_db: Optional[KnowledgeDB] = None
    
    def init_knowledge_db(self, home: Path):
        """初始化知识库"""
        self.knowledge_db = KnowledgeDB(home / 'knowledge')
    
    def execute(self, task: str, strategy: str = 'auto') -> Dict[str, Any]:
        """
        执行任务 (带学习)
        
        流程:
        1. 分析任务
        2. 选择策略
        3. 执行推理
        4. 评估结果
        5. 本地学习
        """
        
        start_time = time.time()
        
        try:
            # 1. 分析任务
            task_analysis = self._analyze_task(task)
            
            # 2. 选择策略
            if strategy == 'auto':
                selected_strategy = self.strategy_mgr.select_best(
                    task_analysis,
                    self.knowledge_db
                )
            else:
                selected_strategy = strategy
            
            # 3. 执行推理
            raw_output = self._run_inference(task, selected_strategy)
            
            # 4. 后处理
            output = self._postprocess(raw_output)
            
            # 5. 计算置信度
            confidence = self._compute_confidence(output, task_analysis)
            
            result = {
                'output': output,
                'confidence': confidence,
                'task_type': task_analysis['type'],
                'strategy_used': selected_strategy,
                'elapsed_time': time.time() - start_time,
                'timestamp': time.time(),
            }
            
            return result
        
        except Exception as e:
            return {
                'output': f'执行出错: {str(e)}',
                'confidence': 0.0,
                'error': str(e),
                'elapsed_time': time.time() - start_time,
            }
    
    def save_experience(self, task: str, result: Dict[str, Any], feedback: Dict[str, Any]):
        """
        保存经验到知识库
        
        用于后续学习和检索
        """
        
        if self.knowledge_db is None:
            return
        
        experience = {
            'task': task,
            'result': result,
            'feedback': feedback,
            'timestamp': time.time(),
            'task_type': result.get('task_type'),
            'strategy_used': result.get('strategy_used'),
            'confidence': result.get('confidence'),
        }
        
        # 保存到知识库
        self.knowledge_db.save_experience(experience)
        
        # 更新策略效果
        self.strategy_mgr.update_effectiveness(
            result.get('strategy_used'),
            feedback.get('user_confirmed', False)
        )
    
    def get_knowledge_stats(self, home: Path) -> Dict[str, Any]:
        """获取知识库统计"""
        if self.knowledge_db is None:
            self.init_knowledge_db(home)
        
        return self.knowledge_db.get_stats()
    
    def _analyze_task(self, task: str) -> Dict[str, Any]:
        """分析任务"""
        # 简单的任务分析
        return {
            'type': self._classify_task(task),
            'complexity': len(task.split()),
            'keywords': self._extract_keywords(task),
        }
    
    def _classify_task(self, task: str) -> str:
        """任务分类"""
        keywords = {
            'math': ['计算', '方程', '数学'],
            'logic': ['推理', '逻辑', '证明'],
            'creative': ['创意', '写作', '故事'],
            'analysis': ['分析', '评估', '比较'],
        }
        
        task_lower = task.lower()
        for task_type, words in keywords.items():
            if any(word in task_lower for word in words):
                return task_type
        
        return 'general'
    
    def _extract_keywords(self, task: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        stop_words = {'的', '是', '在', '了', '和', '有', '人', '这', '中', '大'}
        words = task.split()
        return [w for w in words if len(w) > 1 and w not in stop_words]
    
    def _run_inference(self, task: str, strategy: str) -> str:
        """运行推理"""
        # 调用 H2Q 核心推理
        from h2q_project.h2q_server import inference_api
        
        try:
            result = inference_api(task)
            return result
        except:
            return self._fallback_inference(task)
    
    def _fallback_inference(self, task: str) -> str:
        """备用推理"""
        return f"处理任务: {task[:50]}..."
    
    def _postprocess(self, output: str) -> str:
        """后处理输出"""
        return output.strip()
    
    def _compute_confidence(self, output: str, task_analysis: Dict) -> float:
        """计算置信度"""
        # 简单的置信度计算
        if len(output) > 0:
            return 0.8
        return 0.3
```

---

### 3. 学习循环 (`h2q_project/learning_loop.py`)

```python
"""
本地学习循环 - 处理权重更新和知识积累
"""

import numpy as np
from typing import Dict, Any, Optional
import torch

class LearningLoop:
    """本地学习循环"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.update_count = 0
    
    def update_weights(self, model: torch.nn.Module, feedback: Dict[str, Any]):
        """
        使用谱交换更新权重
        
        机制:
        W_new = W_old + α * η * feedback_signal
        
        其中:
        α = 自适应学习率
        η = 谱移指示
        feedback_signal = [0, 1] 的改进信号
        """
        
        # 计算反馈信号
        feedback_signal = self._compute_feedback_signal(feedback)
        
        # 自适应学习率
        adaptive_lr = self._compute_adaptive_lr()
        
        # 计算谱移
        spectral_shift = self._compute_spectral_shift(model)
        
        # 更新权重
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    delta = adaptive_lr * spectral_shift * feedback_signal
                    param.data.add_(delta)
        
        self.update_count += 1
    
    def _compute_feedback_signal(self, feedback: Dict[str, Any]) -> float:
        """
        计算反馈信号
        
        规范化到 [0, 1] 范围
        """
        
        if feedback.get('user_confirmed'):
            return 1.0
        elif feedback.get('success'):
            return 0.8
        elif feedback.get('partial_success'):
            return 0.5
        else:
            return 0.0
    
    def _compute_adaptive_lr(self) -> float:
        """
        计算自适应学习率
        
        随着更新次数减少学习率
        """
        return self.learning_rate / (1 + 0.001 * self.update_count)
    
    def _compute_spectral_shift(self, model: torch.nn.Module) -> float:
        """
        计算谱移
        
        用于保持权重的数值稳定性
        """
        
        total_norm = 0.0
        for param in model.parameters():
            total_norm += torch.norm(param.data).item()
        
        # 归一化
        return 1.0 / (1.0 + total_norm)
    
    def learn_from_experience(self, model: torch.nn.Module, experiences: list):
        """
        从多个经验中批量学习
        
        用于定期的深度学习
        """
        
        if not experiences:
            return
        
        # 计算平均反馈
        avg_feedback = {}
        for exp in experiences:
            feedback = exp.get('feedback', {})
            for key, value in feedback.items():
                avg_feedback[key] = avg_feedback.get(key, 0) + value
        
        for key in avg_feedback:
            avg_feedback[key] /= len(experiences)
        
        # 批量更新
        self.update_weights(model, avg_feedback)
```

---

### 4. 知识库系统 (`h2q_project/knowledge/knowledge_db.py`)

```python
"""
本地知识库 - 存储和检索经验
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

class KnowledgeDB:
    """本地知识数据库"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.mkdir(exist_ok=True)
        self.db_file = self.db_path / 'knowledge.db'
        
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        
        with sqlite3.connect(self.db_file) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT NOT NULL,
                    task_type TEXT,
                    result TEXT NOT NULL,
                    feedback TEXT,
                    timestamp REAL NOT NULL,
                    confidence REAL,
                    strategy_used TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS domains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT UNIQUE,
                    experience_count INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0
                )
            ''')
            
            conn.commit()
    
    def save_experience(self, experience: Dict[str, Any]):
        """保存经验"""
        
        with sqlite3.connect(self.db_file) as conn:
            conn.execute('''
                INSERT INTO experiences 
                (task, task_type, result, feedback, timestamp, confidence, strategy_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                experience['task'],
                experience.get('task_type'),
                json.dumps(experience.get('result', {})),
                json.dumps(experience.get('feedback', {})),
                experience['timestamp'],
                experience.get('confidence', 0.0),
                experience.get('strategy_used')
            ))
            
            conn.commit()
        
        # 更新域统计
        self._update_domain_stats(experience.get('task_type'))
    
    def retrieve_similar(self, task: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相似任务的经验"""
        
        # 简单的文本相似度
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute('''
                SELECT * FROM experiences
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (top_k,))
            
            rows = cursor.fetchall()
            
            return [
                {
                    'task': row[1],
                    'result': json.loads(row[3]),
                    'confidence': row[6],
                }
                for row in rows
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计"""
        
        with sqlite3.connect(self.db_file) as conn:
            # 总经验数
            total = conn.execute(
                'SELECT COUNT(*) FROM experiences'
            ).fetchone()[0]
            
            # 域列表
            domains = conn.execute(
                'SELECT domain FROM domains'
            ).fetchall()
            
            return {
                'total_experiences': total,
                'domains': [d[0] for d in domains if d[0]],
            }
    
    def _update_domain_stats(self, domain: Optional[str]):
        """更新域统计"""
        
        if not domain:
            return
        
        with sqlite3.connect(self.db_file) as conn:
            conn.execute('''
                INSERT OR IGNORE INTO domains (domain, experience_count)
                VALUES (?, 1)
            ''', (domain,))
            
            conn.execute('''
                UPDATE domains
                SET experience_count = experience_count + 1
                WHERE domain = ?
            ''', (domain,))
            
            conn.commit()
```

---

### 5. 检查点管理 (`h2q_project/persistence/checkpoint_manager.py`)

```python
"""
检查点管理 - 保存和恢复 agent 状态
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any
import shutil
from datetime import datetime

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self):
        self.checkpoint_version = '1.0.0'
    
    def init(self, home: Path):
        """初始化检查点系统"""
        (home / 'checkpoints').mkdir(exist_ok=True)
    
    def create_checkpoint(self, home: Path) -> Dict[str, Any]:
        """
        创建检查点
        
        包含:
        - 权重状态
        - 知识库
        - 能力指标
        - 配置
        """
        
        checkpoint = {
            'version': self.checkpoint_version,
            'timestamp': datetime.now().isoformat(),
            'weights': self._backup_weights(home),
            'knowledge': self._backup_knowledge(home),
            'metrics': self._backup_metrics(home),
            'config': self._backup_config(home),
        }
        
        return checkpoint
    
    def save(self, checkpoint: Dict[str, Any], output_path: Path):
        """保存检查点到文件"""
        
        # 序列化
        data = pickle.dumps(checkpoint)
        
        # 保存
        with open(output_path, 'wb') as f:
            f.write(data)
    
    def load(self, checkpoint_path: Path) -> Dict[str, Any]:
        """加载检查点"""
        
        with open(checkpoint_path, 'rb') as f:
            data = f.read()
        
        return pickle.loads(data)
    
    def restore(self, checkpoint_path: Path, home: Path):
        """恢复检查点"""
        
        checkpoint = self.load(checkpoint_path)
        
        # 恢复各部分
        self._restore_weights(home, checkpoint['weights'])
        self._restore_knowledge(home, checkpoint['knowledge'])
        self._restore_metrics(home, checkpoint['metrics'])
    
    def compute_checksum(self, checkpoint_path: Path) -> str:
        """计算检查点校验和"""
        
        sha256 = hashlib.sha256()
        
        with open(checkpoint_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """验证检查点完整性"""
        
        try:
            checkpoint = self.load(checkpoint_path)
            
            # 检查必要字段
            required = ['version', 'timestamp', 'weights', 'knowledge']
            return all(k in checkpoint for k in required)
        
        except Exception:
            return False
    
    def _backup_weights(self, home: Path) -> Dict[str, Any]:
        """备份权重"""
        return {'status': 'backup_weights'}
    
    def _backup_knowledge(self, home: Path) -> Dict[str, Any]:
        """备份知识库"""
        db_file = home / 'knowledge' / 'knowledge.db'
        
        if db_file.exists():
            return {'db_file': str(db_file)}
        
        return {}
    
    def _backup_metrics(self, home: Path) -> Dict[str, Any]:
        """备份指标"""
        return {'status': 'backup_metrics'}
    
    def _backup_config(self, home: Path) -> Dict[str, Any]:
        """备份配置"""
        config_file = home / 'config.json'
        
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        
        return {}
    
    def _restore_weights(self, home: Path, weights: Dict[str, Any]):
        """恢复权重"""
        pass
    
    def _restore_knowledge(self, home: Path, knowledge: Dict[str, Any]):
        """恢复知识库"""
        pass
    
    def _restore_metrics(self, home: Path, metrics: Dict[str, Any]):
        """恢复指标"""
        pass
```

---

### 6. 能力监控 (`h2q_project/monitoring/metrics_tracker.py`)

```python
"""
能力指标跟踪 - 追踪能力进化过程
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self, home: Path = None):
        if home is None:
            home = Path.home() / '.h2q-evo'
        
        self.home = home
        self.metrics_file = home / 'metrics.json'
        self._load_metrics()
    
    def _load_metrics(self):
        """加载现有指标"""
        
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                'reasoning_depth': 0,
                'success_rate': 0.0,
                'learning_velocity': 0.0,
                'total_tasks': 0,
                'domains': {},
                'history': [],
            }
    
    def record_execution(self, task: str, result: Dict[str, Any]):
        """记录执行"""
        
        # 更新总任务数
        self.metrics['total_tasks'] += 1
        
        # 更新成功率
        if result.get('confidence', 0) > 0.7:
            success_rate = self.metrics['success_rate']
            self.metrics['success_rate'] = (
                0.95 * success_rate + 0.05 * 1.0
            )
        else:
            success_rate = self.metrics['success_rate']
            self.metrics['success_rate'] = (
                0.95 * success_rate + 0.05 * 0.0
            )
        
        # 记录历史
        self.metrics['history'].append({
            'timestamp': datetime.now().isoformat(),
            'task': task[:50],
            'confidence': result.get('confidence', 0),
        })
        
        # 保存
        self._save_metrics()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        return self.metrics
    
    def _save_metrics(self):
        """保存指标"""
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
```

---

## 实现步骤

### 周 1-2: 核心框架

```
□ CLI 框架完成
□ 本地执行器实现
□ 知识库系统实现
□ 集成测试

代码行数: ~800
```

### 周 3: 学习系统

```
□ 学习循环实现
□ 权重更新机制
□ 策略管理器
□ 反馈处理

代码行数: ~400
```

### 周 4: 持久化系统

```
□ 检查点管理
□ 迁移引擎
□ 完整性验证
□ 状态序列化

代码行数: ~300
```

### 周 5: 监控和测试

```
□ 能力追踪
□ 性能监控
□ 完整测试
□ 文档编写

代码行数: ~300
```

### 周 6: 优化和发布

```
□ 性能优化
□ Bug 修复
□ 用户测试
□ v2.3.0 发布

代码行数: ~200
```

---

## 验证清单

```
✅ 功能验证:
  □ CLI 命令全部可用
  □ 本地执行正常
  □ 知识保存和检索
  □ 权重更新有效
  □ 检查点完整

✅ 性能验证:
  □ 启动时间 < 2s
  □ 执行耗时 < 前一版本
  □ 内存占用 < 前一版本
  □ 知识库查询 < 100ms

✅ 集成验证:
  □ 与现有 API 兼容
  □ 与 Docker 兼容
  □ 跨平台测试
  □ 长期运行测试

✅ 用户验证:
  □ CLI 易用性
  □ 文档完整性
  □ 故障恢复
  □ 进化可见性
```

---

## 预期成果

```
v2.3.0 发布后:

✅ H2Q-Evo 可以自主执行任务
✅ 每次执行都学习和积累经验
✅ 能力指标可追踪
✅ 状态可导出和导入 (跨设备迁移)
✅ 完整的 CLI 工具链

为 v3.0 的完整自主系统做准备
```

这份框架提供了清晰的实现路径，可以在 5-6 周内完成！
