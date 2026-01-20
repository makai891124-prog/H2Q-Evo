#!/usr/bin/env python3
"""
推理模板库：为证明链提供模板ID/名称/步骤及可记录的执行轨迹。
"""

import random
import time
from datetime import datetime
from typing import Dict, List, Sequence


TEMPLATES: Dict[str, Dict] = {
    "mathematics": {
        "id": "P1",
        "name": "约束优化问题",
        "steps": [
            "识别目标函数与约束",
            "构造拉格朗日函数 L = f + λ·g",
            "求解一阶必要条件 ∇L = 0",
            "验证KKT/二阶充分条件",
            "检查边界与可行域"
        ],
    },
    "physics": {
        "id": "P2",
        "name": "量子力学计算",
        "steps": [
            "识别量子系统与势能形式",
            "构建哈密顿算符 Ĥ = T̂ + V̂",
            "求解定态薛定谔方程 Ĥψ = Eψ",
            "计算可观测量期望",
            "验证归一化与边界条件"
        ],
    },
    "chemistry": {
        "id": "P3",
        "name": "化学反应机理",
        "steps": [
            "识别反应类型与底物",
            "分析过渡态/活化能",
            "确定速控步与动力学模型",
            "评估催化/溶剂效应",
            "验证产物与立体化学"
        ],
    },
    "biology": {
        "id": "P4",
        "name": "生物分子动力学",
        "steps": [
            "界定体系与初始构象",
            "建立力场/能量函数",
            "模拟轨迹与构象采样",
            "计算自由能/稳定性",
            "验证实验约束或功能位点"
        ],
    },
    "engineering": {
        "id": "P5",
        "name": "工程结构优化",
        "steps": [
            "建立物理/几何模型",
            "网格与载荷/边界设置",
            "求解应力/模态",
            "灵敏度分析与拓扑优化",
            "验证约束与安全裕度"
        ],
    },
}


def select_template(domain: str) -> Dict:
    """返回域对应的模板（未知域返回通用占位符）。"""
    if domain in TEMPLATES:
        return TEMPLATES[domain]
    return {
        "id": "P0",
        "name": "通用跨域推理",
        "steps": [
            "问题分解",
            "知识检索",
            "逻辑推导",
            "一致性检查"
        ],
    }


def build_trace(template: Dict) -> List[Dict]:
    """生成带时间戳和耗时的执行轨迹（轻量模拟）。"""
    steps = template.get("steps", [])
    trace: List[Dict] = []
    for idx, step in enumerate(steps, 1):
        start = time.time()
        duration_ms = random.uniform(5, 30)
        finish = start + duration_ms / 1000.0
        trace.append({
            "step": idx,
            "description": step,
            "status": "done",
            "confidence": 0.9,
            "started_at": datetime.fromtimestamp(start).isoformat(),
            "finished_at": datetime.fromtimestamp(finish).isoformat(),
            "duration_ms": round(duration_ms, 2)
        })
    return trace


def build_runtime_trace(template: Dict, phase_durations_ms: Sequence[float]) -> List[Dict]:
    """根据真实测量的阶段耗时构造执行轨迹。

    phase_durations_ms: 依次对应模板步骤的耗时，若步骤多于耗时，剩余标记为pending。
    """
    steps = template.get("steps", [])
    trace: List[Dict] = []
    now = time.time()
    current = now
    for idx, step in enumerate(steps, 1):
        dur = float(phase_durations_ms[idx - 1]) if idx - 1 < len(phase_durations_ms) else 0.0
        start_ts = current
        finish_ts = current + dur / 1000.0
        status = "done" if dur > 0 else "pending"
        confidence = 0.9 if dur > 0 else 0.5
        trace.append({
            "step": idx,
            "description": step,
            "status": status,
            "confidence": confidence,
            "started_at": datetime.fromtimestamp(start_ts).isoformat(),
            "finished_at": datetime.fromtimestamp(finish_ts).isoformat(),
            "duration_ms": round(dur, 2)
        })
        current = finish_ts
    return trace


if __name__ == "__main__":
    import json
    print(json.dumps(TEMPLATES, ensure_ascii=False, indent=2))
