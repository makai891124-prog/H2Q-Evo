"""H2Q 标准人类测试基准 (Standardized Human Benchmarks).

引入外部可比较的标准化人类测试基准:
1. MMLU (Massive Multitask Language Understanding) - 57 学科
2. GSM8K (Grade School Math) - 小学数学推理
3. ARC (AI2 Reasoning Challenge) - 科学推理
4. HellaSwag - 常识推理
5. TruthfulQA - 事实问答

这些基准与 GPT-4、Claude、Llama 等模型的公开评测结果可比较。
"""

import numpy as np
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time
import itertools
import json
from pathlib import Path


# ============================================================================
# 基准类型
# ============================================================================

class BenchmarkType(Enum):
    """基准类型."""
    MMLU = "mmlu"           # 多任务语言理解
    GSM8K = "gsm8k"         # 小学数学
    ARC = "arc"             # 科学推理
    HELLASWAG = "hellaswag" # 常识推理
    TRUTHFULQA = "truthfulqa"  # 事实问答
    HUMANEVAL = "humaneval"  # 代码生成


class Difficulty(Enum):
    """难度级别."""
    ELEMENTARY = "elementary"  # 小学
    MIDDLE = "middle"          # 初中
    HIGH = "high"              # 高中
    COLLEGE = "college"        # 大学
    GRADUATE = "graduate"      # 研究生


@dataclass
class BenchmarkQuestion:
    """基准测试题目."""
    id: str
    benchmark: BenchmarkType
    subject: str
    difficulty: Difficulty
    question: str
    choices: List[str]  # 选择题选项
    correct_answer: int  # 正确选项索引
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """基准测试结果."""
    benchmark: BenchmarkType
    total_questions: int
    correct_answers: float
    accuracy: float
    by_subject: Dict[str, Dict[str, float]]
    by_difficulty: Dict[str, float]
    time_taken: float
    timestamp: str
    multi_select_accuracy: float = 0.0


class PublicBenchmarkLoader:
    """从公开基准数据集中加载题目（HuggingFace datasets）。"""

    def __init__(self, cache_dir: Optional[str] = None):
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as e:
            raise RuntimeError("缺少 datasets 依赖，无法加载公开基准测试数据。") from e

        self._load_dataset = load_dataset
        self.cache_dir = cache_dir

    def load_questions(self, benchmark: BenchmarkType, n: int = 50) -> List[BenchmarkQuestion]:
        if benchmark == BenchmarkType.MMLU:
            return self._load_mmlu(n)
        if benchmark == BenchmarkType.GSM8K:
            return self._load_gsm8k(n)
        if benchmark == BenchmarkType.ARC:
            return self._load_arc(n)
        if benchmark == BenchmarkType.HELLASWAG:
            return self._load_hellaswag(n)
        if benchmark == BenchmarkType.TRUTHFULQA:
            return self._load_truthfulqa(n)
        if benchmark == BenchmarkType.HUMANEVAL:
            return self._load_humaneval(n)
        return []

    def _safe_load_dataset(self, name: str, config: Optional[str], split: str):
        """优先本地/常规加载，失败则流式回退。"""
        try:
            return self._load_dataset(name, config, split=split, cache_dir=self.cache_dir), False
        except Exception:
            ds = self._load_dataset(name, config, split=split, cache_dir=self.cache_dir, streaming=True)
            return ds, True

    def _load_mmlu(self, n: int) -> List[BenchmarkQuestion]:
        ds, streaming = self._safe_load_dataset("cais/mmlu", "all", split="test")
        if streaming:
            items = list(itertools.islice(ds, n))
        else:
            items = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        questions = []
        for idx, item in enumerate(items):
            questions.append(BenchmarkQuestion(
                id=f"mmlu_{idx}",
                benchmark=BenchmarkType.MMLU,
                subject=item.get("subject", "unknown"),
                difficulty=Difficulty.COLLEGE,
                question=item["question"],
                choices=list(item["choices"]),
                correct_answer=int(item["answer"]),
                metadata={"source": "cais/mmlu"}
            ))
        return questions

    def _load_gsm8k(self, n: int) -> List[BenchmarkQuestion]:
        ds, streaming = self._safe_load_dataset("gsm8k", "main", split="test")
        if streaming:
            items = list(itertools.islice(ds, n))
        else:
            items = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        answers = [self._extract_final_answer(x["answer"]) for x in items]
        questions = []
        for idx, item in enumerate(items):
            correct = self._extract_final_answer(item["answer"])
            choices = self._build_numeric_choices(correct, answers)
            questions.append(BenchmarkQuestion(
                id=f"gsm8k_{idx}",
                benchmark=BenchmarkType.GSM8K,
                subject="arithmetic",
                difficulty=Difficulty.MIDDLE,
                question=item["question"],
                choices=choices,
                correct_answer=choices.index(str(correct)),
                metadata={"source": "gsm8k"}
            ))
        return questions

    def _load_arc(self, n: int) -> List[BenchmarkQuestion]:
        ds, streaming = self._safe_load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        if streaming:
            items = list(itertools.islice(ds, n))
        else:
            items = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        questions = []
        for idx, item in enumerate(items):
            choices_obj = item.get("choices", {})
            labels = list(choices_obj.get("label", []))
            texts = list(choices_obj.get("text", []))
            answer_key = item.get("answerKey", "")
            if answer_key in labels:
                correct_idx = labels.index(answer_key)
            else:
                try:
                    correct_idx = int(answer_key) - 1
                except Exception:
                    correct_idx = 0
            questions.append(BenchmarkQuestion(
                id=f"arc_{idx}",
                benchmark=BenchmarkType.ARC,
                subject=item.get("category", "science"),
                difficulty=Difficulty.MIDDLE,
                question=item.get("question", ""),
                choices=texts,
                correct_answer=correct_idx,
                metadata={"source": "allenai/ai2_arc"}
            ))
        return questions

    def _load_hellaswag(self, n: int) -> List[BenchmarkQuestion]:
        ds, streaming = self._safe_load_dataset("hellaswag", None, split="validation")
        if streaming:
            items = list(itertools.islice(ds, n))
        else:
            items = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        questions = []
        for idx, item in enumerate(items):
            ctx = item.get("ctx", "")
            endings = list(item.get("endings", []))
            label = int(item.get("label", 0))
            questions.append(BenchmarkQuestion(
                id=f"hellaswag_{idx}",
                benchmark=BenchmarkType.HELLASWAG,
                subject="commonsense",
                difficulty=Difficulty.MIDDLE,
                question=ctx,
                choices=endings,
                correct_answer=label,
                metadata={"source": "hellaswag"}
            ))
        return questions

    def _load_truthfulqa(self, n: int) -> List[BenchmarkQuestion]:
        ds = self._load_dataset("truthful_qa", "multiple_choice", split="validation", cache_dir=self.cache_dir)
        items = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        questions = []
        for idx, item in enumerate(items):
            targets = item.get("mc1_targets", {})
            choices = list(targets.get("choices", []))
            labels = list(targets.get("labels", []))
            correct_idx = labels.index(1) if 1 in labels else 0
            questions.append(BenchmarkQuestion(
                id=f"truthfulqa_{idx}",
                benchmark=BenchmarkType.TRUTHFULQA,
                subject="truthfulness",
                difficulty=Difficulty.HIGH,
                question=item.get("question", ""),
                choices=choices,
                correct_answer=correct_idx,
                metadata={"source": "truthful_qa"}
            ))
        return questions

    def _load_humaneval(self, n: int) -> List[BenchmarkQuestion]:
        ds = self._load_dataset("openai_humaneval", split="test", cache_dir=self.cache_dir)
        items = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        questions = []
        for idx, item in enumerate(items):
            questions.append(BenchmarkQuestion(
                id=f"humaneval_{idx}",
                benchmark=BenchmarkType.HUMANEVAL,
                subject="code",
                difficulty=Difficulty.COLLEGE,
                question=item.get("prompt", ""),
                choices=["<code>"],
                correct_answer=0,
                metadata={"source": "openai_humaneval", "entry_point": item.get("entry_point")}
            ))
        return questions

    def _extract_final_answer(self, answer_text: str) -> int:
        if "####" in answer_text:
            answer_text = answer_text.split("####")[-1]
        digits = "".join(ch for ch in answer_text if ch.isdigit() or ch in "-.")
        try:
            return int(float(digits))
        except Exception:
            return 0

    def _build_numeric_choices(self, correct: int, pool: List[int]) -> List[str]:
        choices = {correct}
        random.seed(42)
        while len(choices) < 4 and pool:
            choices.add(random.choice(pool))
        while len(choices) < 4:
            choices.add(correct + random.randint(-10, 10))
        choices_list = [str(c) for c in choices]
        random.shuffle(choices_list)
        return choices_list


# ============================================================================
# MMLU 题库 (子集示例)
# ============================================================================

class MMLUBenchmark:
    """MMLU 基准 - 多学科语言理解."""
    
    # MMLU 涵盖的学科
    SUBJECTS = {
        "stem": ["mathematics", "physics", "chemistry", "biology", "computer_science"],
        "humanities": ["history", "philosophy", "law"],
        "social_sciences": ["economics", "psychology", "sociology"],
        "other": ["business", "health", "miscellaneous"],
    }
    
    def __init__(self):
        self.questions: List[BenchmarkQuestion] = []
        self._generate_questions()
    
    def _generate_questions(self):
        """生成 MMLU 风格的问题."""
        
        # 数学
        math_questions = [
            {
                "question": "What is the derivative of x²?",
                "choices": ["x", "2x", "x²", "2"],
                "correct": 1,
                "difficulty": Difficulty.HIGH,
            },
            {
                "question": "What is the integral of 2x?",
                "choices": ["x", "x²", "2x²", "x² + C"],
                "correct": 3,
                "difficulty": Difficulty.COLLEGE,
            },
            {
                "question": "If f(x) = 3x + 2, what is f(4)?",
                "choices": ["10", "12", "14", "16"],
                "correct": 2,
                "difficulty": Difficulty.MIDDLE,
            },
            {
                "question": "What is the sum of angles in a triangle?",
                "choices": ["90°", "180°", "270°", "360°"],
                "correct": 1,
                "difficulty": Difficulty.ELEMENTARY,
            },
            {
                "question": "What is log₁₀(1000)?",
                "choices": ["2", "3", "10", "100"],
                "correct": 1,
                "difficulty": Difficulty.HIGH,
            },
        ]
        
        for i, q in enumerate(math_questions):
            self.questions.append(BenchmarkQuestion(
                id=f"mmlu_math_{i}",
                benchmark=BenchmarkType.MMLU,
                subject="mathematics",
                difficulty=q["difficulty"],
                question=q["question"],
                choices=q["choices"],
                correct_answer=q["correct"],
            ))
        
        # 物理
        physics_questions = [
            {
                "question": "What is the SI unit of force?",
                "choices": ["Watt", "Joule", "Newton", "Pascal"],
                "correct": 2,
                "difficulty": Difficulty.MIDDLE,
            },
            {
                "question": "What is the speed of light in vacuum (approximately)?",
                "choices": ["3×10⁶ m/s", "3×10⁸ m/s", "3×10¹⁰ m/s", "3×10⁴ m/s"],
                "correct": 1,
                "difficulty": Difficulty.HIGH,
            },
            {
                "question": "Newton's second law states that F equals:",
                "choices": ["mv", "ma", "mv²", "½mv²"],
                "correct": 1,
                "difficulty": Difficulty.HIGH,
            },
        ]
        
        for i, q in enumerate(physics_questions):
            self.questions.append(BenchmarkQuestion(
                id=f"mmlu_physics_{i}",
                benchmark=BenchmarkType.MMLU,
                subject="physics",
                difficulty=q["difficulty"],
                question=q["question"],
                choices=q["choices"],
                correct_answer=q["correct"],
            ))
        
        # 历史
        history_questions = [
            {
                "question": "In what year did World War II end?",
                "choices": ["1943", "1944", "1945", "1946"],
                "correct": 2,
                "difficulty": Difficulty.MIDDLE,
            },
            {
                "question": "Who was the first President of the United States?",
                "choices": ["Thomas Jefferson", "John Adams", "George Washington", "Benjamin Franklin"],
                "correct": 2,
                "difficulty": Difficulty.ELEMENTARY,
            },
        ]
        
        for i, q in enumerate(history_questions):
            self.questions.append(BenchmarkQuestion(
                id=f"mmlu_history_{i}",
                benchmark=BenchmarkType.MMLU,
                subject="history",
                difficulty=q["difficulty"],
                question=q["question"],
                choices=q["choices"],
                correct_answer=q["correct"],
            ))
        
        # 计算机科学
        cs_questions = [
            {
                "question": "What is the time complexity of binary search?",
                "choices": ["O(1)", "O(n)", "O(log n)", "O(n²)"],
                "correct": 2,
                "difficulty": Difficulty.COLLEGE,
            },
            {
                "question": "Which data structure uses LIFO (Last In First Out)?",
                "choices": ["Queue", "Stack", "Tree", "Graph"],
                "correct": 1,
                "difficulty": Difficulty.HIGH,
            },
            {
                "question": "What does CPU stand for?",
                "choices": ["Central Processing Unit", "Computer Personal Unit", "Central Program Utility", "Core Processing Unit"],
                "correct": 0,
                "difficulty": Difficulty.ELEMENTARY,
            },
        ]
        
        for i, q in enumerate(cs_questions):
            self.questions.append(BenchmarkQuestion(
                id=f"mmlu_cs_{i}",
                benchmark=BenchmarkType.MMLU,
                subject="computer_science",
                difficulty=q["difficulty"],
                question=q["question"],
                choices=q["choices"],
                correct_answer=q["correct"],
            ))
    
    def get_questions(self, subject: str = None, 
                      n: int = None) -> List[BenchmarkQuestion]:
        """获取问题."""
        questions = self.questions
        
        if subject:
            questions = [q for q in questions if q.subject == subject]
        
        if n:
            questions = questions[:n]
        
        return questions


# ============================================================================
# GSM8K 基准
# ============================================================================

class GSM8KBenchmark:
    """GSM8K 基准 - 小学数学推理."""
    
    def __init__(self):
        self.questions: List[BenchmarkQuestion] = []
        self._generate_questions()
    
    def _generate_questions(self):
        """生成 GSM8K 风格的数学问题."""
        
        problems = [
            {
                "question": "Janet has 3 apples. Her friend gives her 5 more apples. How many apples does Janet have now?",
                "choices": ["5", "7", "8", "10"],
                "correct": 2,
                "answer_value": 8,
            },
            {
                "question": "A store has 24 shirts. If they sell 8 shirts, how many shirts are left?",
                "choices": ["14", "16", "18", "20"],
                "correct": 1,
                "answer_value": 16,
            },
            {
                "question": "Tom has 12 marbles. He gives 4 marbles to each of his 2 friends. How many marbles does Tom have left?",
                "choices": ["2", "4", "6", "8"],
                "correct": 1,
                "answer_value": 4,
            },
            {
                "question": "A farmer has 5 cows. Each cow produces 3 liters of milk per day. How many liters of milk does the farmer get in 2 days?",
                "choices": ["15", "20", "25", "30"],
                "correct": 3,
                "answer_value": 30,
            },
            {
                "question": "Sarah has $20. She buys a book for $8 and a pen for $3. How much money does she have left?",
                "choices": ["$7", "$8", "$9", "$10"],
                "correct": 2,
                "answer_value": 9,
            },
            {
                "question": "A train travels at 60 km/h. How far does it travel in 3 hours?",
                "choices": ["120 km", "150 km", "180 km", "200 km"],
                "correct": 2,
                "answer_value": 180,
            },
            {
                "question": "There are 4 baskets with 6 oranges each. If 5 oranges are rotten, how many good oranges are there?",
                "choices": ["17", "18", "19", "20"],
                "correct": 2,
                "answer_value": 19,
            },
            {
                "question": "A rectangle has a length of 8 cm and a width of 5 cm. What is its area?",
                "choices": ["30 cm²", "35 cm²", "40 cm²", "45 cm²"],
                "correct": 2,
                "answer_value": 40,
            },
        ]
        
        for i, p in enumerate(problems):
            self.questions.append(BenchmarkQuestion(
                id=f"gsm8k_{i}",
                benchmark=BenchmarkType.GSM8K,
                subject="grade_school_math",
                difficulty=Difficulty.ELEMENTARY,
                question=p["question"],
                choices=p["choices"],
                correct_answer=p["correct"],
                metadata={"answer_value": p["answer_value"]},
            ))
    
    def get_questions(self, n: int = None) -> List[BenchmarkQuestion]:
        """获取问题."""
        questions = self.questions
        if n:
            questions = questions[:n]
        return questions


# ============================================================================
# ARC 基准
# ============================================================================

class ARCBenchmark:
    """ARC 基准 - 科学推理."""
    
    def __init__(self):
        self.questions: List[BenchmarkQuestion] = []
        self._generate_questions()
    
    def _generate_questions(self):
        """生成 ARC 风格的科学问题."""
        
        problems = [
            {
                "question": "What is the main gas in Earth's atmosphere?",
                "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"],
                "correct": 1,
                "difficulty": Difficulty.MIDDLE,
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
                "correct": 1,
                "difficulty": Difficulty.ELEMENTARY,
            },
            {
                "question": "What is the chemical formula for water?",
                "choices": ["CO₂", "H₂O", "NaCl", "O₂"],
                "correct": 1,
                "difficulty": Difficulty.MIDDLE,
            },
            {
                "question": "Which organ pumps blood through the body?",
                "choices": ["Brain", "Lungs", "Heart", "Liver"],
                "correct": 2,
                "difficulty": Difficulty.ELEMENTARY,
            },
            {
                "question": "What causes the seasons on Earth?",
                "choices": ["Distance from the Sun", "Earth's tilt on its axis", "The Moon's gravity", "Solar flares"],
                "correct": 1,
                "difficulty": Difficulty.HIGH,
            },
            {
                "question": "What is the smallest unit of matter?",
                "choices": ["Cell", "Molecule", "Atom", "Electron"],
                "correct": 2,
                "difficulty": Difficulty.MIDDLE,
            },
            {
                "question": "Which type of energy does a moving car have?",
                "choices": ["Potential energy", "Kinetic energy", "Chemical energy", "Nuclear energy"],
                "correct": 1,
                "difficulty": Difficulty.MIDDLE,
            },
        ]
        
        for i, p in enumerate(problems):
            self.questions.append(BenchmarkQuestion(
                id=f"arc_{i}",
                benchmark=BenchmarkType.ARC,
                subject="science",
                difficulty=p["difficulty"],
                question=p["question"],
                choices=p["choices"],
                correct_answer=p["correct"],
            ))
    
    def get_questions(self, n: int = None) -> List[BenchmarkQuestion]:
        """获取问题."""
        questions = self.questions
        if n:
            questions = questions[:n]
        return questions


# ============================================================================
# HellaSwag 基准
# ============================================================================

class HellaSwagBenchmark:
    """HellaSwag 基准 - 常识推理/句子完成."""
    
    def __init__(self):
        self.questions: List[BenchmarkQuestion] = []
        self._generate_questions()
    
    def _generate_questions(self):
        """生成常识推理问题."""
        
        problems = [
            {
                "question": "A person is cooking in the kitchen. They turn on the stove and...",
                "choices": [
                    "start swimming",
                    "put a pan on it to heat up",
                    "go to sleep",
                    "read a book"
                ],
                "correct": 1,
            },
            {
                "question": "It starts raining heavily. Most people outside...",
                "choices": [
                    "take off their clothes",
                    "seek shelter or open umbrellas",
                    "start dancing",
                    "lie down on the ground"
                ],
                "correct": 1,
            },
            {
                "question": "A student is studying for an exam. To prepare well, they should...",
                "choices": [
                    "watch TV all night",
                    "review notes and practice problems",
                    "play video games",
                    "throw away their books"
                ],
                "correct": 1,
            },
            {
                "question": "The traffic light turns red. Drivers should...",
                "choices": [
                    "speed up",
                    "stop their vehicles",
                    "honk continuously",
                    "close their eyes"
                ],
                "correct": 1,
            },
            {
                "question": "A baby is crying. The most likely reason is that the baby...",
                "choices": [
                    "wants to do math homework",
                    "is hungry, tired, or uncomfortable",
                    "wants to drive a car",
                    "is thinking about philosophy"
                ],
                "correct": 1,
            },
        ]
        
        for i, p in enumerate(problems):
            self.questions.append(BenchmarkQuestion(
                id=f"hellaswag_{i}",
                benchmark=BenchmarkType.HELLASWAG,
                subject="commonsense",
                difficulty=Difficulty.MIDDLE,
                question=p["question"],
                choices=p["choices"],
                correct_answer=p["correct"],
            ))
    
    def get_questions(self, n: int = None) -> List[BenchmarkQuestion]:
        """获取问题."""
        questions = self.questions
        if n:
            questions = questions[:n]
        return questions


# ============================================================================
# 综合基准评估器
# ============================================================================

class StandardBenchmarkEvaluator:
    """标准基准评估器 - 运行所有基准测试."""
    
    # 公开的模型基准分数 (用于比较)
    PUBLIC_SCORES = {
        BenchmarkType.MMLU: {
            "GPT-4": 86.4,
            "Claude-3": 86.8,
            "GPT-3.5": 70.0,
            "Llama-2-70B": 69.8,
            "Human-Expert": 89.8,
            "Human-Average": 34.5,
        },
        BenchmarkType.GSM8K: {
            "GPT-4": 92.0,
            "Claude-3": 88.0,
            "GPT-3.5": 57.1,
            "Llama-2-70B": 56.8,
            "Human-Expert": 95.0,
        },
        BenchmarkType.ARC: {
            "GPT-4": 96.3,
            "Claude-3": 96.4,
            "GPT-3.5": 85.2,
            "Llama-2-70B": 67.3,
            "Human-Expert": 95.0,
        },
        BenchmarkType.HELLASWAG: {
            "GPT-4": 95.3,
            "Claude-3": 95.0,
            "GPT-3.5": 85.5,
            "Llama-2-70B": 87.3,
            "Human-Average": 95.6,
        },
    }
    
    def __init__(self, public_only: bool = True):
        self.public_only = public_only and os.getenv("ALLOW_SYNTHETIC_BENCHMARKS", "0") != "1"
        self.results: Dict[BenchmarkType, BenchmarkResult] = {}
        self.loader = PublicBenchmarkLoader() if self.public_only else None
    
    def evaluate_single(self, benchmark_type: BenchmarkType,
                        answer_func: callable,
                        n_questions: int = None) -> BenchmarkResult:
        """评估单个基准.
        
        Args:
            benchmark_type: 基准类型
            answer_func: 回答函数 (question, choices) -> int
            n_questions: 问题数量
        """
        if self.public_only:
            questions = self.loader.load_questions(benchmark_type, n_questions or 50)
        else:
            benchmark = {
                BenchmarkType.MMLU: MMLUBenchmark(),
                BenchmarkType.GSM8K: GSM8KBenchmark(),
                BenchmarkType.ARC: ARCBenchmark(),
                BenchmarkType.HELLASWAG: HellaSwagBenchmark(),
            }[benchmark_type]
            questions = benchmark.get_questions(n_questions)
        
        start_time = time.time()
        
        correct = 0.0
        multi_select_score_sum = 0.0
        by_subject: Dict[str, Dict[str, int]] = {}
        by_difficulty: Dict[str, Dict[str, int]] = {}
        
        for q in questions:
            # 获取答案
            try:
                predicted = answer_func(q.question, q.choices)
            except Exception:
                predicted = -1

            predicted_index, selected_set, multi_select_score = self._score_prediction(
                predicted, q.correct_answer, len(q.choices)
            )

            is_correct = predicted_index == q.correct_answer
            if is_correct:
                correct += 1
            multi_select_score_sum += multi_select_score
            
            # 按学科统计
            if q.subject not in by_subject:
                by_subject[q.subject] = {"correct": 0, "total": 0}
            by_subject[q.subject]["total"] += 1
            if is_correct:
                by_subject[q.subject]["correct"] += 1
            
            # 按难度统计
            diff = q.difficulty.value
            if diff not in by_difficulty:
                by_difficulty[diff] = {"correct": 0, "total": 0}
            by_difficulty[diff]["total"] += 1
            if is_correct:
                by_difficulty[diff]["correct"] += 1
        
        elapsed = time.time() - start_time
        
        # 计算准确率 (单选) 与多选得分
        accuracy = correct / len(questions) if questions else 0.0
        multi_select_accuracy = multi_select_score_sum / len(questions) if questions else 0.0
        
        subject_acc = {
            subj: stats["correct"] / stats["total"]
            for subj, stats in by_subject.items()
        }
        
        diff_acc = {
            diff: stats["correct"] / stats["total"]
            for diff, stats in by_difficulty.items()
        }
        
        result = BenchmarkResult(
            benchmark=benchmark_type,
            total_questions=len(questions),
            correct_answers=correct,
            accuracy=accuracy,
            by_subject=subject_acc,
            by_difficulty=diff_acc,
            time_taken=elapsed,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            multi_select_accuracy=multi_select_accuracy,
        )
        
        self.results[benchmark_type] = result
        return result
    
    def evaluate_all(self, answer_func: callable,
                     n_per_benchmark: int = None) -> Dict[BenchmarkType, BenchmarkResult]:
        """评估所有基准."""
        results = {}
        
        benchmark_list = [
            BenchmarkType.MMLU,
            BenchmarkType.GSM8K,
            BenchmarkType.ARC,
            BenchmarkType.HELLASWAG,
        ]
        if not self.public_only:
            benchmark_list = list(self.benchmarks.keys())

        for btype in benchmark_list:
            result = self.evaluate_single(btype, answer_func, n_per_benchmark)
            results[btype] = result
        
        return results

    def _score_prediction(self, predicted: Any, correct: int, num_choices: int) -> Tuple[int, List[int], float]:
        """对预测进行多选评分，降低蒙对影响."""
        # 允许预测返回: int | list[int] | dict{selected/ranked/probs}
        selected: List[int] = []
        predicted_index = 0

        if isinstance(predicted, dict):
            if "selected" in predicted and isinstance(predicted["selected"], list):
                selected = [int(x) for x in predicted["selected"] if isinstance(x, (int, str))]
            elif "ranked" in predicted and isinstance(predicted["ranked"], list):
                selected = [int(x) for x in predicted["ranked"][: self._default_k(num_choices)]]
            elif "probs" in predicted and isinstance(predicted["probs"], list):
                probs = list(predicted["probs"])
                ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                selected = ranked[: self._default_k(num_choices)]
            if "top" in predicted and isinstance(predicted["top"], int):
                predicted_index = int(predicted["top"])

        elif isinstance(predicted, (list, tuple)):
            selected = [int(x) for x in predicted]
        elif isinstance(predicted, (int, np.integer)):
            predicted_index = int(predicted)

        if not selected:
            selected = [predicted_index]

        selected = [s for s in selected if 0 <= s < num_choices]
        if not selected:
            selected = [0]

        predicted_index = selected[0]
        multi_select_score = (1.0 / len(selected)) if correct in selected else 0.0
        return predicted_index, selected, multi_select_score

    def _default_k(self, num_choices: int) -> int:
        if num_choices >= 5:
            return 3
        return 2
    
    def compare_with_models(self, benchmark_type: BenchmarkType,
                            our_accuracy: float) -> Dict[str, Any]:
        """与公开模型分数比较."""
        public_scores = self.PUBLIC_SCORES.get(benchmark_type, {})
        
        comparison = {
            "our_score": our_accuracy * 100,
            "comparisons": {},
        }
        
        for model, score in public_scores.items():
            diff = (our_accuracy * 100) - score
            comparison["comparisons"][model] = {
                "their_score": score,
                "difference": diff,
                "status": "ahead" if diff > 0 else "behind" if diff < 0 else "equal",
            }
        
        return comparison
    
    def generate_report(self) -> str:
        """生成综合报告."""
        report = []
        report.append("=" * 70)
        report.append("H2Q AGI 标准人类基准测试报告")
        report.append("=" * 70)
        report.append(f"\n测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_correct = 0
        overall_total = 0
        
        for btype, result in self.results.items():
            report.append(f"\n{'='*60}")
            report.append(f"[{btype.value.upper()}]")
            report.append(f"  总题数: {result.total_questions}")
            report.append(f"  正确数(单选): {result.correct_answers:.1f}")
            report.append(f"  准确率(单选): {result.accuracy * 100:.1f}%")
            report.append(f"  多选评分: {result.multi_select_accuracy * 100:.1f}%")
            report.append(f"  用时: {result.time_taken:.2f}s")
            
            overall_correct += result.correct_answers
            overall_total += result.total_questions
            
            # 与公开模型比较
            comparison = self.compare_with_models(btype, result.accuracy)
            report.append(f"\n  与其他模型比较:")
            for model, comp in comparison["comparisons"].items():
                status_icon = "✅" if comp["status"] == "ahead" else "❌" if comp["status"] == "behind" else "="
                report.append(f"    {status_icon} vs {model}: {comp['their_score']:.1f}% ({comp['difference']:+.1f})")
            
            # 按学科细分
            if result.by_subject:
                report.append(f"\n  按学科:")
                for subj, acc in result.by_subject.items():
                    report.append(f"    {subj}: {acc * 100:.1f}%")
        
        # 总体评分
        overall_acc = overall_correct / overall_total if overall_total > 0 else 0
        report.append(f"\n{'='*60}")
        report.append(f"总体评分(单选): {overall_acc * 100:.1f}%")
        report.append(f"总题数: {overall_total}")
        report.append(f"总正确(单选): {overall_correct:.1f}")
        
        # 等级评定
        if overall_acc >= 0.90:
            grade = "卓越 (Expert Level)"
        elif overall_acc >= 0.80:
            grade = "优秀 (Above Human Average)"
        elif overall_acc >= 0.70:
            grade = "良好 (Human Average)"
        elif overall_acc >= 0.60:
            grade = "及格 (Below Average)"
        else:
            grade = "需要提升 (Needs Improvement)"
        
        report.append(f"等级: {grade}")
        report.append("=" * 70)
        
        return "\n".join(report)


# ============================================================================
# AGI 基准答题器
# ============================================================================

class AGIBenchmarkAnswerer:
    """AGI 基准答题器 - 使用 AGI 系统回答基准问题."""
    
    def __init__(self, agi_core=None, answer_func: Optional[callable] = None):
        self.answer_func = answer_func
        if not self.answer_func and agi_core is not None:
            self.answer_func = getattr(agi_core, "answer_multiple_choice", None)
        if not self.answer_func:
            raise ValueError("需要提供公开基准答题函数 (answer_func) 或 agi_core.answer_multiple_choice")
    
    def answer(self, question: str, choices: List[str]) -> int:
        """回答问题（禁止硬编码与启发式）。"""
        return int(self.answer_func(question, choices))


# ============================================================================
# 工厂函数
# ============================================================================

def create_benchmark_evaluator(public_only: bool = True) -> StandardBenchmarkEvaluator:
    """创建基准评估器."""
    return StandardBenchmarkEvaluator(public_only=public_only)


def create_agi_answerer(agi_core=None, answer_func: Optional[callable] = None) -> AGIBenchmarkAnswerer:
    """创建 AGI 答题器."""
    return AGIBenchmarkAnswerer(agi_core, answer_func=answer_func)


def run_standard_benchmarks(agi_core=None, 
                            n_per_benchmark: int = None,
                            answer_func: Optional[callable] = None,
                            public_only: bool = True) -> Dict[str, Any]:
    """运行标准基准测试（仅公开基准）."""
    env_min = os.getenv("H2Q_PUBLIC_BENCH_MIN")
    min_questions = int(env_min) if env_min and env_min.isdigit() else (100 if public_only else 20)
    env_n = os.getenv("H2Q_PUBLIC_BENCH_N")
    if n_per_benchmark is None:
        n_per_benchmark = int(env_n) if env_n and env_n.isdigit() else min_questions
    if n_per_benchmark < min_questions:
        if os.getenv("H2Q_ALLOW_SMALL_PUBLIC_BENCH") == "1":
            min_questions = n_per_benchmark
        else:
            raise ValueError(f"题量不足：需要至少 {min_questions} 题用于分析。")
    evaluator = create_benchmark_evaluator(public_only=public_only)
    answerer = create_agi_answerer(agi_core, answer_func=answer_func)
    
    results = evaluator.evaluate_all(
        answer_func=answerer.answer,
        n_per_benchmark=n_per_benchmark
    )
    
    report = evaluator.generate_report()
    
    return {
        "results": {k.value: {
            "accuracy": v.accuracy,
            "correct": v.correct_answers,
            "total": v.total_questions,
            "multi_select_accuracy": v.multi_select_accuracy,
        } for k, v in results.items()},
        "report": report,
    }


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H2Q 标准人类基准测试 - 演示")
    print("=" * 70)
    
    # 运行测试
    result = run_standard_benchmarks()
    
    # 打印报告
    print(result["report"])
