"""H2Q 符号-神经融合推理引擎 (Neuro-Symbolic Reasoning Engine).

实现AGI核心能力：
1. 符号逻辑推理 (演绎、归纳、溯因)
2. 神经网络模式识别
3. 符号-神经双向转换
4. 可解释推理链生成

参考文献:
- Garcez et al., "Neural-Symbolic Learning and Reasoning" (2019)
- Mao et al., "The Neuro-Symbolic Concept Learner" (2019)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from collections import deque
from enum import Enum
import time
import re


# ============================================================================
# 符号系统基础
# ============================================================================

class LogicType(Enum):
    """逻辑类型."""
    DEDUCTIVE = "deductive"      # 演绎: A→B, A ⊢ B
    INDUCTIVE = "inductive"      # 归纳: 观察→规律
    ABDUCTIVE = "abductive"      # 溯因: B, A→B ⊢ A (最佳解释)


@dataclass
class Symbol:
    """符号表示."""
    name: str
    type: str  # concept, relation, variable, constant
    arity: int = 0  # 关系的元数
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name == other.name


@dataclass
class Predicate:
    """谓词逻辑表达式."""
    relation: Symbol
    arguments: List[Symbol]
    negated: bool = False
    confidence: float = 1.0
    
    def __str__(self):
        neg = "¬" if self.negated else ""
        args = ", ".join(a.name for a in self.arguments)
        return f"{neg}{self.relation.name}({args})"
    
    def __hash__(self):
        return hash(str(self))


@dataclass
class Rule:
    """推理规则: antecedents → consequent."""
    antecedents: List[Predicate]  # 前提
    consequent: Predicate         # 结论
    confidence: float = 1.0       # 规则置信度
    name: str = ""
    
    def __str__(self):
        ants = " ∧ ".join(str(a) for a in self.antecedents)
        return f"{ants} → {self.consequent}"


@dataclass
class ProofStep:
    """证明步骤."""
    step_id: int
    rule_applied: str
    premises: List[str]
    conclusion: str
    confidence: float


@dataclass
class ReasoningResult:
    """推理结果."""
    query: str
    answer: Any
    confidence: float
    logic_type: LogicType
    proof_chain: List[ProofStep]
    neural_embedding: np.ndarray
    latency_ms: float
    explanations: List[str]


# ============================================================================
# 符号知识库
# ============================================================================

class SymbolicKnowledgeBase:
    """符号知识库 - 存储事实和规则."""
    
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}
        self.facts: Set[Predicate] = set()
        self.rules: List[Rule] = []
        self.type_hierarchy: Dict[str, Set[str]] = {}  # 类型继承
        
        # 初始化基础类型
        self._init_base_types()
        self._init_common_rules()
    
    def _init_base_types(self):
        """初始化基础类型层次."""
        self.type_hierarchy = {
            "entity": {"object", "agent", "event"},
            "object": {"physical", "abstract"},
            "agent": {"human", "animal", "ai"},
            "event": {"action", "state_change"},
            "physical": {"solid", "liquid", "gas"},
        }
    
    def _init_common_rules(self):
        """初始化常识规则."""
        # 传递性规则
        is_a = Symbol("is_a", "relation", arity=2)
        X = Symbol("X", "variable")
        Y = Symbol("Y", "variable")
        Z = Symbol("Z", "variable")
        
        # is_a(X, Y) ∧ is_a(Y, Z) → is_a(X, Z)
        self.rules.append(Rule(
            antecedents=[
                Predicate(is_a, [X, Y]),
                Predicate(is_a, [Y, Z])
            ],
            consequent=Predicate(is_a, [X, Z]),
            confidence=1.0,
            name="transitivity_is_a"
        ))
        
        # 因果传递性
        causes = Symbol("causes", "relation", arity=2)
        # causes(X, Y) ∧ causes(Y, Z) → causes(X, Z)
        self.rules.append(Rule(
            antecedents=[
                Predicate(causes, [X, Y]),
                Predicate(causes, [Y, Z])
            ],
            consequent=Predicate(causes, [X, Z]),
            confidence=0.8,  # 因果链可能衰减
            name="transitivity_causes"
        ))
    
    def add_symbol(self, name: str, type: str, **attrs) -> Symbol:
        """添加符号."""
        if name not in self.symbols:
            self.symbols[name] = Symbol(name, type, attributes=attrs)
        return self.symbols[name]
    
    def add_fact(self, relation: str, *args: str, confidence: float = 1.0):
        """添加事实."""
        rel_sym = self.add_symbol(relation, "relation", arity=len(args))
        arg_syms = [self.add_symbol(a, "constant") for a in args]
        fact = Predicate(rel_sym, arg_syms, confidence=confidence)
        self.facts.add(fact)
        return fact
    
    def add_rule(self, rule: Rule):
        """添加规则."""
        self.rules.append(rule)
    
    def query_facts(self, relation: str, *args) -> List[Predicate]:
        """查询事实."""
        results = []
        for fact in self.facts:
            if fact.relation.name != relation:
                continue
            if len(args) == 0:
                results.append(fact)
                continue
            # 检查参数匹配
            match = True
            for i, arg in enumerate(args):
                if arg is not None and i < len(fact.arguments):
                    if fact.arguments[i].name != arg:
                        match = False
                        break
            if match:
                results.append(fact)
        return results


# ============================================================================
# 神经嵌入模块
# ============================================================================

class NeuralEmbedder:
    """神经嵌入器 - 符号到向量的映射."""
    
    def __init__(self, embed_dim: int = 64, seed: int = 42):
        np.random.seed(seed)
        self.embed_dim = embed_dim
        
        # 符号嵌入缓存
        self.symbol_embeddings: Dict[str, np.ndarray] = {}
        
        # 关系嵌入矩阵 (用于关系推理)
        self.relation_matrices: Dict[str, np.ndarray] = {}
        
        # 神经网络权重
        scale = np.sqrt(2.0 / embed_dim)
        self.W_compose = np.random.randn(embed_dim, embed_dim * 2).astype(np.float32) * scale
        self.W_project = np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale
    
    def embed_symbol(self, symbol: Symbol) -> np.ndarray:
        """获取或创建符号嵌入."""
        if symbol.name not in self.symbol_embeddings:
            # 基于名称哈希生成确定性嵌入
            hash_val = hash(symbol.name)
            np.random.seed(hash_val % (2**31))
            embed = np.random.randn(self.embed_dim).astype(np.float32)
            embed = embed / (np.linalg.norm(embed) + 1e-8)
            self.symbol_embeddings[symbol.name] = embed
        return self.symbol_embeddings[symbol.name]
    
    def embed_predicate(self, pred: Predicate) -> np.ndarray:
        """嵌入谓词表达式."""
        rel_embed = self.embed_symbol(pred.relation)
        
        if len(pred.arguments) == 0:
            return rel_embed
        
        # 聚合参数嵌入
        arg_embeds = [self.embed_symbol(a) for a in pred.arguments]
        arg_mean = np.mean(arg_embeds, axis=0)
        
        # 组合关系和参数
        combined = np.concatenate([rel_embed, arg_mean])
        composed = np.tanh(self.W_compose @ combined)
        
        if pred.negated:
            composed = -composed
        
        return composed * pred.confidence
    
    def embed_rule(self, rule: Rule) -> np.ndarray:
        """嵌入规则."""
        # 嵌入前提
        ant_embeds = [self.embed_predicate(p) for p in rule.antecedents]
        ant_mean = np.mean(ant_embeds, axis=0) if ant_embeds else np.zeros(self.embed_dim)
        
        # 嵌入结论
        cons_embed = self.embed_predicate(rule.consequent)
        
        # 规则嵌入 = 前提到结论的变换
        rule_embed = np.tanh(self.W_project @ (cons_embed - ant_mean))
        return rule_embed * rule.confidence
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================================
# 推理引擎
# ============================================================================

class NeuroSymbolicReasoner:
    """神经符号融合推理器."""
    
    def __init__(self, embed_dim: int = 64):
        self.kb = SymbolicKnowledgeBase()
        self.embedder = NeuralEmbedder(embed_dim)
        
        # 推理缓存
        self.inference_cache: Dict[str, ReasoningResult] = {}
        
        # 统计
        self.total_inferences = 0
        self.cache_hits = 0
    
    def add_knowledge(self, relation: str, *args, confidence: float = 1.0):
        """添加知识."""
        self.kb.add_fact(relation, *args, confidence=confidence)
    
    def add_rule(self, name: str, antecedents: List[Tuple], consequent: Tuple, 
                 confidence: float = 1.0):
        """添加规则.
        
        Args:
            name: 规则名称
            antecedents: 前提列表 [(relation, arg1, arg2, ...), ...]
            consequent: 结论 (relation, arg1, arg2, ...)
            confidence: 置信度
        """
        ant_preds = []
        for ant in antecedents:
            rel = self.kb.add_symbol(ant[0], "relation", arity=len(ant)-1)
            args = [self.kb.add_symbol(a, "variable" if a[0].isupper() else "constant") 
                    for a in ant[1:]]
            ant_preds.append(Predicate(rel, args))
        
        cons_rel = self.kb.add_symbol(consequent[0], "relation", arity=len(consequent)-1)
        cons_args = [self.kb.add_symbol(a, "variable" if a[0].isupper() else "constant") 
                     for a in consequent[1:]]
        cons_pred = Predicate(cons_rel, cons_args)
        
        rule = Rule(ant_preds, cons_pred, confidence, name)
        self.kb.add_rule(rule)
    
    def reason(self, query: str, max_depth: int = 5) -> ReasoningResult:
        """执行推理."""
        start_time = time.perf_counter()
        self.total_inferences += 1
        
        # 检查缓存
        if query in self.inference_cache:
            self.cache_hits += 1
            cached = self.inference_cache[query]
            return cached
        
        # 解析查询
        parsed = self._parse_query(query)
        
        # 根据查询类型选择推理策略
        if parsed["type"] == "fact_check":
            result = self._deductive_reasoning(parsed, max_depth)
        elif parsed["type"] == "find_cause":
            result = self._abductive_reasoning(parsed, max_depth)
        elif parsed["type"] == "generalize":
            result = self._inductive_reasoning(parsed, max_depth)
        else:
            # 混合推理
            result = self._hybrid_reasoning(parsed, max_depth)
        
        result.latency_ms = (time.perf_counter() - start_time) * 1000
        
        # 缓存结果
        self.inference_cache[query] = result
        
        return result
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """解析查询."""
        query_lower = query.lower()
        
        # 检测查询类型
        if any(w in query_lower for w in ["is", "does", "can", "是否", "能否"]):
            q_type = "fact_check"
        elif any(w in query_lower for w in ["why", "cause", "because", "为什么", "原因"]):
            q_type = "find_cause"
        elif any(w in query_lower for w in ["all", "every", "generally", "所有", "通常"]):
            q_type = "generalize"
        else:
            q_type = "general"
        
        # 提取关系和实体 (简化实现)
        entities = re.findall(r'\b([A-Z][a-z]+)\b', query)
        
        return {
            "type": q_type,
            "raw": query,
            "entities": entities,
            "embedding": self._embed_query(query)
        }
    
    def _embed_query(self, query: str) -> np.ndarray:
        """嵌入查询文本."""
        # 简化: 基于词哈希
        words = query.lower().split()
        if not words:
            return np.zeros(self.embedder.embed_dim, dtype=np.float32)
        
        embeddings = []
        for word in words:
            sym = Symbol(word, "word")
            embeddings.append(self.embedder.embed_symbol(sym))
        
        return np.mean(embeddings, axis=0)
    
    def _deductive_reasoning(self, parsed: Dict, max_depth: int) -> ReasoningResult:
        """演绎推理: 从规则和事实推导结论."""
        proof_chain = []
        step_id = 0
        confidence = 1.0
        
        # 前向链推理
        inferred_facts = set(self.kb.facts)
        
        for depth in range(max_depth):
            new_facts = set()
            
            for rule in self.kb.rules:
                # 尝试应用规则
                bindings = self._match_rule(rule, inferred_facts)
                
                for binding in bindings:
                    # 实例化结论
                    new_pred = self._apply_binding(rule.consequent, binding)
                    
                    if new_pred not in inferred_facts and new_pred not in new_facts:
                        new_facts.add(new_pred)
                        step_id += 1
                        proof_chain.append(ProofStep(
                            step_id=step_id,
                            rule_applied=rule.name,
                            premises=[str(self._apply_binding(a, binding)) 
                                     for a in rule.antecedents],
                            conclusion=str(new_pred),
                            confidence=rule.confidence
                        ))
                        confidence *= rule.confidence
            
            if not new_facts:
                break
            
            inferred_facts.update(new_facts)
        
        # 检查是否回答了查询
        answer = self._check_query_answer(parsed, inferred_facts)
        
        return ReasoningResult(
            query=parsed["raw"],
            answer=answer,
            confidence=min(confidence, 0.99),
            logic_type=LogicType.DEDUCTIVE,
            proof_chain=proof_chain,
            neural_embedding=parsed["embedding"],
            latency_ms=0,
            explanations=self._generate_explanations(proof_chain)
        )
    
    def _abductive_reasoning(self, parsed: Dict, max_depth: int) -> ReasoningResult:
        """溯因推理: 寻找最佳解释."""
        proof_chain = []
        explanations = []
        
        # 查找可能的原因
        query_embed = parsed["embedding"]
        candidate_causes = []
        
        for fact in self.kb.facts:
            fact_embed = self.embedder.embed_predicate(fact)
            sim = self.embedder.similarity(query_embed, fact_embed)
            if sim > 0.3:
                candidate_causes.append((fact, sim))
        
        # 按相似度排序
        candidate_causes.sort(key=lambda x: x[1], reverse=True)
        
        # 生成解释
        for i, (cause, sim) in enumerate(candidate_causes[:3]):
            step = ProofStep(
                step_id=i + 1,
                rule_applied="abductive_hypothesis",
                premises=[str(cause)],
                conclusion=f"可能解释: {cause}",
                confidence=sim
            )
            proof_chain.append(step)
            explanations.append(f"假说 {i+1}: {cause} (置信度: {sim:.2f})")
        
        best_answer = candidate_causes[0] if candidate_causes else None
        confidence = best_answer[1] if best_answer else 0.0
        
        return ReasoningResult(
            query=parsed["raw"],
            answer=str(best_answer[0]) if best_answer else "无法找到解释",
            confidence=confidence,
            logic_type=LogicType.ABDUCTIVE,
            proof_chain=proof_chain,
            neural_embedding=parsed["embedding"],
            latency_ms=0,
            explanations=explanations
        )
    
    def _inductive_reasoning(self, parsed: Dict, max_depth: int) -> ReasoningResult:
        """归纳推理: 从实例归纳规律."""
        proof_chain = []
        
        # 收集相关事实
        entities = parsed.get("entities", [])
        related_facts = []
        
        for fact in self.kb.facts:
            for arg in fact.arguments:
                if arg.name in entities or not entities:
                    related_facts.append(fact)
                    break
        
        # 寻找模式
        relation_counts: Dict[str, int] = {}
        for fact in related_facts:
            rel = fact.relation.name
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        # 生成归纳结论
        patterns = []
        for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
            if count >= 2:  # 至少2个实例
                confidence = min(count / 10.0, 0.9)  # 基于实例数
                patterns.append((rel, count, confidence))
                
                proof_chain.append(ProofStep(
                    step_id=len(proof_chain) + 1,
                    rule_applied="inductive_generalization",
                    premises=[f"{count} instances of {rel}"],
                    conclusion=f"Pattern: {rel} is common",
                    confidence=confidence
                ))
        
        if patterns:
            answer = f"归纳规律: {patterns[0][0]} (基于 {patterns[0][1]} 个实例)"
            confidence = patterns[0][2]
        else:
            answer = "样本不足，无法归纳"
            confidence = 0.0
        
        return ReasoningResult(
            query=parsed["raw"],
            answer=answer,
            confidence=confidence,
            logic_type=LogicType.INDUCTIVE,
            proof_chain=proof_chain,
            neural_embedding=parsed["embedding"],
            latency_ms=0,
            explanations=[f"发现 {len(patterns)} 个模式"]
        )
    
    def _hybrid_reasoning(self, parsed: Dict, max_depth: int) -> ReasoningResult:
        """混合推理: 结合多种策略."""
        # 并行执行三种推理
        deductive = self._deductive_reasoning(parsed, max_depth)
        abductive = self._abductive_reasoning(parsed, max_depth)
        inductive = self._inductive_reasoning(parsed, max_depth)
        
        # 选择最高置信度的结果
        results = [deductive, abductive, inductive]
        best = max(results, key=lambda r: r.confidence)
        
        # 综合解释
        all_explanations = []
        all_explanations.extend([f"[演绎] {e}" for e in deductive.explanations])
        all_explanations.extend([f"[溯因] {e}" for e in abductive.explanations])
        all_explanations.extend([f"[归纳] {e}" for e in inductive.explanations])
        
        return ReasoningResult(
            query=parsed["raw"],
            answer=best.answer,
            confidence=best.confidence,
            logic_type=best.logic_type,
            proof_chain=best.proof_chain,
            neural_embedding=parsed["embedding"],
            latency_ms=0,
            explanations=all_explanations
        )
    
    def _match_rule(self, rule: Rule, facts: Set[Predicate]) -> List[Dict[str, Symbol]]:
        """匹配规则，返回所有可能的变量绑定."""
        bindings = [{}]
        
        for ant in rule.antecedents:
            new_bindings = []
            for binding in bindings:
                for fact in facts:
                    if fact.relation.name != ant.relation.name:
                        continue
                    if len(fact.arguments) != len(ant.arguments):
                        continue
                    
                    # 尝试扩展绑定
                    new_binding = binding.copy()
                    match = True
                    
                    for ant_arg, fact_arg in zip(ant.arguments, fact.arguments):
                        if ant_arg.type == "variable":
                            if ant_arg.name in new_binding:
                                if new_binding[ant_arg.name].name != fact_arg.name:
                                    match = False
                                    break
                            else:
                                new_binding[ant_arg.name] = fact_arg
                        elif ant_arg.name != fact_arg.name:
                            match = False
                            break
                    
                    if match:
                        new_bindings.append(new_binding)
            
            bindings = new_bindings
            if not bindings:
                break
        
        return bindings
    
    def _apply_binding(self, pred: Predicate, binding: Dict[str, Symbol]) -> Predicate:
        """应用变量绑定到谓词."""
        new_args = []
        for arg in pred.arguments:
            if arg.type == "variable" and arg.name in binding:
                new_args.append(binding[arg.name])
            else:
                new_args.append(arg)
        
        return Predicate(pred.relation, new_args, pred.negated, pred.confidence)
    
    def _check_query_answer(self, parsed: Dict, facts: Set[Predicate]) -> str:
        """检查是否回答了查询."""
        query_embed = parsed["embedding"]
        
        best_match = None
        best_sim = 0.0
        
        for fact in facts:
            fact_embed = self.embedder.embed_predicate(fact)
            sim = self.embedder.similarity(query_embed, fact_embed)
            if sim > best_sim:
                best_sim = sim
                best_match = fact
        
        if best_match and best_sim > 0.4:
            return f"是 ({best_match})"
        else:
            return "无法确定"
    
    def _generate_explanations(self, proof_chain: List[ProofStep]) -> List[str]:
        """生成人类可读的解释."""
        explanations = []
        for step in proof_chain:
            exp = f"步骤 {step.step_id}: 应用规则 '{step.rule_applied}'"
            exp += f" 从 {step.premises} 得到 {step.conclusion}"
            exp += f" (置信度: {step.confidence:.2f})"
            explanations.append(exp)
        return explanations
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息."""
        return {
            "total_inferences": self.total_inferences,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_inferences),
            "knowledge_base": {
                "symbols": len(self.kb.symbols),
                "facts": len(self.kb.facts),
                "rules": len(self.kb.rules)
            }
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_neuro_symbolic_reasoner(embed_dim: int = 64) -> NeuroSymbolicReasoner:
    """创建神经符号推理器."""
    reasoner = NeuroSymbolicReasoner(embed_dim)
    
    # 添加一些常识知识
    # 类型层次
    reasoner.add_knowledge("is_a", "dog", "animal")
    reasoner.add_knowledge("is_a", "cat", "animal")
    reasoner.add_knowledge("is_a", "animal", "living_thing")
    reasoner.add_knowledge("is_a", "human", "animal")
    
    # 属性
    reasoner.add_knowledge("has_property", "dog", "loyal")
    reasoner.add_knowledge("has_property", "cat", "independent")
    reasoner.add_knowledge("has_property", "animal", "needs_food")
    
    # 因果关系
    reasoner.add_knowledge("causes", "hunger", "eating")
    reasoner.add_knowledge("causes", "eating", "satisfaction")
    reasoner.add_knowledge("causes", "rain", "wet_ground")
    
    return reasoner


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("H2Q 符号-神经融合推理引擎 - 演示")
    print("=" * 70)
    
    reasoner = create_neuro_symbolic_reasoner()
    
    # 测试查询
    queries = [
        "Is dog an animal?",
        "Is dog a living_thing?",
        "Why is the ground wet?",
        "What causes satisfaction?",
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        print("-" * 50)
        
        result = reasoner.reason(query)
        
        print(f"答案: {result.answer}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"推理类型: {result.logic_type.value}")
        print(f"延迟: {result.latency_ms:.2f} ms")
        
        if result.explanations:
            print("解释:")
            for exp in result.explanations[:3]:
                print(f"  - {exp}")
    
    print("\n" + "=" * 70)
    print("统计信息:")
    stats = reasoner.get_stats()
    print(f"  总推理次数: {stats['total_inferences']}")
    print(f"  知识库符号: {stats['knowledge_base']['symbols']}")
    print(f"  知识库事实: {stats['knowledge_base']['facts']}")
    print(f"  知识库规则: {stats['knowledge_base']['rules']}")
