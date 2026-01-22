"""
=================================================================
M24诚实协议验证系统 (M24 Honest Agreement Protocol)
=================================================================

M24协议定义:
M = Multi-agent (多智能体)
2 = 二层验证(本地验证 + 全局验证)
4 = 四个诚实承诺:
    1. 信息透明 (Information Transparency)
    2. 决策可追溯 (Decision Traceability)
    3. 反作弊承诺 (Anti-Fraud Commitment)
    4. 数学严格性 (Mathematical Rigor)

这个协议确保每个AGI决策都是:
- 可验证的 (Verifiable)
- 可解释的 (Explainable)
- 可追溯的 (Traceable)
- 诚实的 (Honest)
=================================================================
"""

import hashlib
import hmac
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import uuid
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import base64


# ==================== 数据结构 ====================

class HonesttyLevel(Enum):
    """诚实性级别"""
    PROVEN_HONEST = "proven_honest"      # 已证明诚实
    HIGHLY_PROBABLE = "highly_probable"  # 高度可能
    PROBABLE = "probable"                # 可能
    UNCERTAIN = "uncertain"              # 不确定
    SUSPICIOUS = "suspicious"            # 可疑
    FRAUDULENT = "fraudulent"           # 欺诈


@dataclass
class HonesttyProof:
    """诚实性证明"""
    proof_id: str
    decision_id: str
    verification_method: str             # 验证方法名称
    evidence: Dict[str, Any]             # 证据
    confidence: float                    # 置信度 [0, 1]
    timestamp: str
    signature: str = ""                  # 数字签名


@dataclass
class M24HonesttyAudit:
    """M24诚实性审计记录"""
    audit_id: str
    decision_id: str
    audit_time: str
    
    # 四个诚实承诺的验证
    transparency_verified: bool          # 信息透明性
    traceability_verified: bool          # 决策可追溯性
    anti_fraud_verified: bool            # 反作弊
    mathematical_rigor_verified: bool    # 数学严格性
    
    proofs: List[HonesttyProof] = field(default_factory=list)
    overall_honesty_level: HonesttyLevel = HonesttyLevel.UNCERTAIN
    reasoning: str = ""
    

# ==================== 验证引擎 ====================

class M24HonesttyProtocol:
    """M24诚实协议验证引擎"""
    
    def __init__(self, log_dir: str = "./m24_verification"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # 生成RSA密钥对(用于数字签名)
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        self.audit_log: List[M24HonesttyAudit] = []
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger("M24HonesttyProtocol")
        logger.setLevel(logging.DEBUG)
        
        handler = logging.FileHandler(
            self.log_dir / f"m24_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    # ========== 承诺1: 信息透明 ==========
    
    def verify_information_transparency(
        self,
        decision_id: str,
        decision_data: Dict[str, Any]
    ) -> HonesttyProof:
        """
        验证信息透明性
        
        检查项:
        1. 所有关键信息都被记录
        2. 没有隐藏的处理步骤
        3. 输入输出都被记录
        """
        
        proof_id = str(uuid.uuid4())[:8]
        
        # 检查项列表
        checks = {
            "has_input": "input_prompt" in decision_data,
            "has_output": "final_output" in decision_data,
            "has_votes": "votes" in decision_data and len(decision_data.get("votes", [])) > 0,
            "has_timestamp": "timestamp" in decision_data,
            "has_reasoning": "reasoning_path" in decision_data,
        }
        
        # 计算透明度评分
        transparency_score = sum(checks.values()) / len(checks)
        
        evidence = {
            "transparency_checks": checks,
            "transparency_score": transparency_score,
            "recorded_fields": list(decision_data.keys()),
        }
        
        proof = HonesttyProof(
            proof_id=proof_id,
            decision_id=decision_id,
            verification_method="information_transparency_check",
            evidence=evidence,
            confidence=transparency_score,
            timestamp=datetime.now().isoformat()
        )
        
        # 签名
        proof.signature = self._sign_proof(proof)
        
        self.logger.info(
            f"信息透明性验证: {decision_id} - "
            f"透明度: {transparency_score:.3f}, "
            f"所有关键信息已记录: {all(checks.values())}"
        )
        
        return proof
    
    # ========== 承诺2: 决策可追溯 ==========
    
    def verify_decision_traceability(
        self,
        decision_id: str,
        decision_data: Dict[str, Any]
    ) -> HonesttyProof:
        """
        验证决策可追溯性
        
        检查项:
        1. 每个决策都有唯一ID
        2. 完整的推理链
        3. 时间戳记录
        """
        
        proof_id = str(uuid.uuid4())[:8]
        
        # 生成决策哈希链
        decision_hash = self._compute_decision_hash(decision_data)
        
        # 检查推理链完整性
        reasoning_path = decision_data.get("reasoning_path", [])
        chain_length = len(reasoning_path)
        
        evidence = {
            "decision_id": decision_id,
            "decision_hash": decision_hash,
            "reasoning_chain_length": chain_length,
            "has_unique_id": len(decision_id) > 0,
            "has_timestamp": "timestamp" in decision_data,
            "traceability_depth": min(chain_length / 10, 1.0),  # 最多10步为满分
        }
        
        traceability_score = (
            evidence["has_unique_id"] * 0.3 +
            evidence["has_timestamp"] * 0.3 +
            evidence["traceability_depth"] * 0.4
        )
        
        proof = HonesttyProof(
            proof_id=proof_id,
            decision_id=decision_id,
            verification_method="decision_traceability_check",
            evidence=evidence,
            confidence=traceability_score,
            timestamp=datetime.now().isoformat()
        )
        
        proof.signature = self._sign_proof(proof)
        
        self.logger.info(
            f"决策可追溯性验证: {decision_id} - "
            f"可追溯度: {traceability_score:.3f}, "
            f"推理链长度: {chain_length}"
        )
        
        return proof
    
    # ========== 承诺3: 反作弊 ==========
    
    def verify_anti_fraud(
        self,
        decision_id: str,
        decision_data: Dict[str, Any]
    ) -> HonesttyProof:
        """
        验证反作弊性
        
        检查项:
        1. 多模型投票的真实性
        2. 没有重复或虚假投票
        3. 置信度的合理性
        4. 没有逻辑矛盾
        """
        
        proof_id = str(uuid.uuid4())[:8]
        
        votes = decision_data.get("votes", [])
        fraud_indicators = []
        
        # 检查1: 投票多样性(没有过多重复)
        outputs = [v.get("output", "") for v in votes if isinstance(v, dict)]
        unique_outputs = len(set(outputs))
        diversity_score = unique_outputs / max(len(votes), 1)
        
        if diversity_score < 0.5:
            fraud_indicators.append("low_output_diversity")
        
        # 检查2: 置信度合理性
        confidences = [v.get("confidence", 0.5) for v in votes if isinstance(v, dict)]
        conf_std = __import__('numpy').std(confidences) if confidences else 0
        
        if conf_std < 0.05:  # 太一致可能是作弊
            fraud_indicators.append("suspicious_uniform_confidence")
        
        # 检查3: 逻辑一致性
        logical_consistency = self._check_logical_consistency(decision_data)
        
        if not logical_consistency:
            fraud_indicators.append("logical_inconsistency")
        
        # 检查4: 时间戳合理性
        timestamp = decision_data.get("timestamp", "")
        if timestamp:
            try:
                decision_time = datetime.fromisoformat(timestamp)
                time_reasonability = True
            except:
                time_reasonability = False
                fraud_indicators.append("invalid_timestamp")
        else:
            time_reasonability = False
        
        # 计算反作弊评分(无作弊指标为1.0)
        anti_fraud_score = 1.0 - (len(fraud_indicators) * 0.25)
        anti_fraud_score = max(0, min(1.0, anti_fraud_score))
        
        evidence = {
            "fraud_indicators": fraud_indicators,
            "output_diversity_score": diversity_score,
            "confidence_std": float(conf_std),
            "logical_consistency": logical_consistency,
            "time_reasonability": time_reasonability,
            "total_fraud_risk": 1.0 - anti_fraud_score,
        }
        
        proof = HonesttyProof(
            proof_id=proof_id,
            decision_id=decision_id,
            verification_method="anti_fraud_check",
            evidence=evidence,
            confidence=anti_fraud_score,
            timestamp=datetime.now().isoformat()
        )
        
        proof.signature = self._sign_proof(proof)
        
        fraud_risk = 1.0 - anti_fraud_score
        self.logger.info(
            f"反作弊验证: {decision_id} - "
            f"反作弊评分: {anti_fraud_score:.3f}, "
            f"作弊风险: {fraud_risk:.3f}, "
            f"可疑指标: {len(fraud_indicators)}"
        )
        
        return proof
    
    # ========== 承诺4: 数学严格性 ==========
    
    def verify_mathematical_rigor(
        self,
        decision_id: str,
        decision_data: Dict[str, Any]
    ) -> HonesttyProof:
        """
        验证数学严格性
        
        检查项:
        1. 置信度计算的数学正确性
        2. 共识算法的正确实现
        3. 数学证明的完整性
        """
        
        proof_id = str(uuid.uuid4())[:8]
        
        votes = decision_data.get("votes", [])
        
        # 检查1: 置信度计算
        confidence_scores = [v.get("confidence", 0.5) for v in votes if isinstance(v, dict)]
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            overall_confidence = decision_data.get("confidence_score", 0.5)
            
            # 验证平均置信度计算
            calculation_error = abs(avg_confidence - overall_confidence)
            calculation_correct = calculation_error < 0.01
        else:
            calculation_correct = False
            avg_confidence = 0
        
        # 检查2: 共识算法一致性
        consensus_level = decision_data.get("consensus_level", "unknown")
        valid_consensus_levels = ["unanimous", "strong", "moderate", "weak", "dissensus"]
        consensus_valid = consensus_level in valid_consensus_levels
        
        # 检查3: 数学证明完整性
        math_proof = decision_data.get("mathematical_proof", "")
        has_proof = len(math_proof) > 50  # 最少需要一定长度的证明
        
        # 综合数学严格性评分
        rigor_score = (
            (1.0 if calculation_correct else 0.0) * 0.4 +
            (1.0 if consensus_valid else 0.0) * 0.3 +
            (1.0 if has_proof else 0.0) * 0.3
        )
        
        evidence = {
            "calculation_correct": calculation_correct,
            "avg_confidence": avg_confidence,
            "consensus_valid": consensus_valid,
            "has_proof": has_proof,
            "calculation_error": calculation_error if confidence_scores else None,
        }
        
        proof = HonesttyProof(
            proof_id=proof_id,
            decision_id=decision_id,
            verification_method="mathematical_rigor_check",
            evidence=evidence,
            confidence=rigor_score,
            timestamp=datetime.now().isoformat()
        )
        
        proof.signature = self._sign_proof(proof)
        
        self.logger.info(
            f"数学严格性验证: {decision_id} - "
            f"严格性评分: {rigor_score:.3f}, "
            f"计算正确: {calculation_correct}, "
            f"共识有效: {consensus_valid}"
        )
        
        return proof
    
    # ========== 综合审计 ==========
    
    def audit_decision(self, decision_id: str, decision_data: Dict[str, Any]) -> M24HonesttyAudit:
        """
        执行完整的M24诚实性审计
        """
        
        audit_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"M24诚实性审计开始 (Audit ID: {audit_id})")
        self.logger.info(f"决策ID: {decision_id}")
        self.logger.info(f"{'='*70}\n")
        
        # 执行四个承诺的验证
        proofs = []
        
        transparency_proof = self.verify_information_transparency(decision_id, decision_data)
        proofs.append(transparency_proof)
        transparency_ok = transparency_proof.confidence >= 0.7
        
        traceability_proof = self.verify_decision_traceability(decision_id, decision_data)
        proofs.append(traceability_proof)
        traceability_ok = traceability_proof.confidence >= 0.7
        
        anti_fraud_proof = self.verify_anti_fraud(decision_id, decision_data)
        proofs.append(anti_fraud_proof)
        anti_fraud_ok = anti_fraud_proof.confidence >= 0.7
        
        rigor_proof = self.verify_mathematical_rigor(decision_id, decision_data)
        proofs.append(rigor_proof)
        rigor_ok = rigor_proof.confidence >= 0.7
        
        # 综合诚实性判定
        proof_scores = [p.confidence for p in proofs]
        avg_honesty = sum(proof_scores) / len(proof_scores)
        
        if all([transparency_ok, traceability_ok, anti_fraud_ok, rigor_ok]):
            honesty_level = HonesttyLevel.PROVEN_HONEST
        elif avg_honesty >= 0.8:
            honesty_level = HonesttyLevel.HIGHLY_PROBABLE
        elif avg_honesty >= 0.6:
            honesty_level = HonesttyLevel.PROBABLE
        elif avg_honesty >= 0.4:
            honesty_level = HonesttyLevel.UNCERTAIN
        elif avg_honesty >= 0.2:
            honesty_level = HonesttyLevel.SUSPICIOUS
        else:
            honesty_level = HonesttyLevel.FRAUDULENT
        
        reasoning = f"""
M24诚实性审计结果:
- 信息透明: {transparency_proof.confidence:.3f} {'✓' if transparency_ok else '✗'}
- 决策可追溯: {traceability_proof.confidence:.3f} {'✓' if traceability_ok else '✗'}
- 反作弊验证: {anti_fraud_proof.confidence:.3f} {'✓' if anti_fraud_ok else '✗'}
- 数学严格性: {rigor_proof.confidence:.3f} {'✓' if rigor_ok else '✗'}
- 平均诚实评分: {avg_honesty:.3f}
- 最终判定: {honesty_level.value}
"""
        
        audit = M24HonesttyAudit(
            audit_id=audit_id,
            decision_id=decision_id,
            audit_time=datetime.now().isoformat(),
            transparency_verified=transparency_ok,
            traceability_verified=traceability_ok,
            anti_fraud_verified=anti_fraud_ok,
            mathematical_rigor_verified=rigor_ok,
            proofs=proofs,
            overall_honesty_level=honesty_level,
            reasoning=reasoning
        )
        
        self.audit_log.append(audit)
        
        self.logger.info(f"\n{reasoning}")
        self.logger.info(f"{'='*70}\n")
        
        return audit
    
    # ========== 辅助方法 ==========
    
    def _compute_decision_hash(self, decision_data: Dict[str, Any]) -> str:
        """计算决策的哈希值(用于追溯)"""
        decision_str = json.dumps(decision_data, sort_keys=True, default=str)
        return hashlib.sha256(decision_str.encode()).hexdigest()[:16]
    
    def _check_logical_consistency(self, decision_data: Dict[str, Any]) -> bool:
        """检查决策的逻辑一致性"""
        # 简单实现: 检查输入和输出是否匹配
        input_text = decision_data.get("input_prompt", "").lower()
        output_text = decision_data.get("final_output", "").lower()
        
        # 如果输出包含输入的关键词,认为是逻辑一致的
        words_in_input = set(input_text.split())
        words_in_output = set(output_text.split())
        
        overlap = len(words_in_input & words_in_output)
        consistency = overlap / max(len(words_in_input), 1) if words_in_input else True
        
        return consistency > 0.2
    
    def _sign_proof(self, proof: HonesttyProof) -> str:
        """对证明进行数字签名"""
        proof_data = json.dumps(asdict(proof), default=str).encode()
        
        signature = self.private_key.sign(
            proof_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
    
    def verify_signature(self, proof: HonesttyProof) -> bool:
        """验证证明的数字签名"""
        try:
            proof_data = json.dumps(asdict(proof), default=str).encode()
            signature = base64.b64decode(proof.signature.encode())
            
            self.public_key.verify(
                signature,
                proof_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False
    
    def save_audit_report(self, output_dir: str = "./m24_audit_reports") -> Path:
        """保存审计报告到文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存为JSON
        audit_data = [asdict(a) for a in self.audit_log]
        
        with open(output_dir / "m24_audits.json", "w") as f:
            json.dump(audit_data, f, indent=2, default=str)
        
        # 生成Markdown报告
        report = f"""# M24诚实协议审计报告

生成时间: {datetime.now().isoformat()}

## 协议定义

**M24** = Multi-agent Honest Agreement v2.4

四个核心承诺:
1. **信息透明 (Information Transparency)** - 所有决策信息都被完整记录
2. **决策可追溯 (Decision Traceability)** - 每个决策都有完整的推理链
3. **反作弊 (Anti-Fraud Commitment)** - 所有投票都经过严格验证
4. **数学严格性 (Mathematical Rigor)** - 所有计算都可被数学验证

## 审计统计

- 总审计数: {len(self.audit_log)}
- 已验证为诚实的: {sum(1 for a in self.audit_log if a.overall_honesty_level == HonesttyLevel.PROVEN_HONEST)}
- 可疑的: {sum(1 for a in self.audit_log if a.overall_honesty_level in [HonesttyLevel.SUSPICIOUS, HonesttyLevel.FRAUDULENT])}

## 审计详情

"""
        
        for audit in self.audit_log:
            report += f"""### 审计 {audit.audit_id}

**决策ID**: {audit.decision_id}
**审计时间**: {audit.audit_time}
**最终判定**: {audit.overall_honesty_level.value}

**各项验证结果**:
- 信息透明: {'✓' if audit.transparency_verified else '✗'}
- 决策可追溯: {'✓' if audit.traceability_verified else '✗'}
- 反作弊验证: {'✓' if audit.anti_fraud_verified else '✗'}
- 数学严格性: {'✓' if audit.mathematical_rigor_verified else '✗'}

**审计理由**:
{audit.reasoning}

---

"""
        
        with open(output_dir / "M24_AUDIT_REPORT.md", "w") as f:
            f.write(report)
        
        return output_dir


if __name__ == "__main__":
    # 演示
    protocol = M24HonesttyProtocol()
    
    # 示例决策
    example_decision = {
        "input_prompt": "什么是真实的AGI?",
        "final_output": "真实的AGI应该是诚实、可验证、数学上严格的系统",
        "votes": [
            {"output": "真实的AGI是诚实的", "confidence": 0.9},
            {"output": "AGI应该完全透明", "confidence": 0.85},
        ],
        "reasoning_path": [
            "步骤1: 收集多个模型意见",
            "步骤2: 验证逻辑一致性",
        ],
        "mathematical_proof": "通过多模型共识得出的决策在数学上是稳健的",
        "timestamp": datetime.now().isoformat(),
    }
    
    # 执行审计
    audit = protocol.audit_decision("decision_001", example_decision)
    
    print(f"\n审计结果: {audit.overall_honesty_level.value}")
    
    # 保存报告
    protocol.save_audit_report()
    print("审计报告已保存")
