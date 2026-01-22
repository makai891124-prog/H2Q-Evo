"""H2Q AGI 核心模块.

实现学术级AGI核心能力:
- 神经符号推理 (Neuro-Symbolic Reasoning)
- 因果推理 (Causal Inference)
- 层次化规划 (Hierarchical Planning)
- 元学习 (Meta-Learning)
- 持续学习 (Continual Learning)

增强模块 (利用 H2Q 数学优势):
- 四元数增强元学习 (Quaternion-Enhanced Meta-Learning)
- 分形增强规划 (Fractal-Enhanced Planning)

参考AGI理论框架:
- Goertzel, "Artificial General Intelligence" (2007)
- Legg & Hutter, "Universal Intelligence" (2007)
- Lake et al., "Building Machines That Learn and Think Like People" (2017)
- H2Q 四元数-分形微分几何框架
"""

from .neuro_symbolic_reasoner import (
    Symbol,
    Predicate,
    Rule,
    SymbolicKnowledgeBase,
    NeuralEmbedder,
    NeuroSymbolicReasoner,
    create_neuro_symbolic_reasoner,
)

from .causal_inference import (
    CausalNode,
    CausalEdge,
    CausalGraph,
    StructuralCausalModel,
    CausalDiscovery,
    CausalInferenceEngine,
    create_causal_inference_engine,
)

from .hierarchical_planning import (
    State,
    Action,
    Task,
    Method,
    Plan,
    PlanningDomain,
    HTNPlanner,
    GoalDecomposer,
    DynamicReplanner,
    HierarchicalPlanningSystem,
    create_planning_system as create_hierarchical_planning_system,
)

from .meta_learning_core import (
    Task as MetaTask,
    MetaLearningConfig,
    SimpleNetwork,
    MAML,
    Reptile,
    FewShotLearner,
    FewShotResult,
    MetaLearningCore,
    create_meta_learning_core,
)

# H2Q 增强模块
from .quaternion_enhanced_meta import (
    quaternion_multiply,
    quaternion_normalize,
    quaternion_exp,
    quaternion_log,
    compute_fueter_residual,
    compute_berry_phase,
    QuaternionLinear,
    QuaternionEnhancedNetwork,
    QuaternionMAML,
    QMAMLConfig,
    QMetaTask,
    QMetaResult,
    QuaternionMetaLearningCore,
    create_quaternion_meta_learner,
    create_random_qmeta_task,
)

from .fractal_enhanced_planning import (
    fractal_decompose,
    fractal_combine,
    quaternion_state_encode,
    compute_quaternion_distance,
    compute_berry_heuristic,
    fueter_path_validity,
    FractalState,
    FractalAction,
    FractalTask,
    FractalMethod,
    FractalPlan,
    FractalPlanningDomain,
    FractalHTNPlanner,
    FractalGoalDecomposer,
    FractalDynamicReplanner,
    FractalHierarchicalPlanningSystem,
    create_fractal_planning_system,
)

from .continual_learning import (
    ContinualTask,
    ContinualConfig,
    EWC,
    ExperienceReplay,
    PackNet,
    ContinualLearningSystem,
    ContinualLearningResult,
    create_continual_learning_system,
)

# 多模态 AGI 核心
from .multimodal_agi_core import (
    AGIConfig,
    MultimodalAGICore,
    VisionEncoder,
    LanguageEncoder,
    MathReasoningModule,
    CrossModalFusion,
    ClassificationHead,
    MNISTLoader,
    MathDatasetGenerator,
    SimpleQADataset,
    create_multimodal_agi,
    load_mnist_dataset,
    generate_math_dataset,
    generate_qa_dataset,
)

# 人类标准考试系统
from .human_standard_exam import (
    ExamCategory,
    DifficultyLevel,
    ExamQuestion,
    ExamResult,
    QuestionBankGenerator,
    ExamScorer,
    HumanStandardExam,
    create_exam,
    run_quick_assessment,
)

# 分形记忆压缩
from .fractal_memory_compression import (
    CompressionLevel,
    MemoryBlock,
    FractalNode,
    QuaternionWavelet,
    FractalCompressor,
    FractalMemoryDatabase,
    create_fractal_memory_db,
)

# 知识获取
from .knowledge_acquisition import (
    ResourceSource,
    ResourceType,
    KnowledgeResource,
    AcquisitionConfig,
    ComplianceFilter,
    WikipediaFetcher,
    ArxivFetcher,
    MathProblemGenerator,
    KnowledgeAcquisitionManager,
    create_knowledge_acquisition_manager,
)

# 自主进化
from .autonomous_evolution import (
    EvolutionState,
    Interest,
    LearningEpisode,
    EvolutionConfig,
    InterestGenerator,
    AutonomousEvolutionEngine,
    create_evolution_engine,
)

# 标准人类基准
from .standard_benchmarks import (
    BenchmarkType,
    Difficulty,
    BenchmarkQuestion,
    BenchmarkResult,
    MMLUBenchmark,
    GSM8KBenchmark,
    ARCBenchmark,
    HellaSwagBenchmark,
    StandardBenchmarkEvaluator,
    AGIBenchmarkAnswerer,
    create_benchmark_evaluator,
    create_agi_answerer,
    run_standard_benchmarks,
)

__all__ = [
    # 神经符号推理
    "Symbol",
    "Predicate", 
    "Rule",
    "SymbolicKnowledgeBase",
    "NeuralEmbedder",
    "NeuroSymbolicReasoner",
    "create_neuro_symbolic_reasoner",
    
    # 因果推理
    "CausalNode",
    "CausalEdge",
    "CausalGraph",
    "StructuralCausalModel",
    "CausalDiscovery",
    "CausalInferenceEngine",
    "create_causal_inference_engine",
    
    # 层次化规划
    "State",
    "Action",
    "Task",
    "Method",
    "Plan",
    "PlanningDomain",
    "HTNPlanner",
    "GoalDecomposer",
    "DynamicReplanner",
    "HierarchicalPlanningSystem",
    "create_hierarchical_planning_system",
    
    # 元学习
    "MetaTask",
    "MetaLearningConfig",
    "SimpleNetwork",
    "MAML",
    "Reptile",
    "FewShotLearner",
    "FewShotResult",
    "MetaLearningCore",
    "create_meta_learning_core",
    
    # 持续学习
    "ContinualTask",
    "ContinualConfig",
    "EWC",
    "ExperienceReplay",
    "PackNet",
    "ContinualLearningSystem",
    "ContinualLearningResult",
    "create_continual_learning_system",
    
    # 多模态 AGI
    "AGIConfig",
    "MultimodalAGICore",
    "VisionEncoder",
    "LanguageEncoder",
    "MathReasoningModule",
    "CrossModalFusion",
    "ClassificationHead",
    "MNISTLoader",
    "MathDatasetGenerator",
    "SimpleQADataset",
    "create_multimodal_agi",
    "load_mnist_dataset",
    "generate_math_dataset",
    "generate_qa_dataset",
    
    # 人类标准考试
    "ExamCategory",
    "DifficultyLevel",
    "ExamQuestion",
    "ExamResult",
    "QuestionBankGenerator",
    "ExamScorer",
    "HumanStandardExam",
    "create_exam",
    "run_quick_assessment",
    
    # 分形记忆压缩
    "CompressionLevel",
    "MemoryBlock",
    "FractalNode",
    "QuaternionWavelet",
    "FractalCompressor",
    "FractalMemoryDatabase",
    "create_fractal_memory_db",
    
    # 知识获取
    "ResourceSource",
    "ResourceType",
    "KnowledgeResource",
    "AcquisitionConfig",
    "ComplianceFilter",
    "WikipediaFetcher",
    "ArxivFetcher",
    "MathProblemGenerator",
    "KnowledgeAcquisitionManager",
    "create_knowledge_acquisition_manager",
    
    # 自主进化
    "EvolutionState",
    "Interest",
    "LearningEpisode",
    "EvolutionConfig",
    "InterestGenerator",
    "AutonomousEvolutionEngine",
    "create_evolution_engine",
    
    # 标准人类基准
    "BenchmarkType",
    "Difficulty",
    "BenchmarkQuestion",
    "BenchmarkResult",
    "MMLUBenchmark",
    "GSM8KBenchmark",
    "ARCBenchmark",
    "HellaSwagBenchmark",
    "StandardBenchmarkEvaluator",
    "AGIBenchmarkAnswerer",
    "create_benchmark_evaluator",
    "create_agi_answerer",
    "run_standard_benchmarks",
    
    # 24小时自主进化
    "Evolution24HSystem",
    "SurvivalDaemon",
    
    # 中国网络源
    "ChinaKnowledgeAcquirer",
]

# 可选导入 - 24小时进化系统
try:
    from .evolution_24h import (
        EvolutionConfig as Evolution24HConfig,
        FractalCompressor as Evolution24HCompressor,
        KnowledgeAcquirer as Evolution24HKnowledgeAcquirer,
        CapabilityTester,
        Evolution24HSystem,
    )
except ImportError:
    pass

# 可选导入 - 生存守护进程
try:
    from .survival_daemon import (
        SurvivalConfig,
        SurvivalDaemon,
        ProcessState,
        HeartbeatRecord,
        create_survival_daemon,
    )
except ImportError:
    pass

# 可选导入 - 中国网络源
try:
    from .china_knowledge_source import (
        ChinaSourceConfig,
        HFMirrorDatasetLoader,
        BaiduBaikeAcquirer,
        ChinaOpenDatasets,
        ChinaKnowledgeAcquirer,
        test_china_network,
    )
except ImportError:
    pass
