# H2Q 项目全景审计报告

## 1. 文件目录树 (File Structure)
📂 **h2q_project/**
    📄 `train_zero_memory.py`
    📄 `train_spacetime.py`
    📄 `train_spacetime_vision.py`
    📄 `train_multilingual_decoder.py`
    📄 `train_fdc.py`
    📄 `test_vision_demo.py`
    📄 `train_byte_compression.py`
    📄 `train_tpq_optim.py`
    📄 `demo_interactive.py`
    📄 `train_decoder.py`
    📄 `train_fdc_pure.py`
    📄 `verify_multimodal_cpu.py`
    📄 `hello_human.py`
    📄 `train_multilingual.py`
    📄 `train_synesthesia_unified.py`
    📄 `train_hierarchy.py`
    📄 `demo_neural_zip.py`
    📄 `__init__.py`
    📄 `train_fractal.py`
    📄 `train_discrete_cpu.py`
    📄 `accelerate_gpt2.py`
    📄 `train_synesthesia_avtg.py`
    📄 `test_multilingual.py`
    📄 `train_vision_core.py`
    📄 `train_multimodal.py`
    📄 `train_h2q.py`
    📄 `code_analyzer.py`
    📄 `train_compression_test.py`
    📄 `train_distillation.py`
    📄 `train_full_stack_v2.py`
    📄 `deploy_reasoning_vault.py`
    📄 `train_manual_reversible.py`
    📄 `train_knot.py`
    📄 `train_omniscience.py`
    📄 `run_experiment.py`
    📄 `benchmark_latency.py`
    📄 `train_reversible_vision.py`
    📄 `h2q_server.py`
    📄 `train_arithmetic.py`
    📄 `run_language_simulation.py`
    📄 `virtual_giant_test.py`
    📄 `demo_universal_zip.py`
    📂 **tools/**
        📄 `spacetime_loader.py`
        📄 `byte_loader.py`
        📄 `prism_converter.py`
        📄 `data_loader.py`
        📄 `__init__.py`
        📄 `extract_qwen_crystal.py`
        📄 `registry_audit.py`
        📄 `h2q_bridge.py`
        📄 `multi_prism.py`
        📄 `vision_loader.py`
        📄 `mix_corpus_generator.py`
        📄 `code_writer.py`
    📂 **checkpoints/**
    📂 **tests/**
        📄 `test_cem.py`
        📄 `test_system_integration.py`
        📄 `__init__.py`
        📄 `test_sst.py`
        📄 `test_reversible_manifold.py`
        📄 `test_trace_formula.py`
        📄 `test_cost_functional.py`
        📄 `test_crystal_integration.py`
        📄 `test_api_contract.py`
        📄 `test_dde.py`
    📂 **h2q/**
        📄 `train_zero_memory.py`
        📄 `hierarchical_decoder.py`
        📄 `system.py`
        📄 `gut_kernel.py`
        📄 `quaternion_ops.py`
        📄 `knot_kernel.py`
        📄 `prism_engine.py`
        📄 `fdc_kernel.py`
        📄 `cem.py`
        📄 `dream_engine.py`
        📄 `meta_learner.py`
        📄 `__init__.py`
        📄 `decision_engine.py`
        📄 `cost_functional.py`
        📄 `spacetime_kernel.py`
        📄 `group_ops.py`
        📄 `production_logical_generator.py`
        📄 `spacetime_3d_kernel.py`
        📄 `engine.py`
        📄 `fractal_embedding.py`
        📄 `reversible_kernel.py`
        📄 `trace_formula.py`
        📄 `kernel_engine.py`
        📄 `kernels.py`
        📄 `dde.py`
        📄 `train_distillation.py`
        📄 `train_full_stack_v2.py`
        📄 `vision_loader.py`
        📄 `hierarchical_system.py`
        📄 `main.py`
        📄 `benchmark_latency.py`
        📄 `dreaming.py`
        📄 `manual_reversible_kernel.py`
        📂 **visualization/**
            📄 `spectral_dream_visualizer.py`
            📄 `__init__.py`
            📄 `holomorphic_path_visualizer.py`
        📂 **kernels/**
            📄 `cmeb.py`
            📄 `__init__.py`
            📄 `metal_spectral_det.py`
            📄 `resonance_tiling.py`
            📄 `m4_quat_conv.py`
            📄 `topological_braiding.py`
            📂 **quantization/**
                📄 `tpq_v2.py`
                📄 `__init__.py`
        📂 **topology/**
            📄 `__init__.py`
            📄 `entropy_routing.py`
        📂 **grounding/**
            📄 `genomic_streamer.py`
            📄 `__init__.py`
        📂 **core/**
            📄 `hhk.py`
            📄 `metal_jit_bridge.py`
            📄 `ddfl_dream_connector.py`
            📄 `spacetime.py`
            📄 `synesthesia_engine.py`
            📄 `generation.py`
            📄 `berry_phase_sync.py`
            📄 `manifold.py`
            📄 `unified_orchestrator.py`
            📄 `interpolation.py`
            📄 `interface_registry.py`
            📄 `gter_system.py`
            📄 `dynamic_inference.py`
            📄 `synesthesia_loss.py`
            📄 `spectral_tuner.py`
            📄 `geodesic_surgery.py`
            📄 `manifold_recovery.py`
            📄 `adapter.py`
            📄 `manifold_scaler.py`
            📄 `compression.py`
            📄 `isomorphism_bridge.py`
            📄 `ttd_scheduler.py`
            📄 `geodesic_kernel.py`
            📄 `manifold_shield.py`
            📄 `tpq_engine.py`
            📄 `__init__.py`
            📄 `decision_engine.py`
            📄 `gter_diagnostic.py`
            📄 `decision.py`
            📄 `discrete_decision_engine.py`
            📄 `strider.py`
            📄 `ddfl_integration.py`
            📄 `engine.py`
            📄 `autonomous_system.py`
            📄 `manifold_audit.py`
            📄 `reversible_kernel.py`
            📄 `interferometer.py`
            📄 `orchestrator.py`
            📄 `homeostatic_trainer.py`
            📄 `cas_kernel.py`
            📄 `berry_cross_attenuator.py`
            📄 `topology.py`
            📄 `manifold_alignment.py`
            📄 `adaptive_striding.py`
            📄 `memory_crystal.py`
            📄 `holomorphic_controller.py`
            📄 `synesthesia_bridge.py`
            📄 `sst.py`
            📄 `ddfl.py`
            📄 `logic_auditing.py`
            📄 `fueter_beam_search.py`
            📄 `zwi_engine.py`
            📄 `bprm.py`
            📄 `unified_kernel.py`
            📄 `uam_trainer.py`
            📄 `resonator.py`
            📄 `resonance_buffer.py`
            📂 **metrics/**
                📄 `__init__.py`
                📄 `geodesic_integrator.py`
            📂 **kernels/**
                📄 `__init__.py`
            📂 **topology/**
                📄 `entropy_router.py`
                📄 `__init__.py`
                📄 `knot_hash.py`
            📂 **memory/**
                📄 `h2q_vault.py`
                📄 `mps_swap.py`
                📄 `topological_forgetting.py`
                📄 `__init__.py`
                📄 `gpim.py`
                📄 `berry_kv_cache.py`
                📄 `rskh_vault.py`
            📂 **layers/**
                📄 `usc_barycenter.py`
                📄 `__init__.py`
                📄 `hamilton_reversible_cell.py`
                📄 `spectral_pooling.py`
            📂 **accelerators/**
                📄 `__init__.py`
                📄 `hamilton_amx_bridge.py`
            📂 **calibration/**
                📄 `__init__.py`
                📄 `genomic_vision_suite.py`
                📄 `holonomy_calibrator.py`
                📄 `synesthesia_calibration_suite.py`
                📄 `berry_phase.py`
            📂 **shields/**
                📄 `manifold_shield.py`
                📄 `__init__.py`
            📂 **optimization/**
                📄 `fdc_optimizer.py`
                📄 `holomorphic_projection.py`
                📄 `holomorphic_healing.py`
                📄 `__init__.py`
            📂 **serialization/**
                📄 `__init__.py`
                📄 `manifold_snapshot.py`
                📄 `uqc_handler.py`
            📂 **distillation/**
                📄 `code_genomic_distiller.py`
                📄 `topological_distiller.py`
                📄 `__init__.py`
                📄 `cmi_distiller.py`
                📄 `holonomy_distiller.py`
                📄 `avtg_distiller.py`
                📄 `code_geometric_bridge.py`
            📂 **optimizers/**
                📄 `fdc_optimizer.py`
                📄 `hjb_solver.py`
                📄 `spectral_drag.py`
                📄 `__init__.py`
                📄 `spectral_drag_scheduler.py`
                📄 `su2_momentum.py`
            📂 **quantization/**
                📄 `tpq_engine.py`
                📄 `__init__.py`
                📄 `fractal_quantizer.py`
                📄 `quaternionic_protocol.py`
            📂 **trainers/**
                📄 `berry_synesthesia.py`
                📄 `unified_barycenter_trainer.py`
                📄 `__init__.py`
                📄 `holomorphic_hjb_healer.py`
                📄 `berry_fusion_unified.py`
                📄 `sleep_healer.py`
            📂 **pruning/**
                📄 `__init__.py`
                📄 `geodesic_engine.py`
            📂 **audit/**
                📄 `crosstalk_auditor.py`
                📄 `__init__.py`
                📄 `manifold_audit.py`
                📄 `genomic_invariant_audit.py`
            📂 **persistence/**
                📄 `rskh_uqc_layer.py`
                📄 `l2_super_knot.py`
                📄 `gter_storage.py`
                📄 `__init__.py`
                📄 `spectral_swap_manager.py`
            📂 **diagnostics/**
                📄 `__init__.py`
                📄 `fractal_recovery.py`
            📂 **alignment/**
                📄 `bargmann_aligner.py`
                📄 `__init__.py`
                📄 `cmga_interferometer.py`
                📄 `berry_phase_comparator.py`
                📄 `karcher_flow_aligner.py`
            📂 **ops/**
                📄 `__init__.py`
                📄 `hamilton_amx.py`
            📂 **monitoring/**
                📄 `__init__.py`
                📄 `manifold_audit.py`
            📂 **guards/**
                📄 `holomorphic_sparsity_guard.py`
                📄 `__init__.py`
                📄 `hqa_guard.py`
                📄 `holomorphic_guard.py`
                📄 `holomorphic_guard_middleware.py`
                📄 `holomorphic_beam_search.py`
            📂 **generation/**
                📄 `__init__.py`
                📄 `holomorphic_backtracker.py`
        📂 **memory/**
            📄 `geodesic_window.py`
            📄 `__init__.py`
            📄 `geodesic_replay.py`
        📂 **layers/**
            📄 `interference.py`
            📄 `__init__.py`
            📄 `amx_spinor_interference.py`
            📄 `amx_linear.py`
            📄 `interferometric_gating.py`
            📄 `quantum_alignment.py`
            📂 **fusion/**
                📄 `__init__.py`
                📄 `manifold_interferometer.py`
        📂 **optimizer/**
            📄 `fdc_optimizer.py`
            📄 `__init__.py`
        📂 **calibration/**
            📄 `__init__.py`
            📄 `holonomy.py`
        📂 **bridge/**
            📄 `multimodal.py`
            📄 `__init__.py`
        📂 **experiments/**
            📄 `berry_phase_interferometer.py`
            📄 `synesthesia_4way_alignment.py`
            📄 `__init__.py`
        📂 **dispatch/**
            📄 `amx_orchestrator.py`
            📄 `__init__.py`
            📄 `amx_tiling_dispatcher.py`
        📂 **logic/**
            📄 `holomorphic_filter.py`
            📄 `__init__.py`
            📄 `fueter_pruner.py`
            📄 `holomorphic_gating_unit.py`
        📂 **utils/**
            📄 `mps_compat.py`
            📄 `__init__.py`
            📄 `visualizer.py`
        📂 **models/**
            📄 `hierarchical_decoder.py`
            📄 `h2q_world_model.py`
            📄 `__init__.py`
            📂 **bridges/**
                📄 `__init__.py`
                📄 `berry_phase_synesthesia.py`
            📂 **quantum_geometric/**
                📄 `__init__.py`
                📄 `interferometer.py`
        📂 **vision/**
            📄 `__init__.py`
            📄 `loader.py`
        📂 **optimizers/**
            📄 `fdc_optimizer.py`
            📄 `geodesic_unitary.py`
            📄 `__init__.py`
            📄 `spectral_entropy_wrapper.py`
        📂 **audit/**
            📄 `synesthesia_4way_audit.py`
            📄 `__init__.py`
            📄 `persistence_audit_v1.py`
            📄 `topology.py`
        📂 **resonance/**
            📄 `__init__.py`
            📄 `avt_resonator.py`
        📂 **engines/**
            📄 `__init__.py`
            📄 `decision.py`
        📂 **benchmarks/**
            📄 `zwi_geometric_crystal.py`
            📄 `geodesic_retrieval_benchmark.py`
            📄 `amx_tiled_profiler.py`
            📄 `fractal_latency_amx.py`
            📄 `__init__.py`
            📄 `persistence_stress_test.py`
            📄 `tis_streaming_v1.py`
            📄 `rskh_infinite_stress.py`
            📄 `mps_hamilton_optimizer.py`
            📄 `temporal_knot_persistence.py`
            📄 `infinite_context_persistence_audit.py`
        📂 **persistence/**
            📄 `rskh.py`
            📄 `__init__.py`
        📂 **diagnostics/**
            📄 `fueter_audit.py`
            📄 `__init__.py`
            📄 `manifold_entropy_audit.py`
        📂 **governance/**
            📄 `memory_governor.py`
            📄 `__init__.py`
            📄 `heat_death_governor.py`
            📄 `modality_synchronizer.py`
        📂 **ops/**
            📄 `memory_manager.py`
            📄 `m4_amx_extension.py`
            📄 `__init__.py`
            📄 `mps_amx_bridge.py`
            📄 `m4_amx_bridge.py`
            📄 `rskh_mmap_swapper.py`
        📂 **decision/**
            📄 `__init__.py`
            📄 `dde.py`
        📂 **dna_topology/**
            📄 `__init__.py`
            📄 `topology_engine.py`
        📂 **physics/**
            📄 `spectral_ops.py`
            📄 `__init__.py`
        📂 **routing/**
            📄 `dynamic_precision.py`
            📄 `__init__.py`
        📂 **monitoring/**
            📄 `__init__.py`
            📄 `mhdm.py`
        📂 **data/**
            📄 `universal_stream.py`
            📄 `__init__.py`
            📄 `generator.py`
        📂 **engine/**
            📄 `discrete_decision.py`
            📄 `__init__.py`
            📄 `decision_engine.py`
            📄 `decision.py`
            📄 `curiosity.py`
        📂 **loaders/**
            📄 `audio_knot.py`
            📄 `__init__.py`
        📂 **control/**
            📄 `ast_engine.py`
            📄 `dmdc.py`
            📄 `__init__.py`
        📂 **validation/**
            📄 `__init__.py`
            📄 `compression_audit.py`

## 2. 核心数学实现 (Mathematical Core)
> 以下模块包含关键的几何/拓扑/代数逻辑实现：

### 📄 train_spacetime_vision.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: H2Q Discrete Decision Engine for manifold selection.
  - *Methods*: __init__, forward...

### 📄 train_fdc.py
- **Class `H2Q_Manifold`**
  - *Doc*: [EXPERIMENTAL] Geometric AGI framework grounded in SU(2).

### 📄 train_fdc_pure.py
- **Class `FDCOptimizer`**
  - *Doc*: Fractal Differential Calculus (FDC) Optimizer.
- **Class `SpectralShiftTracker`**
  - *Doc*: Quantifies learning progress η = (1/π) arg{det(S)}.

### 📄 verify_multimodal_cpu.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements η = (1/π) arg{det(S)} to quantify cognitive deflection.
  - *Methods*: __init__, forward...
- **Class `FractalExpansion`**
  - *Doc*: Fractal Expansion Protocol (2 -> 256).
  - *Methods*: __init__, forward...
- **Class `H2QSystem`**
  - *Doc*: The 'AutonomousSystem' equivalent, renamed to align with SU(2) naming conventions.
  - *Methods*: __init__, forward...

### 📄 train_synesthesia_unified.py
- **Class `SynesthesiaUnifiedTrainer`**
  - *Doc*: Trainer for multi-modal alignment on SU(2) manifolds.

### 📄 demo_neural_zip.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [FIXED] Resolved unexpected keyword argument 'num_actions'.
  - *Methods*: __init__, forward...
- **Class `H2QNeuralZip`**
  - *Doc*: Middleware for 8:1 Hierarchical Compression.

### 📄 train_synesthesia_avtg.py
- **Class `AVTGSynesthesiaTrainer`**
  - *Doc*: Unified 4-way modality alignment (Audio, Vision, Text, Genomic).
  - *Methods*: __init__, forward...

### 📄 train_multimodal.py
- **Class `H2QContrastiveLoss`**
  - *Doc*: Implements cross-modal alignment using the Spectral Shift Tracker (η).
  - *Methods*: __init__, forward...

### 📄 train_h2q.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements the Krein-like trace formula for Spectral Shift (η).

### 📄 train_compression_test.py
- **Class `SU2Manifold`**
  - *Doc*: Utility to project 2-dim seeds into SU(2) representations.
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [FIX] Added 'dim' to __init__ to resolve Runtime Error.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates η = (1/π) arg{det(S)} to quantify learning progress.

### 📄 train_full_stack_v2.py
- **Class `SpectralShiftTracker`**
  - *Doc*: [EXPERIMENTAL] Implements η = (1/π) arg{det(S)}

### 📄 deploy_reasoning_vault.py
- **Class `H2QReasoningVaultController`**
  - *Doc*: Orchestrator for the 1M+ Context Reasoning Vault.

### 📄 train_manual_reversible.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: Implements infinitesimal rotations on the SU(2) manifold.
  - *Methods*: __init__, forward...

### 📄 train_omniscience.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: FIX: Removed 'dim' keyword argument from __init__ to resolve Runtime Error.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements η = (1/π) arg{det(S)} to track cognitive progress.

### 📄 tests/test_reversible_manifold.py
- **Class `ReversibleManifoldFunction`**
  - *Doc*: Implements a reversible coupling layer for the 256-dimensional manifold.
  - *Methods*: forward, backward...
- **Class `TestReversibleDrift`**
  - *Doc*: Bit-accurate unit tests to detect L1 gradient drift during manifold reconstruction.

### 📄 h2q/train_zero_memory.py
- **Class `AutonomousSystem`**
  - *Doc*: [STABLE] The H2Q Autonomous System core.
  - *Methods*: __init__, forward...
- **Class `FractalDifferentialCalculus`**
  - *Doc*: [EXPERIMENTAL] Vectorized FDC Implementation.

### 📄 h2q/hierarchical_decoder.py
- **Class `QuaternionLinear`**
  - *Doc*: 四元数线性层
  - *Methods*: __init__, forward...
- **Class `KnotRefiner`**
  - *Doc*: 纽结精炼块
  - *Methods*: __init__, forward...
- **Class `FractalStage`**
  - *Doc*: 分形展开阶段：1 -> 2
  - *Methods*: __init__, forward...

### 📄 h2q/system.py
- **Class `ConceptDecoder`**
  - *Doc*: Decodes quaternionic manifold states into logical concepts.
  - *Methods*: __init__, forward...

### 📄 h2q/gut_kernel.py
- **Class `H2Q_Geometric_Kernel`**
  - *Doc*: GUT 内核 v3.0 (Fractal Edition)
  - *Methods*: __init__, forward...

### 📄 h2q/knot_kernel.py
- **Class `QuaternionLinear`**
  - *Doc*: 四元数线性层
  - *Methods*: __init__, forward...
- **Class `H2Q_Knot_Kernel`**
  - *Doc*: H2Q 纽结内核 (底层拼写核) - 修正版
  - *Methods*: __init__, forward...

### 📄 h2q/prism_engine.py
- **Class `PrismConverter`**
  - *Doc*: Maps Transformer-based SVD embeddings into the SU(2) hypersphere (256-dim).
  - *Methods*: __init__, forward...
- **Class `DiscreteDecisionEngine`**
  - *Doc*: The H2Q Decision Atom processor. 
  - *Methods*: __init__, spectral_shift_tracker, forward...

### 📄 h2q/fdc_kernel.py
- **Class `GeodesicBackprop`**
  - *Doc*: Implementation of gradients as infinitesimal rotations on the SU(2) manifold.
  - *Methods*: forward, backward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates η = (1/π) arg{det(S)} to track cognitive deflection.
  - *Methods*: __init__, forward...
- **Class `FDCKernel`**
  - *Doc*: Fractal Dimension Controller (FDC) Kernel.
  - *Methods*: __init__, forward...

### 📄 h2q/dream_engine.py
- **Class `DreamingMechanism`**
  - *Doc*: Implements Sleep-Phase Gradient Synthesis for the H2Q architecture.

### 📄 h2q/meta_learner.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: STABLE: Fixed __init__ to accept 'dim' argument as required by the H2Q manifold projection.
  - *Methods*: __init__, forward...
- **Class `MetaLearner`**
  - *Doc*: EXPERIMENTAL: Implements the Sleep Phase dreaming mechanism.
  - *Methods*: __init__, calculate_spectral_shift, sleep_phase, record_trace, forward...

### 📄 h2q/cost_functional.py
- **Class `SpectralShiftFunction`**
  - *Doc*: 实现 η(λ) = (1/π) * arg{det(S(λ))}
  - *Methods*: __init__, forward...

### 📄 h2q/spacetime_kernel.py
- **Class `SpacetimeCell`**
  - *Doc*: 时空分形单元 (Spacetime Fractal Cell)
  - *Methods*: __init__, forward...

### 📄 h2q/group_ops.py
- **Class `HamiltonProductAMX`**
  - *Doc*: Optimized Hamilton Product for SU(2) Group Operations.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements the η = (1/π) arg{det(S)} logic for tracking learning progress.

### 📄 h2q/engine.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: Governs cognitive transitions on the quaternionic manifold.
  - *Methods*: __init__, calculate_spectral_shift, forward, reversible_kernel, inverse_reversible_kernel...

### 📄 h2q/fractal_embedding.py
- **Class `FractalEmbedding`**
  - *Doc*: H2Q 分形嵌入 (Fractal Embedding)
  - *Methods*: __init__, forward...

### 📄 h2q/trace_formula.py
- **Class `SpectralShiftTracker`**
  - *Doc*: [STABLE] SpectralShiftTracker (η)
- **Class `ContinuousEnvironmentModel`**
  - *Doc*: [EXPERIMENTAL] ContinuousEnvironmentModel

### 📄 h2q/kernel_engine.py
- **Class `H2Q_Knot_Kernel`**
  - *Doc*: Implements the SU(2) Geodesic Flow using Reversible Kernels.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates η = (1/π) arg{det(S)} to track cognitive phase deflection.
  - *Methods*: __init__, forward...

### 📄 h2q/kernels.py
- **Class `UnifiedTopologicalKernel`**
  - *Doc*: H2Q Unified Topological Kernel: Manages Geodesic Flow on SU(2) manifolds.
  - *Methods*: __init__, _accelerated_matmul, fractal_expand, geodesic_flow, forward...
- **Class `DiscreteDecisionEngine`**
  - *Doc*: Governs discrete transitions within the quaternionic space.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Learning progress tracker derived from the Krein-like trace formula.

### 📄 h2q/dde.py
- **Class `HamiltonProductAMX`**
  - *Doc*: [EXPERIMENTAL] Optimized Hamilton Product for M4 Silicon.
  - *Methods*: forward, backward...
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [STABLE] Fixed DiscreteDecisionEngine to resolve 'num_actions' unexpected keyword argument.
  - *Methods*: __init__, forward, get_spectral_shift...

### 📄 h2q/train_full_stack_v2.py
- **Class `H2QFullStackTrainer`**
  - *Doc*: H2Q Full Stack Trainer v2.1

### 📄 h2q/vision_loader.py
- **Class `VisionLoader`**
  - *Doc*: H2Q Vision Loader: Implements unified YCbCr-to-RGB manifold mapping.
  - *Methods*: __init__, ycbcr_to_rgb_manifold, forward...

### 📄 h2q/main.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements η = (1/π) arg{det(S)} to link atoms to environmental drag.

### 📄 h2q/dreaming.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: FIX: Corrected __init__ to accept 'manifold_dim' instead of 'dim' to resolve 
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates η = (1/π) arg{det(S)} to quantify learning progress.
- **Class `H2QDreamingMechanism`**
  - *Doc*: Synthesizes high-η reasoning traces by exploring the SU(2) manifold 

### 📄 h2q/visualization/spectral_dream_visualizer.py
- **Class `SpectralDreamRenderer`**
  - *Doc*: Visualizes the H2Q Sleep-phase healing cycle by projecting the 256-dimensional 

### 📄 h2q/kernels/cmeb.py
- **Class `CrossModalEntropyBalancer`**
  - *Doc*: CMEB: Synchronizes the Heat-Death Index (Spectral Entropy) between 
  - *Methods*: __init__, calculate_hdi, calculate_eta, inject_fractal_noise, forward...

### 📄 h2q/kernels/metal_spectral_det.py
- **Class `MetalSpectralDet`**
  - *Doc*: Metal-accelerated Spectral Determinant Kernel (Metal-Det).
  - *Methods*: __init__, forward...

### 📄 h2q/kernels/resonance_tiling.py
- **Class `ResonanceTilingKernel`**
  - *Doc*: Resonance-Tiling Kernel optimized for M4 AMX.
  - *Methods*: __init__, _hamilton_tile_prod, forward, _apply_tiled_transformation, inverse...

### 📄 h2q/kernels/m4_quat_conv.py
- **Class `QuatConvM4`**
  - *Doc*: H2Q Quaternionic Convolution (SU(2) Manifold).
  - *Methods*: __init__, _get_hamilton_matrix, forward...

### 📄 h2q/kernels/topological_braiding.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates η = (1/π) arg{det(S)} to track cognitive progress.
  - *Methods*: __init__, forward...

### 📄 h2q/kernels/quantization/tpq_v2.py
- **Class `TPQv2STE`**
  - *Doc*: Straight-Through Estimator for 4-bit Phase Quantization on SU(2).
  - *Methods*: forward, backward...
- **Class `TPQv2Kernel`**
  - *Doc*: Topological Phase Quantizer (v2) with QAT support.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Monitoring tool for cognitive transitions in the manifold.

### 📄 h2q/topology/entropy_routing.py
- **Class `TopologicalEntropyRouter`**
  - *Doc*: [EXPERIMENTAL] Topological Entropy Routing (TER).
  - *Methods*: __init__, compute_heat_death_index, forward, inverse...

### 📄 h2q/grounding/genomic_streamer.py
- **Class `TopologicalFASTAStreamer`**
  - *Doc*: H2Q Topological FASTA-Streamer

### 📄 h2q/core/hhk.py
- **Class `HolomorphicHealingKernel`**
  - *Doc*: Holomorphic Healing Kernel (HHK)
  - *Methods*: __init__, discrete_fueter_operator, forward, verify_geodesic_integrity...

### 📄 h2q/core/spacetime.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Quantifies learning progress via η = (1/π) arg{det(S)}.
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [FIXED] Added 'num_actions' to __init__ to resolve Runtime Error.
  - *Methods*: __init__, forward...

### 📄 h2q/core/synesthesia_engine.py
- **Class `FractalExpansion`**
  - *Doc*: No documentation.
  - *Methods*: __init__, forward...

### 📄 h2q/core/berry_phase_sync.py
- **Class `CrossModal_Berry_Phase_Sync`**
  - *Doc*: H2Q CrossModal_Berry_Phase_Sync
  - *Methods*: __init__, compute_frechet_mean, calculate_spectral_shift, forward, get_berry_curvature...

### 📄 h2q/core/manifold.py
- **Class `ReversibleManifoldFunction`**
  - *Doc*: [STABLE] Reversible Manifold Function for H2Q Framework.
  - *Methods*: forward, backward...
- **Class `ManifoldLayer`**
  - *Doc*: [EXPERIMENTAL] SU(2) Geodesic Flow Layer.
  - *Methods*: __init__, forward, verify_reconstruction...

### 📄 h2q/core/unified_orchestrator.py
- **Class `H2Q_Unified_Orchestrator`**
  - *Doc*: H2Q_Unified_Orchestrator: Automates transitions between Wake (External SGD) 
  - *Methods*: __init__, calculate_krein_eta, forward, _wake_cycle, _transition_to_sleep...

### 📄 h2q/core/interpolation.py
- **Class `SpectralSlerp`**
  - *Doc*: Implements Spectral Spherical Linear Interpolation (Slerp) for SU(2) manifold states.
  - *Methods*: __init__, forward, audit_transition...

### 📄 h2q/core/interface_registry.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: Base class for discrete decision atoms in the H2Q manifold.
- **Class `TopologicalRegistryInterdictor`**
  - *Doc*: Audits registry access to ensure manifold symmetry and prevent topological tears.

### 📄 h2q/core/gter_system.py
- **Class `GTER`**
  - *Doc*: Unified Geodesic Trace-Error Recovery (GTER) Middleware.

### 📄 h2q/core/dynamic_inference.py
- **Class `DynamicEtaModulatedPipeline`**
  - *Doc*: Dynamic η-Modulated Inference Pipeline.
  - *Methods*: __init__, forward, audit_pipeline_integrity...

### 📄 h2q/core/synesthesia_loss.py
- **Class `SynesthesiaInterferenceLoss`**
  - *Doc*: Entangles η-signatures from Vision (YCbCr) and Text (Byte-stream) manifolds.
  - *Methods*: __init__, forward, audit_resonance...

### 📄 h2q/core/spectral_tuner.py
- **Class `SpectralEntropyAutoTuner`**
  - *Doc*: SpectralEntropyAutoTuner: Monitors and regulates the manifold health of the H2Q system.
  - *Methods*: __init__, calculate_effective_rank, _generate_fractal_noise, forward...

### 📄 h2q/core/geodesic_surgery.py
- **Class `GeodesicGradientSurgery`**
  - *Doc*: Implements Geodesic Gradient Surgery (GGS) for the H2Q architecture.

### 📄 h2q/core/manifold_recovery.py
- **Class `UnitaryRecoveryHook`**
  - *Doc*: [STABLE] Unitary Recovery Hook
- **Class `ManifoldUnitaryRecovery`**
  - *Doc*: Wrapper for manual execution of the recovery process.

### 📄 h2q/core/adapter.py
- **Class `OnlineLearningAdapter`**
  - *Doc*: Adapter to inject real-time manifold alignment into the H2Q Server.

### 📄 h2q/core/manifold_scaler.py
- **Class `DynamicManifoldScaler`**
  - *Doc*: Modulates sequence striding (2:1 to 16:1) based on real-time Heat-Death Index (HDI) 
  - *Methods*: __init__, calculate_heat_death_index, get_compression_ratio, forward, reversible_reconstruct...

### 📄 h2q/core/compression.py
- **Class `SpectralShiftTracker`**
  - *Doc*: No documentation.
  - *Methods*: __init__, forward...

### 📄 h2q/core/ttd_scheduler.py
- **Class `TopologicalTimeDilation`**
  - *Doc*: TTD Scheduler: Dynamically adjusts compute allocation (Fractal Depth and Geodesic Step)

### 📄 h2q/core/geodesic_kernel.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements the Krein-like trace formula: η = (1/π) arg{det(S)}.

### 📄 h2q/core/manifold_shield.py
- **Class `ManifoldSingularityShield`**
  - *Doc*: Diagnostic monitor for detecting manifold collapse (det(S) -> 0).

### 📄 h2q/core/tpq_engine.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: STABLE: Fixed initialization to resolve 'dim' keyword error.
  - *Methods*: __init__, forward...

### 📄 h2q/core/decision_engine.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: H2Q Discrete Decision Engine (DDE)
  - *Methods*: __init__, compute_spectral_shift, update_autonomy_schedule, forward...

### 📄 h2q/core/gter_diagnostic.py
- **Class `GTERDiagnostic`**
  - *Doc*: Geodesic Trace-Error Recovery (GTER) Diagnostic.
  - *Methods*: __init__, calculate_spectral_shift, monitor_and_recover, forward...

### 📄 h2q/core/decision.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: H2Q Discrete Decision Engine (DDE).
  - *Methods*: __init__, forward, get_action_distribution...

### 📄 h2q/core/discrete_decision_engine.py
- **Class `LatentConfig`**
  - *Doc*: Unified configuration for H2Q Discrete Decision Engines.
- **Class `DiscreteDecisionEngine`**
  - *Doc*: The Discrete Decision Engine (DDE) is the primary switching mechanism in the H2Q architecture.
  - *Methods*: __init__, forward...

### 📄 h2q/core/strider.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements η = (1/π) arg{det(S)} based on SU(2) geodesic flow.
  - *Methods*: __init__, forward...

### 📄 h2q/core/ddfl_integration.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: STABLE: Corrected initialization to resolve 'dim' keyword error.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: EXPERIMENTAL: Implements the Krein-like trace formula for η.

### 📄 h2q/core/engine.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements η = (1/π) arg{det(S)} to track cognitive deflection.
  - *Methods*: __init__, forward...
- **Class `ReversibleQuaternionicKernel`**
  - *Doc*: Additive coupling for O(1) memory: y1 = x1 + F(x2); y2 = x2 + G(y1)
  - *Methods*: __init__, forward...
- **Class `FractalExpansion`**
  - *Doc*: Expands 2-atom seeds to 256-dim quaternionic knots.
  - *Methods*: __init__, forward...

### 📄 h2q/core/autonomous_system.py
- **Class `AutonomousSystem`**
  - *Doc*: H2Q Autonomous System utilizing TPQ-v2 and Spectral Shift Tracking (η).
  - *Methods*: __init__, _calculate_eta, forward...

### 📄 h2q/core/manifold_audit.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: FIX: Removed 'dim' argument causing Runtime Error.
  - *Methods*: __init__, forward...
- **Class `CrossManifoldAudit`**
  - *Doc*: [EXPERIMENTAL] Utility to measure spectral overlap between Vision, Text, and Code manifolds.

### 📄 h2q/core/reversible_kernel.py
- **Class `SpectralShiftTracker`**
  - *Doc*: No documentation.
- **Class `ReversibleFractalLayer`**
  - *Doc*: No documentation.
  - *Methods*: __init__, forward, validate_reconstruction...

### 📄 h2q/core/interferometer.py
- **Class `FractalExpansion`**
  - *Doc*: Atom 1: Fractal Expansion (2 -> 256)
  - *Methods*: __init__, forward...
- **Class `BerryPhaseInterferometer`**
  - *Doc*: TASK: Multimodal alignment via Pancharatnam-Berry phase.
  - *Methods*: __init__, _to_spinor, forward...

### 📄 h2q/core/orchestrator.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [FIXED] Resolved 'unexpected keyword argument dim'. 
  - *Methods*: __init__, forward...
- **Class `UnifiedSleepOrchestrator`**
  - *Doc*: H2Q Orchestrator: Automates Wake/Sleep transitions via Spectral Entropy.
  - *Methods*: __init__, calculate_spectral_entropy, calculate_spectral_shift, sleep_phase_replay, forward...

### 📄 h2q/core/homeostatic_trainer.py
- **Class `HomeostaticTrainer`**
  - *Doc*: Homeostatic Trainer: Orchestrates the transition between Wake (SGD) and 

### 📄 h2q/core/berry_cross_attenuator.py
- **Class `BerryPhaseCrossAttenuator`**
  - *Doc*: Berry-Phase Cross-Attenuator: Replaces Euclidean dot-product attention with 
  - *Methods*: __init__, _get_quaternion_conjugate, forward, audit_manifold...

### 📄 h2q/core/topology.py
- **Class `TopologicalPruningHook`**
  - *Doc*: Implements dynamic zeroing of manifold atoms based on Spectral Shift volatility.
- **Class `DiscreteDecisionEngine`**
  - *Doc*: REVISED: Fixed __init__ to resolve 'dim' keyword argument error.
  - *Methods*: __init__, update_topology, forward...

### 📄 h2q/core/manifold_alignment.py
- **Class `H2QContrastiveLoss`**
  - *Doc*: Implements Cross-Modal Isomorphism between Spacetime and Multilingual manifolds.
  - *Methods*: __init__, compute_spectral_shift, forward...

### 📄 h2q/core/adaptive_striding.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements η = (1/π) arg{det(S)} to track environmental drag μ(E).

### 📄 h2q/core/memory_crystal.py
- **Class `MemoryManager`**
  - *Doc*: Architect of the H2Q Memory Crystal system.

### 📄 h2q/core/holomorphic_controller.py
- **Class `UnifiedHolomorphicController`**
  - *Doc*: Unified Holomorphic Gating Controller (UHGC).
  - *Methods*: __init__, forward, create_safe_dde...

### 📄 h2q/core/synesthesia_bridge.py
- **Class `SynesthesiaBridge`**
  - *Doc*: H2Q Synesthesia Bridge: Enforces Berry Phase consistency between Vision (YCbCr) 
  - *Methods*: __init__, compute_berry_phase, forward...

### 📄 h2q/core/sst.py
- **Class `SpectralShiftTracker`**
  - *Doc*: 谱位移追踪器 (SST)。

### 📄 h2q/core/ddfl.py
- **Class `SpectralShiftTracker`**
  - *Doc*: No documentation.

### 📄 h2q/core/logic_auditing.py
- **Class `HolomorphicAuditKernel`**
  - *Doc*: EXPERIMENTAL CODE: Holomorphic Logic Auditing.

### 📄 h2q/core/fueter_beam_search.py
- **Class `FueterGuidedBeamSearch`**
  - *Doc*: Unified Holomorphic Beam Search (UHBS).

### 📄 h2q/core/zwi_engine.py
- **Class `GeometricCrystal`**
  - *Doc*: [EXPERIMENTAL] Zero-Weight Inference (ZWI) Engine.
  - *Methods*: __init__, _apply_geodesic_flow, forward...

### 📄 h2q/core/bprm.py
- **Class `SU2ExponentialMap`**
  - *Doc*: Maps su(2) Lie Algebra elements to SU(2) Group elements (Unit Quaternions).
  - *Methods*: forward...
- **Class `BerryPhaseRecurrentManifold`**
  - *Doc*: Berry-Phase Recurrent Manifold (BPRM).
  - *Methods*: __init__, forward, get_holonomy_signature...

### 📄 h2q/core/unified_kernel.py
- **Class `HamiltonProductAMX`**
  - *Doc*: [EXPERIMENTAL] Unified L0 Topological Spelling Kernel.
  - *Methods*: __init__, _hamilton_product, spectral_shift_tracker, forward...

### 📄 h2q/core/uam_trainer.py
- **Class `H2QUnifiedAutonomousMaster`**
  - *Doc*: UAM Trainer: Orchestrates Wake-phase SGD and Sleep-phase Geodesic Healing.

### 📄 h2q/core/resonator.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements the Krein-like trace formula: η = (1/π) arg{det(S)}
  - *Methods*: __init__, forward...
- **Class `UnifiedMultimodalResonator`**
  - *Doc*: H2Q Unified Resonator: Entangles Vision, Text, and Audio.
  - *Methods*: __init__, _calculate_pb_phase, forward...

### 📄 h2q/core/resonance_buffer.py
- **Class `H2QResonanceBuffer`**
  - *Doc*: H2Q-Resonance-Buffer: A persistent state-layer utilizing SU(2) group theory.
  - *Methods*: __init__, _quaternion_multiply, _compute_spectral_shift, forward...

### 📄 h2q/core/metrics/geodesic_integrator.py
- **Class `GeodesicPathIntegrator`**
  - *Doc*: [STABLE] Geodesic-Path-Integrator

### 📄 h2q/core/topology/entropy_router.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [STABLE] Fixed initialization to resolve 'dim' keyword error.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: [EXPERIMENTAL] Calculates eta = (1/pi) arg{det(S)} to map environmental drag.
  - *Methods*: __init__, forward...

### 📄 h2q/core/topology/knot_hash.py
- **Class `SubKnotHasher`**
  - *Doc*: [STABLE] Recursive Sub-Knot Hashing Engine.
  - *Methods*: __init__, _to_su2, compute_spectral_shift, forward...

### 📄 h2q/core/memory/h2q_vault.py
- **Class `H2QVault`**
  - *Doc*: H2Q-Vault: Persistent memory layer for O(1) retrieval of context knots.

### 📄 h2q/core/memory/mps_swap.py
- **Class `ManifoldPagingSystem`**
  - *Doc*: Spectral Swap Middleware: Monitors the 16GB RAM ceiling and dynamically 
- **Class `GeodesicPrefetcher`**
  - *Doc*: Experimental: Predicts which knots will be needed based on Geodesic Flow.

### 📄 h2q/core/memory/topological_forgetting.py
- **Class `TopologicalForgettingController`**
  - *Doc*: Manages RSKH Vault pruning using eta-volatility metrics.

### 📄 h2q/core/memory/berry_kv_cache.py
- **Class `QuaternionicBerryCache`**
  - *Doc*: Quaternionic Berry-Phase KV-Cache

### 📄 h2q/core/memory/rskh_vault.py
- **Class `RSKHVault`**
  - *Doc*: Recursive Sub-Knot Hashing (RSKH) Vault with Saliency-Based Knot Eviction.

### 📄 h2q/core/layers/usc_barycenter.py
- **Class `USCBarycenter`**
  - *Doc*: Unified Synesthesia Center (USC) Barycenter.
  - *Methods*: __init__, _tiled_hamilton_update, karcher_flow, forward...

### 📄 h2q/core/layers/hamilton_reversible_cell.py
- **Class `HamiltonReversibleFunction`**
  - *Doc*: Manual Autograd function for Hamilton Reversible Cell.
  - *Methods*: forward, backward, _hamilton_amx_tile...

### 📄 h2q/core/layers/spectral_pooling.py
- **Class `SpectralManifoldPooling`**
  - *Doc*: Spectral Manifold Pooling Layer.
  - *Methods*: __init__, forward, _audit_spectral_transition, extra_repr...

### 📄 h2q/core/accelerators/hamilton_amx_bridge.py
- **Class `HamiltonAMXBridge`**
  - *Doc*: Architectural Bridge for M4 Silicon AMX acceleration.
  - *Methods*: __init__, forward, _fallback_hamilton_product, audit_throughput...

### 📄 h2q/core/calibration/genomic_vision_suite.py
- **Class `GenomicVisionCalibrationSuite`**
  - *Doc*: Berry-Phase Calibration Suite for Genomic-Vision synesthesia.

### 📄 h2q/core/calibration/holonomy_calibrator.py
- **Class `CrossModalHolonomyCalibrator`**
  - *Doc*: H2Q Cross-Modal Holonomy Calibrator
  - *Methods*: __init__, _to_quaternion, _compute_su2_rotation, calculate_holonomy, forward...

### 📄 h2q/core/calibration/synesthesia_calibration_suite.py
- **Class `SynesthesiaCalibrationSuite`**
  - *Doc*: Synesthesia Calibration Suite

### 📄 h2q/core/calibration/berry_phase.py
- **Class `BerryPhaseCalibrator`**
  - *Doc*: H2Q Cross-Modal Calibration Suite.
  - *Methods*: __init__, _to_quaternion, compute_berry_curvature, forward...

### 📄 h2q/core/shields/manifold_shield.py
- **Class `ManifoldSingularityShield`**
  - *Doc*: H2Q Runtime Wrapper: Manifold Singularity Shield
  - *Methods*: __init__, compute_effective_rank, fractal_noise_injection, forward...

### 📄 h2q/core/optimization/fdc_optimizer.py
- **Class `FDCOptimizer`**
  - *Doc*: Fractal Differential Calculus (FDC) Optimizer.

### 📄 h2q/core/optimization/holomorphic_projection.py
- **Class `HolomorphicGradientHook`**
  - *Doc*: Implements a Holomorphic Gradient Projection Hook.

### 📄 h2q/core/optimization/holomorphic_healing.py
- **Class `HolomorphicHealingBackprop`**
  - *Doc*: HolomorphicHealingBackprop Kernel
  - *Methods*: __init__, compute_fueter_residual, synthesize_healing_rotation, heal_trace, forward...

### 📄 h2q/core/serialization/manifold_snapshot.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: FIX: Removed 'dim' from __init__ to resolve Runtime Error.
- **Class `RSKHEncoder`**
  - *Doc*: Reversible Symmetric Kernel Hashing (RSKH).
- **Class `ManifoldSnapshot`**
  - *Doc*: Handles persistent storage of 1M+ context knots.

### 📄 h2q/core/serialization/uqc_handler.py
- **Class `UQCManager`**
  - *Doc*: Architect of the .h2q format. 

### 📄 h2q/core/distillation/code_genomic_distiller.py
- **Class `CodeGenomicDistiller`**
  - *Doc*: Aligns StarCoder byte-streams with Genomic FASTA manifolds via Berry Phase interference.
  - *Methods*: __init__, _compute_berry_phase, forward, distillation_step...

### 📄 h2q/core/distillation/topological_distiller.py
- **Class `SU2Projector`**
  - *Doc*: Projects 256-D vectors into the SU(2) unit hypersphere.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Quantifies learning progress via the Krein-like trace formula.

### 📄 h2q/core/distillation/cmi_distiller.py
- **Class `CMIDistiller`**
  - *Doc*: Cross-Manifold Interference (CMI) Distiller.
  - *Methods*: __init__, _to_quaternion_manifold, calculate_spectral_interference, forward, distill_step...

### 📄 h2q/core/distillation/holonomy_distiller.py
- **Class `CrossModalHolonomyDistiller`**
  - *Doc*: CMHD: Forces Vision (YCbCr) and Text (Byte-stream) manifolds to converge
  - *Methods*: __init__, compute_spectral_shift, forward...

### 📄 h2q/core/distillation/avtg_distiller.py
- **Class `AVTGIsomorphismDistiller`**
  - *Doc*: AVT-G Isomorphism Distiller
  - *Methods*: __init__, _to_quaternions, _geodesic_distance, karcher_flow, forward...

### 📄 h2q/core/optimizers/fdc_optimizer.py
- **Class `FDCOptimizer`**
  - *Doc*: Fractal Differential Calculus (FDC) Optimizer with Topological Braking.

### 📄 h2q/core/optimizers/hjb_solver.py
- **Class `HJBGeodesicSolver`**
  - *Doc*: Hamilton-Jacobi-Bellman Optimizer for H2Q.

### 📄 h2q/core/optimizers/spectral_drag.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates the Krein-like spectral shift η = (1/π) arg{det(S)}
- **Class `SpectralDragOptimizer`**
  - *Doc*: SDO: Modulates learning rate as an inverse function of environmental drag μ(E).

### 📄 h2q/core/optimizers/spectral_drag_scheduler.py
- **Class `SpectralDragScheduler`**
  - *Doc*: Spectral-Drag-Scheduler: An adaptive learning rate controller for the H2Q architecture.

### 📄 h2q/core/optimizers/su2_momentum.py
- **Class `SU2ParallelTransportOptimizer`**
  - *Doc*: Implements Parallel-Transport Momentum on the S³ manifold.

### 📄 h2q/core/quantization/tpq_engine.py
- **Class `TopologicalPhaseQuantizer`**
  - *Doc*: [EXPERIMENTAL] TPQ Engine

### 📄 h2q/core/quantization/fractal_quantizer.py
- **Class `FractalWeightQuantizer`**
  - *Doc*: Implements 4-bit Fractal Weight Quantization (FWQ).
  - *Methods*: __init__, _generate_fractal_bins, forward...

### 📄 h2q/core/quantization/quaternionic_protocol.py
- **Class `QuaternionicQuantizer`**
  - *Doc*: [STABLE] Quaternionic Quantization Protocol for SU(2) Manifolds.
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements the Krein-like trace formula: η = (1/π) arg{det(S)}

### 📄 h2q/core/trainers/berry_synesthesia.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [STABLE] Corrected implementation to resolve 'unexpected keyword argument dim'.
  - *Methods*: __init__, forward...
- **Class `BerryPhaseSynesthesiaTrainer`**
  - *Doc*: [EXPERIMENTAL] Synchronizes Vision and Text manifolds via SU(2) Geometric Phase.
  - *Methods*: __init__, _build_fractal_encoder, project_to_su2, compute_spectral_shift, forward...

### 📄 h2q/core/trainers/holomorphic_hjb_healer.py
- **Class `HolomorphicHJBHealer`**
  - *Doc*: Implements the Sleep Phase trainer using Hamilton-Jacobi-Bellman (HJB) equations

### 📄 h2q/core/trainers/berry_fusion_unified.py
- **Class `AVTGBerryFusion`**
  - *Doc*: Unified Cross-Modal Berry Phase Fusion Engine.
  - *Methods*: __init__, forward...

### 📄 h2q/core/trainers/sleep_healer.py
- **Class `H2QSleepHealer`**
  - *Doc*: The Sleep-Phase Self-Healing Trainer.

### 📄 h2q/core/pruning/geodesic_engine.py
- **Class `GeodesicPruningEngine`**
  - *Doc*: H2Q Geodesic Pruning Engine

### 📄 h2q/core/audit/crosstalk_auditor.py
- **Class `ManifoldCrosstalkAuditor`**
  - *Doc*: Manifold Crosstalk Auditor

### 📄 h2q/core/audit/manifold_audit.py
- **Class `ManifoldSingularityAudit`**
  - *Doc*: H2Q Manifold Singularity Audit (MSA)

### 📄 h2q/core/audit/genomic_invariant_audit.py
- **Class `GenomicInvariantAudit`**
  - *Doc*: Expands the audit to support streaming FASTA data and Berry Phase synchronization

### 📄 h2q/core/persistence/rskh_uqc_layer.py
- **Class `RSKH_UQC_Persistence`**
  - *Doc*: RSKH-UQC Unified Persistence Layer.

### 📄 h2q/core/persistence/l2_super_knot.py
- **Class `L2SuperKnotPersistence`**
  - *Doc*: L2 'Super-Knot' Persistence Layer.
  - *Methods*: __init__, fractal_expand, rskh_v2_step, forward...

### 📄 h2q/core/persistence/gter_storage.py
- **Class `GTERStorage`**
  - *Doc*: Geodesic Trace-Error Recovery (GTER) Persistent Storage.

### 📄 h2q/core/persistence/spectral_swap_manager.py
- **Class `KnotMetadata`**
  - *Doc*: No documentation.
- **Class `SpectralSwapManager`**
  - *Doc*: Unified persistence controller for the H2Q Manifold.

### 📄 h2q/core/diagnostics/fractal_recovery.py
- **Class `KreinTracker`**
  - *Doc*: Implements the Spectral Shift Tracker (η) based on the Krein-like trace formula.
- **Class `FractalRecoverySystem`**
  - *Doc*: [EXPERIMENTAL] Monitors Fractal Expansion rank and injects 'Fractal Noise' (h ± δ).

### 📄 h2q/core/alignment/bargmann_aligner.py
- **Class `BargmannInvariantAligner`**
  - *Doc*: BargmannInvariantAligner: Measures the 3-point geometric phase loop 
  - *Methods*: __init__, compute_bargmann_phase, forward, audit_isomorphism...

### 📄 h2q/core/alignment/cmga_interferometer.py
- **Class `BerryPhaseInterferometer`**
  - *Doc*: Aligns Audio, Vision, and Text by calculating the geometric phase (Berry Phase)
  - *Methods*: __init__, _to_quaternions, _compute_spectral_shift, forward...

### 📄 h2q/core/alignment/berry_phase_comparator.py
- **Class `BerryPhaseComparator`**
  - *Doc*: Metal-accelerated Pancharatnam-Berry phase comparator.
  - *Methods*: __init__, _calculate_pancharatnam_phase, detect_drift, forward...

### 📄 h2q/core/alignment/karcher_flow_aligner.py
- **Class `CrossModalKarcherFlowAligner`**
  - *Doc*: Implements a Karcher Flow Aligner to synchronize Audio, Vision, and Text η-signatures.

### 📄 h2q/core/ops/hamilton_amx.py
- **Class `HamiltonOptimizer`**
  - *Doc*: Architectural Component: Optimized Quaternionic Matrix Multiplication.
  - *Methods*: __init__, solve_engine_init_error, forward...

### 📄 h2q/core/monitoring/manifold_audit.py
- **Class `ManifoldAuditor`**
  - *Doc*: H2Q Unified Manifold Audit Dashboard.

### 📄 h2q/core/guards/holomorphic_sparsity_guard.py
- **Class `HolomorphicSparsityGuard`**
  - *Doc*: Holomorphic Sparsity Guard
  - *Methods*: __init__, compute_fueter_residual, forward, audit_logic_curvature...

### 📄 h2q/core/guards/hqa_guard.py
- **Class `HQAGuard`**
  - *Doc*: Higher-Order Quaternionic Analytic Guard (HQA-Guard).
  - *Methods*: __init__, _compute_fueter_gradient, calculate_logic_curvature, prune_hallucinations, forward...

### 📄 h2q/core/guards/holomorphic_guard.py
- **Class `HolomorphicReasoningGuard`**
  - *Doc*: [EXPERIMENTAL] Holomorphic Reasoning Guard (HRG).

### 📄 h2q/core/generation/holomorphic_backtracker.py
- **Class `HolomorphicBacktracker`**
  - *Doc*: M24-CW Implementation: Holomorphic Backtracking Decoder.

### 📄 h2q/memory/geodesic_window.py
- **Class `SlidingWindowGeodesicMemory`**
  - *Doc*: Sliding-Window Geodesic Memory (SWGM) based on SU(2) Group Theory.
  - *Methods*: __init__, _hamilton_product, _geodesic_fade, update, forward...

### 📄 h2q/memory/geodesic_replay.py
- **Class `GeodesicTraceHealer`**
  - *Doc*: Sleep-phase optimizer that iterates through GTERStorage traces and applies 

### 📄 h2q/layers/interference.py
- **Class `CPIGating`**
  - *Doc*: Constructive Phase Interference Gating (CPIG).
  - *Methods*: __init__, _to_spinors, forward, get_spectral_shift...

### 📄 h2q/layers/amx_spinor_interference.py
- **Class `AMXSpinorInterference`**
  - *Doc*: AMX-Tiled Spinor-Interference Layer.
  - *Methods*: __init__, _conjugate_spinor, forward, get_layer_metadata...

### 📄 h2q/layers/amx_linear.py
- **Class `AMXQuaternionicLinear`**
  - *Doc*: AMX-Hot-Swappable Linear Layer optimized for M4 Silicon.
  - *Methods*: __init__, reset_parameters, _tiled_hamilton_matmul, forward, audit_logic_integrity...

### 📄 h2q/layers/interferometric_gating.py
- **Class `InterferometricWaveGating`**
  - *Doc*: [EXPERIMENTAL] SU(2) Interferometric Wave-Gating Layer.
  - *Methods*: __init__, _quaternion_multiply, _compute_interference, forward...

### 📄 h2q/layers/quantum_alignment.py
- **Class `BerryPhaseInterferometer`**
  - *Doc*: [EXPERIMENTAL] Berry Phase Cross-Modality Interferometer
  - *Methods*: forward, backward...

### 📄 h2q/layers/fusion/manifold_interferometer.py
- **Class `ManifoldInterferometer`**
  - *Doc*: H2Q Manifold Interferometer
  - *Methods*: __init__, _to_quaternion, _pancharatnam_phase, _spectral_shift_update, forward...

### 📄 h2q/optimizer/fdc_optimizer.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [STABLE] Fixed initialization signature to resolve 'dim' keyword error.
  - *Methods*: __init__, forward...
- **Class `FDCOptimizer`**
  - *Doc*: [EXPERIMENTAL] Fractal-Derivative-Constrained Optimizer with Holomorphic Logic Auditing.

### 📄 h2q/calibration/holonomy.py
- **Class `BerryPhaseInterferometer`**
  - *Doc*: [EXPERIMENTAL] Detects geometric phase drift (Pancharatnam-Berry phase)
  - *Methods*: __init__, forward...

### 📄 h2q/bridge/multimodal.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Quantifies learning progress via η = (1/π) arg{det(S)}.
  - *Methods*: __init__, forward...
- **Class `DiscreteDecisionEngine`**
  - *Doc*: Handles the selection of geodesic paths within the manifold.
  - *Methods*: __init__, forward...
- **Class `H2QAlignmentBridge`**
  - *Doc*: Synthesizes Vision (YCbCr) and Text (Byte-stream) manifolds using SU(2) symmetry.
  - *Methods*: __init__, project_to_su2, forward...

### 📄 h2q/experiments/berry_phase_interferometer.py
- **Class `MultiModalBerryInterferometer`**
  - *Doc*: Measures geometric phase interference (Pancharatnam-Berry phase) between 
  - *Methods*: __init__, to_quaternion, calculate_geometric_phase, forward...
- **Class `BerryPhaseTrainer`**
  - *Doc*: No documentation.

### 📄 h2q/experiments/synesthesia_4way_alignment.py
- **Class `Synesthesia4WayAligner`**
  - *Doc*: H2Q Synesthesia Aligner: Executes 4-way topological alignment across 
  - *Methods*: __init__, map_to_su2, forward...

### 📄 h2q/dispatch/amx_orchestrator.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: STABLE CODE: Fixed from previous runtime error.

### 📄 h2q/logic/holomorphic_filter.py
- **Class `HolomorphicLogicFilter`**
  - *Doc*: Holomorphic Logic Filter (HLF)
  - *Methods*: __init__, _compute_fueter_divergence, forward...

### 📄 h2q/logic/holomorphic_gating_unit.py
- **Class `HolomorphicGatingUnit`**
  - *Doc*: HGU: Dampens logical paths where the quaternionic field exhibits high divergence (topological tears).
  - *Methods*: __init__, compute_fueter_residual, fueter_gating_hook, apply_to_layer, forward...

### 📄 h2q/utils/visualizer.py
- **Class `H2QVisualizer`**
  - *Doc*: H2Q Spectral Visualizer

### 📄 h2q/models/hierarchical_decoder.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: The DDE maps discrete logic atoms into the continuous manifold.
  - *Methods*: __init__, forward...
- **Class `KnotRefiner`**
  - *Doc*: Stabilizes the hierarchical decoding process by refining 'knots' 
  - *Methods*: __init__, forward...

### 📄 h2q/models/h2q_world_model.py
- **Class `H2QWorldModelPredictor`**
  - *Doc*: H2Q World-Model Predictor
  - *Methods*: __init__, forward, calculate_surprise, map_to_lie_algebra...

### 📄 h2q/models/bridges/berry_phase_synesthesia.py
- **Class `BerryPhaseBridge`**
  - *Doc*: Maps Audio Waveforms to YCbCr Manifolds using SU(2) Geodesic Flow.
  - *Methods*: __init__, compute_spectral_shift, forward...

### 📄 h2q/models/quantum_geometric/interferometer.py
- **Class `BerryPhaseInterferometer`**
  - *Doc*: [EXPERIMENTAL] Berry Phase Cross-Modality Interferometer
  - *Methods*: __init__, _to_spinors, forward...

### 📄 h2q/vision/loader.py
- **Class `VisionLoader`**
  - *Doc*: Architect: M24-Cognitive-Weaver
- **Class `DiscreteDecisionEngine`**
  - *Doc*: STABLE CODE: Fixed __init__ signature to accept 'num_actions'.
  - *Methods*: __init__, forward...

### 📄 h2q/optimizers/fdc_optimizer.py
- **Class `FDCOptimizer`**
  - *Doc*: FDC (Fractal-Differential-Coupling) Optimizer

### 📄 h2q/optimizers/geodesic_unitary.py
- **Class `GeodesicUnitaryOptimizer`**
  - *Doc*: FDC-Optim: Geodesic Unitary Optimizer.

### 📄 h2q/optimizers/spectral_entropy_wrapper.py
- **Class `SpectralEntropyLR`**
  - *Doc*: [STABLE] SED-LR: Spectral Entropy-Driven Learning Rate Wrapper.

### 📄 h2q/audit/synesthesia_4way_audit.py
- **Class `Synesthesia4WayAudit`**
  - *Doc*: Orchestrates a 4-way synesthesia audit (Audio-Vision-Text-Genome).
  - *Methods*: __init__, forward...

### 📄 h2q/audit/persistence_audit_v1.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates η = (1/π) arg{det(S)} via the Krein-like trace formula.

### 📄 h2q/resonance/avt_resonator.py
- **Class `ReversibleResonanceLayer`**
  - *Doc*: O(1) Memory Complexity: Reconstructs activations during backward pass.
  - *Methods*: forward, backward...
- **Class `UnifiedMultimodalResonator`**
  - *Doc*: H2Q Core: Aligns Audio, Vision, and Text via Pancharatnam-Berry Phase.
  - *Methods*: __init__, project_to_s3, forward...

### 📄 h2q/engines/decision.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [STABLE CODE]
  - *Methods*: __init__, forward, update_spectral_tracker...

### 📄 h2q/benchmarks/zwi_geometric_crystal.py
- **Class `ZWIBenchmark`**
  - *Doc*: Zero-Weight Inference (ZWI) Benchmark.

### 📄 h2q/benchmarks/geodesic_retrieval_benchmark.py
- **Class `QuaternionOps`**
  - *Doc*: No documentation.
- **Class `ManifoldSnapshot`**
  - *Doc*: No documentation.

### 📄 h2q/benchmarks/fractal_latency_amx.py
- **Class `HamiltonKernel`**
  - *Doc*: Implements the Hamilton Product (q1 * q2) optimized for MPS/AMX.
  - *Methods*: __init__, forward...

### 📄 h2q/benchmarks/tis_streaming_v1.py
- **Class `ReversibleKnot`**
  - *Doc*: Implements the Manual Reversible Kernel: y1 = x1 + F(x2); y2 = x2 + G(y1).
  - *Methods*: __init__, forward, inverse...
- **Class `SU2ManifoldProjection`**
  - *Doc*: Projects binary atoms into a 256-dim topological manifold using SU(2) symmetry.
  - *Methods*: __init__, forward...
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates η = (1/π) arg{det(S)} to track cognitive progress.

### 📄 h2q/benchmarks/temporal_knot_persistence.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Implements the Krein-like trace formula: η = (1/π) arg{det(S)}
- **Class `FractalExpansion`**
  - *Doc*: Recursive symmetry breaking (h ± δ) to expand 2-atom seeds to target manifold dimensions.
- **Class `ReversibleKnotKernel`**
  - *Doc*: No documentation.
  - *Methods*: __init__, forward...
- **Class `TemporalKnotBenchmark`**
  - *Doc*: No documentation.

### 📄 h2q/persistence/rskh.py
- **Class `SpectralShiftTracker`**
  - *Doc*: [EXPERIMENTAL] Implements the Krein-like trace formula for η-signatures.
  - *Methods*: __init__, forward...
- **Class `RSKH`**
  - *Doc*: Recursive Sub-Knot Hashing (RSKH).
  - *Methods*: __init__, _to_complex_su2, forward, retrieve_state...

### 📄 h2q/diagnostics/fueter_audit.py
- **Class `FueterAnalyticAudit`**
  - *Doc*: [STABLE] Fueter-Analytic Audit (FAA)

### 📄 h2q/diagnostics/manifold_entropy_audit.py
- **Class `ManifoldEntropyAudit`**
  - *Doc*: MEA (Manifold Entropy Audit) Utility

### 📄 h2q/governance/memory_governor.py
- **Class `MemoryPressureManifoldGovernor`**
  - *Doc*: MPMG: Memory Pressure Manifold Governor

### 📄 h2q/governance/heat_death_governor.py
- **Class `HeatDeathGovernor`**
  - *Doc*: Monitors the spectral entropy of the H2Q manifold.
  - *Methods*: __init__, calculate_spectral_entropy, forward...

### 📄 h2q/governance/modality_synchronizer.py
- **Class `ModalitySynchronizer`**
  - *Doc*: Middleware layer utilizing Cross-Modal Entropy Balancer (CMEB) to equalize 
  - *Methods*: __init__, calculate_hdi, forward, audit_synchrony...

### 📄 h2q/ops/memory_manager.py
- **Class `DynamicAMXMemorySwapper`**
  - *Doc*: M4-optimized buffer manager for H2Q SU(2) manifold segments.

### 📄 h2q/ops/m4_amx_extension.py
- **Class `M4AMXExtension`**
  - *Doc*: M4-AMX Tiled Quaternionic GEMM implementation.
  - *Methods*: __init__, forward, audit_throughput...

### 📄 h2q/ops/mps_amx_bridge.py
- **Class `HamiltonAMXBridge`**
  - *Doc*: Architectural Bridge for Quaternionic Operations on Apple Silicon (M4).

### 📄 h2q/ops/m4_amx_bridge.py
- **Class `M4AMXHotSwapBridge`**
  - *Doc*: M4-AMX-Hot-Swap Bridge

### 📄 h2q/ops/rskh_mmap_swapper.py
- **Class `RSKHMmapSwapper`**
  - *Doc*: High-performance memory-mapped persistence layer for H2Q Manifold Knots.

### 📄 h2q/decision/dde.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [STABLE] DiscreteDecisionEngine (DDE)
  - *Methods*: __init__, forward, compute_spectral_shift, apply_fractal_differential...

### 📄 h2q/dna_topology/topology_engine.py
- **Class `DNAQuaternionMapper`**
  - *Doc*: Maps ATCG sequences to SU(2) manifold elements (Quaternions).
- **Class `FractalExpansion`**
  - *Doc*: Projects 2-atom seeds into a 256-dimensional manifold via symmetry breaking (h ± δ).
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates η = (1/π) arg{det(S)} to identify topological invariants.
- **Class `DNATopologyAnalyzer`**
  - *Doc*: No documentation.

### 📄 h2q/physics/spectral_ops.py
- **Class `SpectralEntropyRegularizer`**
  - *Doc*: H2Q Spectral Entropy Regularizer [STABLE]
  - *Methods*: __init__, forward...
- **Class `DiscreteDecisionEngine`**
  - *Doc*: H2Q Discrete Decision Engine [REFACTORED]
  - *Methods*: __init__, forward...

### 📄 h2q/routing/dynamic_precision.py
- **Class `SpectralShiftTracker`**
  - *Doc*: Calculates η = (1/π) arg{det(S)}, linking discrete decision atoms 
  - *Methods*: __init__, forward...

### 📄 h2q/monitoring/mhdm.py
- **Class `ManifoldHeatDeathMonitor`**
  - *Doc*: Monitors the spectral health of 256-dim quaternionic knots.

### 📄 h2q/data/universal_stream.py
- **Class `UniversalStreamLoader`**
  - *Doc*: H2Q Universal Stream Loader

### 📄 h2q/data/generator.py
- **Class `H2QSyntheticEngine`**
  - *Doc*: Generates symbolic logic and synthetic reasoning traces grounded in 

### 📄 h2q/engine/discrete_decision.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: DiscreteDecisionEngine (DDE)
  - *Methods*: __init__, forward, get_spectral_shift...

### 📄 h2q/engine/decision_engine.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [STABLE] DiscreteDecisionEngine (DDE)
  - *Methods*: __init__, forward, update_spectral_shift, __repr__...

### 📄 h2q/engine/decision.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [STABLE] Standardized Discrete Decision Engine (DDE).
  - *Methods*: __init__, forward, update_temperature, __repr__...

### 📄 h2q/engine/curiosity.py
- **Class `CuriosityProposer`**
  - *Doc*: The Proposer: A background curiosity engine that generates adversarial 

### 📄 h2q/loaders/audio_knot.py
- **Class `AudioKnotLoader`**
  - *Doc*: H2Q Audio-Knot Loader

### 📄 h2q/control/ast_engine.py
- **Class `AutomatedSleepTrigger`**
  - *Doc*: AST (Automated-Sleep-Trigger) for H2Q Architecture.

### 📄 h2q/control/dmdc.py
- **Class `DynamicManifoldDepthController`**
  - *Doc*: DMDC: Dynamic Manifold Depth Controller
  - *Methods*: __init__, calculate_hdi, forward, audit_integrity...

### 📄 h2q/validation/compression_audit.py
- **Class `DiscreteDecisionEngine`**
  - *Doc*: [STABLE] Fixed implementation of the DDE.
  - *Methods*: __init__, forward...
- **Class `HierarchicalDecoder`**
  - *Doc*: [STABLE] Fractal Expansion: 2 -> 4 -> 8 ... -> 256.
  - *Methods*: __init__, forward...

## 3. 依赖关系图 (Dependency Graph)
```mermaid
graph TD
    train_zero_memory_py --> h2q
    train_spacetime_py --> h2q
    train_multilingual_decoder_py --> h2q
    test_vision_demo_py --> h2q
    train_byte_compression_py --> h2q
    train_tpq_optim_py --> h2q
    train_decoder_py --> h2q
    train_multilingual_py --> h2q
    train_synesthesia_unified_py --> h2q
    train_hierarchy_py --> h2q
    train_fractal_py --> h2q
    train_discrete_cpu_py --> h2q
    accelerate_gpt2_py --> h2q
    train_synesthesia_avtg_py --> h2q
    test_multilingual_py --> h2q
    train_vision_core_py --> h2q
    train_distillation_py --> h2q
    deploy_reasoning_vault_py --> h2q
    train_knot_py --> h2q
    run_experiment_py --> h2q
    benchmark_latency_py --> h2q
    train_reversible_vision_py --> h2q
    h2q_server_py --> h2q
    train_arithmetic_py --> h2q
    run_language_simulation_py --> h2q
    virtual_giant_test_py --> h2q
    demo_universal_zip_py --> h2q
    tools_h2q_bridge_py --> h2q
    tests_test_cem_py --> h2q
    tests_test_system_integration_py --> h2q
    tests_test_sst_py --> h2q
    tests_test_trace_formula_py --> h2q
    tests_test_cost_functional_py --> h2q
    tests_test_crystal_integration_py --> h2q
    tests_test_api_contract_py --> h2q
    tests_test_dde_py --> h2q
    h2q_system_py --> h2q
    h2q_dream_engine_py --> h2q
    h2q_production_logical_generator_py --> h2q
    h2q_kernels_py --> h2q
    h2q_train_full_stack_v2_py --> h2q
    h2q_visualization_spectral_dream_visualizer_py --> h2q
    h2q_visualization_holomorphic_path_visualizer_py --> h2q
    h2q_kernels_cmeb_py --> h2q
    h2q_kernels_resonance_tiling_py --> h2q
    h2q_core_hhk_py --> h2q
    h2q_core_ddfl_dream_connector_py --> h2q
    h2q_core_generation_py --> h2q
    h2q_core_berry_phase_sync_py --> h2q
    h2q_core_unified_orchestrator_py --> h2q
    h2q_core_interpolation_py --> h2q
    h2q_core_interface_registry_py --> h2q
    h2q_core_gter_system_py --> h2q
    h2q_core_dynamic_inference_py --> h2q
    h2q_core_synesthesia_loss_py --> h2q
    h2q_core_spectral_tuner_py --> h2q
    h2q_core_manifold_recovery_py --> h2q
    h2q_core_adapter_py --> h2q
    h2q_core_isomorphism_bridge_py --> h2q
    h2q_core_ttd_scheduler_py --> h2q
    h2q_core_homeostatic_trainer_py --> h2q
    h2q_core_cas_kernel_py --> h2q
    h2q_core_berry_cross_attenuator_py --> h2q
    h2q_core_holomorphic_controller_py --> h2q
    h2q_core_fueter_beam_search_py --> h2q
    h2q_core_bprm_py --> h2q
    h2q_core_uam_trainer_py --> h2q
    h2q_core_memory_h2q_vault_py --> h2q
    h2q_core_memory_mps_swap_py --> h2q
    h2q_core_memory_topological_forgetting_py --> h2q
    h2q_core_memory_berry_kv_cache_py --> h2q
    h2q_core_memory_rskh_vault_py --> h2q
    h2q_core_layers_usc_barycenter_py --> h2q
    h2q_core_layers_hamilton_reversible_cell_py --> h2q
    h2q_core_layers_spectral_pooling_py --> h2q
    h2q_core_accelerators_hamilton_amx_bridge_py --> h2q
    h2q_core_calibration_genomic_vision_suite_py --> h2q
    h2q_core_calibration_synesthesia_calibration_suite_py --> h2q
    h2q_core_optimization_fdc_optimizer_py --> h2q
    h2q_core_optimization_holomorphic_projection_py --> h2q
    h2q_core_optimization_holomorphic_healing_py --> h2q
    h2q_core_distillation_code_genomic_distiller_py --> h2q
    h2q_core_distillation_cmi_distiller_py --> h2q
    h2q_core_distillation_avtg_distiller_py --> h2q
    h2q_core_distillation_code_geometric_bridge_py --> h2q
    h2q_core_optimizers_fdc_optimizer_py --> h2q
    h2q_core_optimizers_hjb_solver_py --> h2q
    h2q_core_optimizers_spectral_drag_py --> h2q
    h2q_core_optimizers_spectral_drag_scheduler_py --> h2q
    h2q_core_trainers_unified_barycenter_trainer_py --> h2q
    h2q_core_trainers_holomorphic_hjb_healer_py --> h2q
    h2q_core_trainers_berry_fusion_unified_py --> h2q
    h2q_core_trainers_sleep_healer_py --> h2q
    h2q_core_audit_crosstalk_auditor_py --> h2q
    h2q_core_audit_genomic_invariant_audit_py --> h2q
    h2q_core_persistence_rskh_uqc_layer_py --> h2q
    h2q_core_persistence_l2_super_knot_py --> h2q
    h2q_core_persistence_gter_storage_py --> h2q
    h2q_core_persistence_spectral_swap_manager_py --> h2q
    h2q_core_alignment_bargmann_aligner_py --> h2q
    h2q_core_alignment_berry_phase_comparator_py --> h2q
    h2q_core_alignment_karcher_flow_aligner_py --> h2q
    h2q_core_guards_holomorphic_sparsity_guard_py --> h2q
    h2q_core_guards_hqa_guard_py --> h2q
    h2q_core_guards_holomorphic_guard_middleware_py --> h2q
    h2q_core_guards_holomorphic_beam_search_py --> h2q
    h2q_core_generation_holomorphic_backtracker_py --> h2q
    h2q_memory_geodesic_replay_py --> h2q
    h2q_layers_amx_spinor_interference_py --> h2q
    h2q_layers_amx_linear_py --> h2q
    h2q_experiments_berry_phase_interferometer_py --> h2q
    h2q_experiments_synesthesia_4way_alignment_py --> h2q
    h2q_dispatch_amx_tiling_dispatcher_py --> h2q
    h2q_logic_holomorphic_gating_unit_py --> h2q
    h2q_audit_synesthesia_4way_audit_py --> h2q
    h2q_benchmarks_zwi_geometric_crystal_py --> h2q
    h2q_benchmarks_amx_tiled_profiler_py --> h2q
    h2q_benchmarks_persistence_stress_test_py --> h2q
    h2q_benchmarks_rskh_infinite_stress_py --> h2q
    h2q_benchmarks_infinite_context_persistence_audit_py --> h2q
    h2q_governance_memory_governor_py --> h2q
    h2q_governance_heat_death_governor_py --> h2q
    h2q_governance_modality_synchronizer_py --> h2q
    h2q_ops_m4_amx_extension_py --> h2q
    h2q_ops_m4_amx_bridge_py --> h2q
    h2q_ops_rskh_mmap_swapper_py --> h2q
    h2q_engine_curiosity_py --> h2q
    h2q_control_dmdc_py --> h2q
```
