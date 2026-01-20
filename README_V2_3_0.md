# H2Q-Evo v2.3.0 - Local Learning System

H2Q-Evo v2.3.0 is a self-contained, autonomous agent system with local learning capabilities, checkpoint migration, and continuous knowledge evolution.

## Quick Start

### 1. Installation

```bash
# Install in development mode with CLI entry point
pip install -e .

# Or with specific v2.3.0 dependencies
pip install -r requirements_v2_3_0.txt
```

### 2. Initialize Agent

```bash
h2q init
# Creates ~/.h2q-evo/ with knowledge base, checkpoints, and metrics
```

### 3. Execute Your First Task

```bash
# Run a task and see inference result
h2q execute "What is the capital of France?"

# Run task and save experience to knowledge base
h2q execute "Calculate: 2 + 2" --save-knowledge

# Use specific inference strategy
h2q execute "Solve quadratic equation" --strategy math
```

### 4. Check Agent Status

```bash
# View accumulated knowledge and metrics
h2q status

# Output shows:
# - Total experiences in knowledge base
# - Domain coverage (math, logic, general)
# - Overall success rate and execution history
```

### 5. Save and Migrate Agent

```bash
# Export checkpoint (portable across devices)
h2q export-checkpoint ~/backup/agent_v1.ckpt

# Import checkpoint on another machine
h2q import-checkpoint ~/backup/agent_v1.ckpt

# Verify integrity
h2q export-checkpoint ~/backup/agent_v2.ckpt --verify
```

## System Architecture

### Five-Layer Design

```
┌─────────────────────────────────────────────────────┐
│  Layer 5: Migration & Sync                          │
│  (Cross-device checkpoint protocol)                 │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│  Layer 4: Autonomous Management                     │
│  (Self-adaptive resource management)                │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│  Layer 3: Knowledge Evolution                       │
│  (SQLite DB + experience graph)                     │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: Inference Engine                          │
│  (H2Q core + learning hooks)                        │
└─────────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────────┐
│  Layer 1: Container (Self-contained executable)     │
└─────────────────────────────────────────────────────┘
```

### Core Modules

- **CLI Layer** (`h2q_cli/`): Command-line interface with Click framework
- **Execution Layer**: Task execution with learning integration
- **Knowledge Layer**: SQLite experience database with indexing
- **Persistence Layer**: Checkpoint creation, migration, and integrity checking
- **Monitoring Layer**: Metrics tracking and execution history

## Key Features

### 1. **Local Knowledge Base**
- SQLite database stores experiences (task, result, feedback, timestamp)
- Domain-aware indexing (math, logic, general)
- Automatic experience retrieval for similar future tasks
- Query statistics: `get_stats()` returns domain coverage

### 2. **Autonomous Learning**
- **Feedback Normalization**: Convert user feedback to [-0.5, 1.0] learning signals
- **Strategy Selection**: Agent learns which strategy works best for each task type
- **Metrics Tracking**: Exponential moving average success rate
- **Safe Weight Updates**: Learning loop prepared for future model fine-tuning

### 3. **Checkpoint Migration**
- Full state serialization: knowledge.db + metrics.json + config.json
- SHA256 checksum for integrity verification
- Pickle-based binary format for efficient storage
- Restore on any device: `h2q import-checkpoint file.ckpt`

### 4. **Metrics & Observability**
- Total tasks executed
- Success rate (EMA: 0.95*old + 0.05*new)
- Execution history with timestamps
- Domain coverage statistics
- Persistent JSON-based metrics

## Configuration

### Environment Variables

```bash
# Override default agent home directory
export H2Q_AGENT_HOME=/custom/path/to/agent

# Inference backend (API or local)
export H2Q_INFERENCE_MODE=local  # or 'api'

# API key for remote inference
export GEMINI_API_KEY=your-key-here
```

### Configuration File

Agent configuration stored in `~/.h2q-evo/config.json`:

```json
{
  "version": "2.3.0",
  "agent_id": "unique-uuid",
  "inference_mode": "local",
  "max_checkpoint_size": "500MB",
  "knowledge_retention_days": 365,
  "default_strategy": "auto"
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_v2_3_0_cli.py::TestLocalExecutor -v

# With coverage
pytest tests/ --cov=h2q_project --cov-report=html
```

### Running E2E Smoke Test

```bash
# Automated end-to-end validation
PYTHONPATH=. python -m h2q_project.tools.smoke_cli
```

This runs:
1. `h2q init` - Initialize agent structure
2. `h2q execute` - Run inference task
3. `h2q status` - Display metrics
4. `h2q export-checkpoint` - Create checkpoint

### Development Server

```bash
# Start inference server (development mode)
PYTHONPATH=. python3 -m uvicorn h2q_project.h2q_server:app --reload --host 0.0.0.0 --port 8000
```

## Performance Metrics

### Baseline (Single Task Execution)
- Inference latency: ~100-500ms (local) | ~1-2s (API)
- Knowledge lookup: O(1) indexed by task_type
- Checkpoint save: ~10-50ms (SQLite dump + JSON serialization)
- Checkpoint restore: ~20-100ms

### Scalability
- Knowledge base: Supports 1M+ experiences (SQLite scales to GBs)
- Checkpoint size: ~1-10MB per 100k experiences
- Memory footprint: ~100-200MB (excluding model weights)

## Troubleshooting

### Issue: `h2q: command not found`
**Solution**: Reinstall in development mode
```bash
pip install -e .
```

### Issue: `FileNotFoundError: ~/.h2q-evo`
**Solution**: Initialize agent home
```bash
h2q init
```

### Issue: Knowledge database corruption
**Solution**: Restore from latest checkpoint
```bash
h2q import-checkpoint latest_backup.ckpt
```

### Issue: Checkpoint too large
**Solution**: Archive old experiences
```bash
# (Future feature) Compress experiences older than 30 days
```

## Migration Guide (v2.2.0 → v2.3.0)

### For Existing Users

1. **Backup Current State**
   ```bash
   h2q export-checkpoint ~/backup/v2_2_0.ckpt
   ```

2. **Install v2.3.0**
   ```bash
   pip install -e .
   ```

3. **Initialize New Agent**
   ```bash
   h2q init
   ```

4. **Resume Learning (Optional)**
   ```bash
   h2q import-checkpoint ~/backup/v2_2_0.ckpt
   ```

### Key Differences
- v2.2.0: Stateless inference only
- v2.3.0: Stateful learning with knowledge persistence
- Backward compatible: v2.2.0 checkpoints can be imported

## Roadmap (Future Milestones)

### v2.3.1 (Next 2 weeks)
- [ ] Pytest suite coverage >90%
- [ ] Performance profiling & optimization
- [ ] Docker containerization for local execution

### v3.0 (Next month)
- [ ] Model weight fine-tuning integration
- [ ] Multi-GPU support
- [ ] Distributed checkpoint sync
- [ ] Advanced strategy optimization

### v3.1 (Future)
- [ ] Federated learning with multiple agents
- [ ] Knowledge graph visualization
- [ ] Real-time metrics dashboard

## Contributing

To contribute to v2.3.0:

1. Create feature branch: `git checkout -b feature/your-feature`
2. Follow existing code style (type hints, docstrings, tests)
3. Run tests: `pytest tests/ -v`
4. Submit PR with description

## License & Attribution

Built on H2Q-Evo v2.2.0 core system. See LICENSE file for details.

## Support

For questions or issues:
- Check [troubleshooting](#troubleshooting) section
- Review example scripts in `h2q_project/`
- Open GitHub issue with detailed logs

---

**Version**: 2.3.0 MVP  
**Last Updated**: 2025  
**Status**: Ready for acceptance testing ✅
