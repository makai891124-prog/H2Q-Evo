#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""API接口调查脚本"""

import inspect
import sys
sys.path.insert(0, 'h2q_project')

print("=" * 80)
print("API 接口调查")
print("=" * 80)

# 1. LatentConfig
try:
    from h2q.core.engine import LatentConfig
    print("\n✅ LatentConfig:")
    print(f"   签名: {inspect.signature(LatentConfig.__init__)}")
except Exception as e:
    print(f"\n❌ LatentConfig: {e}")

# 2. AutonomousSystem
try:
    from h2q.system import AutonomousSystem
    print("\n✅ AutonomousSystem:")
    print(f"   签名: {inspect.signature(AutonomousSystem.__init__)}")
except Exception as e:
    print(f"\n❌ AutonomousSystem: {e}")

# 3. get_canonical_dde
try:
    from h2q.core.discrete_decision_engine import get_canonical_dde
    print("\n✅ get_canonical_dde:")
    print(f"   签名: {inspect.signature(get_canonical_dde)}")
except Exception as e:
    print(f"\n❌ get_canonical_dde: {e}")

# 4. KnowledgeDB
try:
    from h2q_project.knowledge.knowledge_db import KnowledgeDB
    print("\n✅ KnowledgeDB:")
    print(f"   签名: {inspect.signature(KnowledgeDB.__init__)}")
except Exception as e:
    print(f"\n❌ KnowledgeDB: {e}")

# 5. LocalExecutor
try:
    sys.path.insert(0, '.')
    from h2q_project.local_executor import LocalExecutor
    print("\n✅ LocalExecutor:")
    print(f"   签名: {inspect.signature(LocalExecutor.__init__)}")
except Exception as e:
    print(f"\n❌ LocalExecutor: {e}")

print("\n" + "=" * 80)
