"""
H2Q-Evo: Unified Mathematical Architecture for AGI
====================================================

Package Structure:
- h2q_project.src.h2q: Core H2Q mathematical framework
- h2q_project.core: Legacy core modules (being migrated)
- h2q_project.server: FastAPI server components
- h2q_project.evolution: Evolution system
- h2q_project.math: Mathematical utilities

Import Guidelines:
- Use absolute imports: from h2q_project.src.h2q.core import UnifiedH2QMathematicalArchitecture
- Avoid relative imports across package boundaries
- Prefer explicit imports over wildcard imports
"""

__version__ = "2.3.0"
__author__ = "H2Q-Evo Team"

# Expose main components for easy access
try:
    from .src.h2q.core.unified_architecture import UnifiedH2QMathematicalArchitecture
    from .src.h2q.core.sst import SpectralShiftTracker
    from .src.h2q.core.discrete_decision_engine import get_canonical_dde, LatentConfig
except ImportError:
    # Fallback for incomplete migration
    pass
