import threading
import time
import torch
import logging
from typing import Optional, Callable

# H2Q Internal Imports based on Registry
from h2q.data.generator import H2QSyntheticEngine
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.sst import SpectralShiftTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("H2Q-Curiosity")

class CuriosityProposer:
    """
    The Proposer: A background curiosity engine that generates adversarial 
    logical atoms to probe the boundaries of the H2Q manifold.
    """
    def __init__(
        self, 
        trainer_callback: Callable,
        idle_threshold_seconds: float = 5.0,
        confidence_threshold: float = 0.85,
        logic_curvature_limit: float = 0.05
    ):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.synthetic_engine = H2QSyntheticEngine()
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()
        
        self.trainer_callback = trainer_callback
        self.idle_threshold = idle_threshold_seconds
        self.confidence_threshold = confidence_threshold
        self.curvature_limit = logic_curvature_limit
        
        self._stop_event = threading.Event()
        self._last_activity_time = time.time()
        self._thread: Optional[threading.Thread] = None

    def mark_activity(self):
        """Update the last activity timestamp to prevent curiosity during active inference."""
        self._last_activity_time = time.time()

    def is_idle(self) -> bool:
        return (time.time() - self._last_activity_time) > self.idle_threshold

    def _run_curiosity_loop(self):
        logger.info("Curiosity Proposer thread started.")
        while not self._stop_event.is_set():
            if self.is_idle():
                try:
                    # 1. Generate Adversarial Question (High Entropy Logic Atom)
                    # We use the synthetic engine to generate a 'knot' that is mathematically valid but novel
                    adversarial_atom = self.synthetic_engine.generate_high_entropy_atom()
                    
                    # 2. Feed into Discrete Decision Engine
                    # The DDE evaluates the atom against the current manifold state
                    with torch.no_grad():
                        decision_output = self.dde(adversarial_atom.to(self.device))
                    
                    # 3. Audit Logical Integrity (Fueter Operator / Curvature)
                    # Logic curvature > 0.05 indicates a non-analytic hallucination or 'topological tear'
                    curvature = decision_output.get("logic_curvature", 0.0)
                    confidence = decision_output.get("confidence", 1.0)
                    
                    # 4. Trigger Training if the system is 'confused' or 'hallucinating'
                    if confidence < self.confidence_threshold or curvature > self.curvature_limit:
                        logger.info(f"Low confidence ({confidence:.2f}) or high curvature ({curvature:.2f}) detected. Triggering training loop.")
                        self.trainer_callback(adversarial_atom)
                        
                except Exception as e:
                    logger.error(f"Curiosity Loop Error: {e}")
            
            # Adaptive sleep to prevent CPU thrashing on M4
            time.sleep(2.0)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_curiosity_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

# Experimental: Integration with H2Q Server Idle State
def initialize_curiosity_engine(trainer_fn: Callable):
    """
    Factory function to instantiate the Proposer.
    STABLE: Uses canonical DDE and Synthetic Engine.
    """
    proposer = CuriosityProposer(trainer_callback=trainer_fn)
    proposer.start()
    return proposer
