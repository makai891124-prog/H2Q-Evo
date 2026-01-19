import time
import threading
import torch
import logging
from typing import Optional

# --- RIGID CONSTRUCTION: ATOMIC IMPORTS ---
from h2q.memory.geodesic_replay import GeodesicTraceHealer
from h2q.core.memory.rskh_vault import RSKHVault
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeodesicHealerService")

class GeodesicHealerService:
    """
    Background micro-service for h2q_server.
    Performs autonomous 'Sleep Phase' manifold consolidation of context knots.
    """
    def __init__(self, vault_path: str = "vault/rskh_main.crystal", device: str = "mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        
        # 1. Initialize RSKH Vault (Memory Atom)
        self.vault = RSKHVault(vault_path=vault_path)
        
        # 2. Initialize DDE (Decision Atom) - FIX: Using canonical getter to avoid 'dim' error
        self.dde = get_canonical_dde()
        
        # 3. Initialize SST (Monitoring Atom)
        self.sst = SpectralShiftTracker()
        
        # 4. Initialize Healer (Logic Atom)
        self.healer = GeodesicTraceHealer(dde=self.dde, device=self.device)
        
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Starts the background healing loop."""
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._autonomous_loop, daemon=True)
        self._thread.start()
        logger.info("GeodesicTraceHealer Service: ACTIVE (Background)")

    def stop(self):
        """Graceful shutdown of the service."""
        self.is_running = False
        if self._thread:
            self._thread.join()
        logger.info("GeodesicTraceHealer Service: TERMINATED")

    def _autonomous_loop(self):
        """
        The 'Sleep Phase' logic: Polls the vault for un-consolidated knots
        and applies geodesic flow to minimize manifold residuals.
        """
        while self.is_running:
            try:
                # ELASTIC WEAVING: Query the Void if vault is idle
                knots = self.vault.retrieve_unconsolidated(limit=64)
                
                if not knots or len(knots) == 0:
                    time.sleep(10) # Idle wait
                    continue

                logger.info(f"Sleep Phase: Healing {len(knots)} context knots...")
                
                # Perform Geodesic Consolidation
                healed_knots, eta_shift = self.healer.heal_trace(knots)
                
                # Update Spectral Shift Tracker
                self.sst.update(eta_shift)
                
                # Commit healed knots back to RSKH Vault
                self.vault.commit_consolidated(healed_knots)
                
                logger.info(f"Consolidation Complete. Spectral Shift η: {eta_shift:.4f}")
                
                # M4 Constraint: Prevent thermal throttling by yielding
                time.sleep(1)

            except Exception as e:
                # EMBRACE NOISE: Log error as a boundary data point
                logger.error(f"Manifold Tear in Healing Loop: {str(e)}")
                time.sleep(5)

if __name__ == "__main__":
    # Standalone deployment test
    service = GeodesicHealerService()
    service.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()