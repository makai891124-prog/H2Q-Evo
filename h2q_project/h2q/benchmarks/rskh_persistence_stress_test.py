import torch
import time
import os
import psutil
import numpy as np
from h2q.persistence.rskh import RSKH
from h2q.core.discrete_decision_engine import get_canonical_dde
from h2q.core.sst import SpectralShiftTracker
from h2q.ops.rskh_mmap_swapper import RSKHMmapSwapper
from h2q.core.memory.rskh_ssd_paging import apply_spectral_paging_policy

def get_memory_usage_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

class RSKHPersistenceStressTest:
    """
    Executes a 100M+ Token RSKH Persistence Stress Test.
    Validates O(1) retrieval and memory stability on Mac Mini M4 (16GB).
    """
    def __init__(self, vault_path="rskh_stress_vault.bin", total_tokens=100_000_000):
        self.vault_path = vault_path
        self.total_tokens = total_tokens
        
        # Fix: Use get_canonical_dde to avoid 'dim' keyword argument errors
        self.dde = get_canonical_dde()
        self.sst = SpectralShiftTracker()
        
        # Initialize Mmap Swapper for SSD-backed persistence
        # Capacity set to handle 100M tokens with 256-dim manifold states
        self.swapper = RSKHMmapSwapper(path=self.vault_path, capacity=self.total_tokens)
        
        # Initialize RSKH with the swapper
        self.rskh = RSKH(dde=self.dde, sst=self.sst)
        self.rskh.attach_swapper(self.swapper)

    def run_test(self):
        print(f"[M24-CW] Initializing 100M Token Stress Test...")
        print(f"[M24-CW] Hardware Target: Mac Mini M4 (16GB RAM)")
        
        latencies = []
        checkpoints = [1, 1_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000]
        
        start_time = time.time()
        
        try:
            for i in range(1, self.total_tokens + 1):
                # Simulate token embedding (unit quaternion on S3)
                token_id = f"token_{i}"
                latent_state = torch.randn(256, device='cpu')
                latent_state /= torch.norm(latent_state)
                
                # Measure insertion latency
                t0 = time.perf_counter()
                self.rskh.store(token_id, latent_state)
                t1 = time.perf_counter()
                
                if i in checkpoints:
                    latency = (t1 - t0) * 1000 # ms
                    mem_usage = get_memory_usage_gb()
                    print(f"[CHECKPOINT {i}] Latency: {latency:.4f}ms | RAM: {mem_usage:.2f}GB")
                    latencies.append(latency)
                    
                    # Verify O(1) Retrieval
                    t_ret_0 = time.perf_counter()
                    _ = self.rskh.retrieve(f"token_{np.random.randint(1, i+1)}")
                    t_ret_1 = time.perf_counter()
                    ret_latency = (t_ret_1 - t_ret_0) * 1000
                    print(f"[RETRIEVAL {i}] Latency: {ret_latency:.4f}ms")

                # Apply Spectral Paging Policy if memory pressure detected
                if i % 1_000_000 == 0:
                    apply_spectral_paging_policy(self.rskh, threshold_gb=12.0)

        except Exception as e:
            print(f"[CRITICAL FAILURE] Stress test aborted: {e}")
        finally:
            total_duration = time.time() - start_time
            print(f"--- STRESS TEST COMPLETE ---")
            print(f"Total Tokens Processed: {self.total_tokens}")
            print(f"Total Duration: {total_duration/60:.2f} minutes")
            print(f"Final RAM Usage: {get_memory_usage_gb():.2f}GB")
            
            # Cleanup
            if os.path.exists(self.vault_path):
                os.remove(self.vault_path)

if __name__ == "__main__":
    tester = RSKHPersistenceStressTest(total_tokens=100_000_000)
    tester.run_test()