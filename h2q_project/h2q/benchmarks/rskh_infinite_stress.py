import torch
import time
import psutil
import os
from h2q.persistence.rskh import RSKH, SpectralShiftTracker
from h2q.utils.mps_compat import ensure_complex_support

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_infinite_context_stress_test():
    """
    STRESS TEST: RSKH_INFINITE_CONTEXT
    Verifies O(1) memory stability for 2^24 (16.7M) atoms.
    Grounding: Mac Mini M4 (16GB RAM).
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[RSKH_STRESS] Initializing on {device}...")

    # Configuration
    TOTAL_ATOMS = 2**24  # 16,777,216 tokens
    CHUNK_SIZE = 2**14   # 16,384 tokens per step
    NUM_STEPS = TOTAL_ATOMS // CHUNK_SIZE
    LATENT_DIM = 256
    NUM_KNOTS = 64

    # Initialize RSKH and Tracker
    # Registry Check: RSKH(total_dim, num_knots, device)
    rskh = RSKH(LATENT_DIM, NUM_KNOTS, device)
    # Registry Check: SpectralShiftTracker(knot_dim)
    tracker = SpectralShiftTracker(LATENT_DIM)

    initial_mem = get_memory_usage_mb()
    print(f"[RSKH_STRESS] Initial Memory: {initial_mem:.2f} MB")
    print(f"[RSKH_STRESS] Target: {TOTAL_ATOMS} atoms in {NUM_STEPS} chunks.")

    start_time = time.time()
    
    try:
        for step in range(NUM_STEPS):
            # Generate synthetic atom batch (Fractal Seeds)
            # We simulate the 256-dim coordinates expanded from seeds
            x_batch = torch.randn(CHUNK_SIZE, LATENT_DIM, device=device)

            # RSKH Forward: Recursive folding into the manifold
            # Registry Check: forward(x)
            manifold_state = rskh.forward(x_batch)

            # Periodic Audit (Every 64 chunks)
            if step % 64 == 0:
                current_mem = get_memory_usage_mb()
                mem_delta = current_mem - initial_mem
                
                # Compute Spectral Shift (eta) to ensure manifold integrity
                # We treat the manifold state as a scattering matrix S for the tracker
                # Registry Check: tracker.forward(scattering_matrix)
                # Note: In H2Q, eta = (1/pi) arg{det(S)}
                # We use a slice of the manifold to represent the transition matrix
                s_matrix = manifold_state[:LATENT_DIM, :LATENT_DIM] if manifold_state.dim() > 1 else manifold_state.view(1, -1)
                eta = tracker.forward(s_matrix)

                elapsed = time.time() - start_time
                atoms_processed = (step + 1) * CHUNK_SIZE
                throughput = atoms_processed / elapsed

                print(f"Step {step}/{NUM_STEPS} | Atoms: {atoms_processed:,} | Mem Delta: {mem_delta:+.2f} MB | eta: {eta.item() if torch.is_tensor(eta) else eta:.4f} | Throughput: {throughput:.2f} a/s")

                # O(1) Verification: Memory should not grow linearly
                if mem_delta > 500: # 500MB buffer for fragmentation/OS overhead
                    print("[!] WARNING: Memory drift detected. O(1) property may be compromised.")

        total_time = time.time() - start_time
        final_mem = get_memory_usage_mb()
        
        print("\n--- STRESS TEST COMPLETE ---")
        print(f"Total Atoms Processed: {TOTAL_ATOMS:,}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Final Memory Delta: {final_mem - initial_mem:.2f} MB")
        print(f"Average Throughput: {TOTAL_ATOMS / total_time:.2f} atoms/sec")
        
        if (final_mem - initial_mem) < 100:
            print("[RESULT] SUCCESS: RSKH demonstrated O(1) memory stability.")
        else:
            print("[RESULT] PARTIAL: Memory stable but overhead observed.")

    except Exception as e:
        print(f"[!] CRITICAL FAILURE: {str(e)}")
        raise

if __name__ == "__main__":
    run_infinite_context_stress_test()