import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class H2QVisualizer:
    """
    H2Q Spectral Visualizer
    
    Maps the 256-dimensional quaternionic manifold states (the 'Dream') 
    into a 2D visual representation of the SU(2) geodesic flow.
    
    STABILITY: Experimental
    GROUNDING: SU(2) Eigenvalue Symmetry
    """

    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_spectral_dream(self, eigenvalues, eta, step=0):
        """
        Generates 'dream_spectrum.png' representing the geometric shape of the thought process.
        
        Args:
            eigenvalues (torch.Tensor): Complex eigenvalues of the SU(2) manifold (N, 2).
            eta (float): The Spectral Shift Tracker value (η).
            step (int): Current training/inference step.
        """
        # Ensure data is on CPU for matplotlib
        if torch.is_tensor(eigenvalues):
            ev_np = eigenvalues.detach().cpu().numpy()
        else:
            ev_np = np.array(eigenvalues)

        # Flatten and extract real/imaginary components
        # In SU(2), eigenvalues are complex pairs on the unit circle
        real = ev_np.real.flatten()
        imag = ev_np.imag.flatten()

        plt.figure(figsize=(10, 10), facecolor='black')
        ax = plt.subplot(111, projection='polar')
        ax.set_facecolor('black')

        # Calculate angles and magnitudes for polar projection
        angles = np.angle(real + 1j*imag)
        magnitudes = np.abs(real + 1j*imag)

        # Plot the Geodesic Flow (The Dream)
        # We use a scatter with alpha to represent the 'Fractal Expansion'
        scatter = ax.scatter(angles, magnitudes, c=angles, cmap='hsv', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        
        # Draw the Unit Circle (The SU(2) Boundary)
        circle_theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(circle_theta, np.ones_like(circle_theta), color='cyan', linestyle='--', alpha=0.3, label='SU(2) Boundary')

        # Metadata Overlay
        plt.title(f"H2Q Spectral Dream | η: {eta:.4f} | Step: {step}", color='white', fontsize=14, pad=20)
        ax.tick_params(colors='white', alpha=0.5)
        ax.grid(True, color='white', alpha=0.1)

        # Save the spectrum
        file_name = f"dream_spectrum_step_{step}.png" if step > 0 else "dream_spectrum.png"
        save_path = os.path.join(self.output_dir, file_name)
        
        plt.savefig(save_path, facecolor='black', edgecolor='none', bbox_inches='tight')
        plt.close()
        
        return save_path

if __name__ == "__main__":
    # Mock data for verification of the Veracity Compact
    # Simulating 128 pairs of SU(2) eigenvalues (256 total)
    phases = np.random.uniform(0, 2*np.pi, 128)
    mock_ev = np.concatenate([np.exp(1j * phases), np.exp(-1j * phases)])
    mock_eta = (1.0 / np.pi) * np.angle(np.linalg.det(np.eye(2)))
    
    viz = H2QVisualizer()
    path = viz.plot_spectral_dream(mock_ev, mock_eta)
    print(f"[VERIFICATION] Visualizer output generated at: {path}")
