import numpy as np
from h2q_project.core.quaternion_geometry import Quaternion

class TrainingMonitor:
    def __init__(self, threshold=1e-5, patience=5):
        self.threshold = threshold
        self.patience = patience
        self.loss_history = []
        self.gradient_norms = []
        self.consecutive_increases = 0

    def log_loss(self, loss):
        self.loss_history.append(loss)

    def log_gradient_norm(self, gradient_norm):
        self.gradient_norms.append(gradient_norm)

    def check_gradient_explosion(self):
        if len(self.gradient_norms) < 2:
            return False
        
        if self.gradient_norms[-1] > 10 * np.mean(self.gradient_norms[:-1]): # Increased sensitivity.
            return True
        return False

    def check_loss_oscillation(self):
        if len(self.loss_history) < 3:
            return False

        last_three_losses = self.loss_history[-3:]
        if last_three_losses[1] > last_three_losses[0] and last_three_losses[1] > last_three_losses[2]:
             # Loss increased then decreased, indicative of oscillation.  Slightly more tolerant
             return True
        return False

    def check_training_stalled(self):
        if len(self.loss_history) < self.patience:
            return False

        recent_losses = self.loss_history[-self.patience:]
        loss_diffs = np.diff(recent_losses)
        
        if all(abs(diff) < self.threshold for diff in loss_diffs):
            self.consecutive_increases += 1
            if self.consecutive_increases >= self.patience / 2:
                return True  # Training stalled
        else:
            self.consecutive_increases = 0
        return False

    def get_status(self):
        status = {}
        status['gradient_explosion'] = self.check_gradient_explosion()
        status['loss_oscillation'] = self.check_loss_oscillation()
        status['training_stalled'] = self.check_training_stalled()
        return status

    def suggest_action(self):
        status = self.get_status()
        if status['gradient_explosion']:
            return "Reduce learning rate, clip gradients."
        elif status['loss_oscillation']:
            return "Reduce learning rate, increase batch size, or try a different optimizer."
        elif status['training_stalled']:
            return "Increase learning rate, add regularization, or modify network architecture."
        else:
            return "No immediate action needed."
