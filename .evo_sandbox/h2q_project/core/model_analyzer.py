import torch
import torch.nn as nn

class ModelAnalyzer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_history = []

    def analyze_training(self, input_data, target_data):
        """Analyzes the model's performance during training."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
            loss = self.criterion(output, target_data)
            # Record the loss and potentially other metrics.
            self.training_history.append(loss.item())

        # Analyze for potential problems.
        gradient_vanish = self.check_gradient_vanishing()
        overfitting = self.check_overfitting(input_data, target_data)
        # Add other checks here

        suggestions = []
        if gradient_vanish:
            suggestions.append("Potential vanishing gradient issue. Consider using ReLU activation or Batch Normalization.")
        if overfitting:
            suggestions.append("Potential overfitting issue.  Consider using dropout or regularization.")

        return suggestions

    def check_gradient_vanishing(self, threshold=1e-5):
        """Checks for gradient vanishing by monitoring the gradient norms."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.norm(param.grad).item() < threshold:
                    return True
        return False

    def check_overfitting(self, input_data, target_data, threshold=0.05):
        """Basic overfitting check by comparing training and validation loss."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
            loss = self.criterion(output, target_data)
            # Compare to a previously recorded 'validation' loss, if available.
            if len(self.training_history) > 1:
                # Assuming the last loss is a 'validation' loss
                previous_loss = self.training_history[-2]
                if loss.item() > previous_loss + threshold:
                    return True
        return False

    def get_training_history(self):
        """Returns the training history (e.g., loss values)."""
        return self.training_history


    def suggest_improvements(self):
        """Suggests improvements based on analysis."""
        suggestions = []
        # Logic to analyze training history and suggest improvements
        if not self.training_history:
            return ["No training data available for analysis."]

        # Example: if loss is not decreasing, suggest reducing learning rate
        if len(self.training_history) > 5 and self.training_history[-1] >= self.training_history[-2]:
            suggestions.append("Consider reducing the learning rate.")

        return suggestions
