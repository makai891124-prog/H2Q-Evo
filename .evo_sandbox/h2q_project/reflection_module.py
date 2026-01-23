class ReflectionModule:
    def reflect(self, loss, epoch_time):
        """Analyzes the training process and suggests improvements."""
        if loss > 0.5:
            if epoch_time > 60:
                return "Consider reducing batch size or simplifying the model architecture to improve training speed and reduce loss."
            else:
                return "Consider increasing the learning rate to reduce loss."
        else:
            if epoch_time > 60:
                return "Consider increasing the model complexity or adding more data, even with the current training speed and low loss. "
            else:
                return "Training is going well. Consider early stopping or further hyperparameter tuning."
