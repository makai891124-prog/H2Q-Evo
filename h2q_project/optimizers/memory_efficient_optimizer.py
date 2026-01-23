import torch
from torch.optim import Optimizer


class MemoryEfficientOptimizer(Optimizer):
    def __init__(self, params, optimizer_class, grad_accumulation_steps=1, **kwargs):
        defaults = dict(grad_accumulation_steps=grad_accumulation_steps, **kwargs)
        super().__init__(params, defaults)

        self.optimizer = optimizer_class(self.param_groups, **kwargs)
        self.grad_accumulation_steps = grad_accumulation_steps
        self.current_step = 0

        for group in self.param_groups:
            group['grad_accumulation_steps'] = grad_accumulation_steps

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        self.current_step += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if self.current_step % group['grad_accumulation_steps'] == 0:
                        p.grad.data.div_(group['grad_accumulation_steps'])
                    else:
                        continue

        if self.current_step % self.grad_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss

    def zero_grad(self): 
        # zero_grad is handled internally by step. Avoid double zeroing gradients
        pass
