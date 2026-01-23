import torch

class BezierCurve:
    def __init__(self, control_points):
        self.control_points = torch.nn.Parameter(torch.tensor(control_points, dtype=torch.float32, requires_grad=True))

    def forward(self, t):
        n = self.control_points.shape[0] - 1
        curve_point = torch.zeros_like(self.control_points[0])
        for i in range(n + 1):
            bernstein_polynomial = self.bernstein_polynomial(n, i, t)
            curve_point += bernstein_polynomial * self.control_points[i]
        return curve_point

    def bernstein_polynomial(self, n, i, t):
        return torch.special.comb(n, i) * (t**i) * ((1 - t)**(n - i))

    def compute_gradients(self, t):
        curve_point = self.forward(t)
        curve_point.backward()
        return self.control_points.grad.clone()

    def reset_gradients(self):
        if self.control_points.grad is not None:
            self.control_points.grad.zero_()


if __name__ == '__main__':
    # Example usage and gradient visualization
    import matplotlib.pyplot as plt

    # Define control points
    control_points = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 0.0]]
    bezier_curve = BezierCurve(control_points)

    # Generate points along the curve
    t_values = torch.linspace(0, 1, 100)
    curve_points = torch.stack([bezier_curve.forward(t) for t in t_values])

    # Visualize the curve
    plt.plot(curve_points[:, 0].detach().numpy(), curve_points[:, 1].detach().numpy(), label='Bezier Curve')
    plt.plot([p[0] for p in control_points], [p[1] for p in control_points], 'ro-', label='Control Points')

    # Compute gradients at a specific t value
    t_value = torch.tensor(0.5, requires_grad=True)
    bezier_curve.reset_gradients()
    gradients = bezier_curve.compute_gradients(t_value)

    # Visualize gradients (as vectors originating from control points)
    scale = 0.1  # Adjust scale for better visualization
    for i, cp in enumerate(control_points):
        grad = gradients[i].detach().numpy()
        plt.arrow(cp[0], cp[1], grad[0] * scale, grad[1] * scale, head_width=0.05, head_length=0.1, fc='g', ec='g', label=f'Gradient {i+1}' if i == 0 else None)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bezier Curve with Gradient Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()
