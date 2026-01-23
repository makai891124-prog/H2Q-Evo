import torch

# 尝试使用 torch.compile 优化性能
if hasattr(torch, 'compile'):
    compile = torch.compile
else:
    def compile(model, *args, **kwargs):
        print("torch.compile not available, returning the model unchanged.")
        return model


@compile
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.tensor([w, x, y, z])


@compile
def quaternion_conjugate(q):
    w, x, y, z = q
    return torch.tensor([w, -x, -y, -z])


@compile
def quaternion_magnitude(q):
    w, x, y, z = q
    return torch.sqrt(w * w + x * x + y * y + z * z)


@compile
def quaternion_normalize(q):
    magnitude = quaternion_magnitude(q)
    if magnitude == 0:
        return torch.tensor([1.0, 0.0, 0.0, 0.0])  # Avoid division by zero
    return q / magnitude


@compile
def quaternion_to_rotation_matrix(q):
    q = quaternion_normalize(q)
    w, x, y, z = q
    rotation_matrix = torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ])
    return rotation_matrix


@compile
def create_fractal(width, height, max_iter):
    x, y = torch.meshgrid(torch.linspace(-2, 1, width), torch.linspace(-1.5, 1.5, height), indexing='xy')
    c = torch.complex(x, y)
    z = torch.zeros_like(c)
    fractal = torch.zeros(width, height, dtype=torch.int32)

    for i in range(max_iter):
        mask = torch.abs(z) < 2
        z[mask] = z[mask] * z[mask] + c[mask]
        fractal[mask] = i

    return fractal