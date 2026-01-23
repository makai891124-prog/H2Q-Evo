import torch

# 尝试启用 torch.compile, 如果可用
try:
    torch.compile(model=None, dynamic=False, options=None, backend="inductor", fullgraph=False)
    compiled = True
except Exception as e:
    print(f"torch.compile not available or failed: {e}")
    compiled = False

def infer(model, inputs):
    """Inference function.  Applies torch.compile if available."""
    global compiled
    if compiled:
        # 编译模型.  因为 torch.compile 不接受 None 输入，故需要一个dummy input
        example_input = tuple(torch.randn(x.shape, device=x.device, dtype=x.dtype) if isinstance(x, torch.Tensor) else x for x in inputs)
        
        def compiled_model(*args):
            return model(*args)
        
        compiled_model = torch.compile(compiled_model, backend="inductor", fullgraph=True)

        #Warm-up
        with torch.no_grad():
            compiled_model(*example_input)
        
        with torch.no_grad():
            output = compiled_model(*inputs)
    else:
        with torch.no_grad():
            output = model(*inputs)
    return output
