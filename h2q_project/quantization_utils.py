import torch

def maybe_half(tensor, use_fp16=False, use_bf16=False):
    if use_fp16:
        return tensor.half()
    elif use_bf16:
        return tensor.bfloat16()
    else:
        return tensor


def cast_model_to_fp16(model):
    model.half()
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.float()

def cast_model_to_bf16(model):
    model.bfloat16()
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.float()
