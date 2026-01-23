import torch


def optimize_model_for_inference(model):
    model.eval()
    # 1. Disable gradients
    for param in model.parameters():
        param.requires_grad = False

    # 2. Use torch.no_grad() context
    # This is done in the inference code

    # 3. Try torch.compile (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model) # Requires PyTorch 2.0+
        except Exception as e:
            print(f"torch.compile failed: {e}")
            pass

    # 4. Quantization (example, needs further setup and calibration)
    # This is a more advanced optimization and needs careful consideration
    # and a representative dataset for calibration.

    return model


