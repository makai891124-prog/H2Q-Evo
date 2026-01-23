import torch
import onnx
import onnxruntime as ort


def optimize_model_with_onnx(model, dummy_input, onnx_path="model.onnx"):
    """Optimizes a PyTorch model using ONNX.

    Args:
        model: PyTorch model to optimize.
        dummy_input: A sample input to trace the model.
        onnx_path: Path to save the ONNX model.

    Returns:
        Path to the optimized ONNX model.
    """
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,  # Choose an appropriate opset version
        do_constant_folding=True,
        input_names=['input'],  # Provide input names
        output_names=['output']  # Provide output names
    )

    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)

    # Optionally, perform further optimizations with ONNX Runtime
    # This might require installation: pip install onnxruntime
    # The following is commented out to avoid requiring onnxruntime
    # and because its effectiveness depends on the model

    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # # Create an inference session with the optimized model
    # session = ort.InferenceSession(onnx_path, sess_options=sess_options)

    # Verify the model (optional, but recommended)
    onnx.checker.check_model(onnx_model)

    return onnx_path


if __name__ == '__main__':
    # Example usage:
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    dummy_input = torch.randn(1, 10)
    onnx_model_path = optimize_model_with_onnx(model, dummy_input)
    print(f"ONNX model saved to: {onnx_model_path}")
