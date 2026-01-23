import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="H2Q Training Script")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="simple_lstm", help="Name of the model to use.")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension of word embeddings.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimension of hidden states in RNNs.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in RNNs.")

    # Data arguments
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--val_file", type=str, default=None, help="Path to the validation data file.")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum sequence length.")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--save_path", type=str, default="checkpoints/model.pt", help="Path to save the trained model.")

    return parser.parse_args()