import torch

from loader import load_das_token_structure


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structure_path = "das_token_structure_HuggingFaceTB__SmolLM2-360M_20260328.pt"
    das = load_das_token_structure(path=structure_path, device=device)

    # Example hidden states (replace with real hidden outputs from teacher/backbone)
    hidden = torch.randn(2, int(das.package["token_mapper_config"]["hidden_dim"]), device=device)

    logits_subset = das.map_token_logits_subset(hidden)
    print("subset logits:", tuple(logits_subset.shape))


if __name__ == "__main__":
    main()
