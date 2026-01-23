import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, num_samples, input_size):
        self.num_samples = num_samples
        self.input_size = input_size
        self.inputs = torch.randn(num_samples, input_size)
        self.labels = torch.randn(num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

if __name__ == '__main__':
    # Example Usage
    dataset = SimpleDataset(num_samples=100, input_size=10)
    print(f'Dataset size: {len(dataset)}')
    input_sample, label_sample = dataset[0]
    print(f'Input sample shape: {input_sample.shape}')
    print(f'Label sample shape: {label_sample.shape}')
