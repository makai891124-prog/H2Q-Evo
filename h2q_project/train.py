import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import time

# 定义模型（简化版本）
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def train():
    # 超参数
    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    # TensorBoard writer
    writer = SummaryWriter('runs/fashion_mnist_experiment')

    # 数据加载
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型、优化器和损失函数
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每100个batch记录一次
            if (i + 1) % 100 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + i + 1)
                running_loss = 0.0

        # 每个epoch结束时计算准确率
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1} Accuracy: {accuracy:.2f}%')
        writer.add_scalar('accuracy', accuracy, epoch+1)

    end_time = time.time()
    training_time = end_time - start_time
    print(f'Total training time: {training_time:.2f} seconds')
    writer.add_scalar('training_time', training_time, 1)

    print('Finished Training')
    writer.close()

if __name__ == '__main__':
    train()
