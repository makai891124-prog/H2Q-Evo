import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from h2q_project.knot_net import KnotNet  # 假设KnotNet定义在 knot_net.py 中
from torch.profiler import profile, record_function, ProfilerActivity

# 定义超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化 KnotNet 模型
model = KnotNet(num_layers=3, in_channels=1, num_classes=10).to(DEVICE)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 训练循环
def train():
    model.train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/knot')) as prof:
        for epoch in range(NUM_EPOCHS):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # 前向传播
                with record_function("KnotNet Forward Pass"):  # 添加record_function
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # 反向传播和优化
                with record_function("KnotNet Backward Pass"):  # 添加record_function
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if (i+1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                    prof.step()  # 记录步骤


# 测试循环
def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

if __name__ == '__main__':
    train()
    test()