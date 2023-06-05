import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WideResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(WideResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.wide_block1 = self._wide_block(16, 16, stride=1)
        self.wide_block2 = self._wide_block(16, 32, stride=2)
        self.wide_block3 = self._wide_block(32, 64, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _wide_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.wide_block1(out)
        out = self.wide_block2(out)
        out = self.wide_block3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
def sharpen(probabilities, temperature=0.5):
    sharpened = probabilities.pow(1 / temperature)
    sharpened /= sharpened.sum(dim=1, keepdim=True)
    return sharpened

def mixup(x, y, alpha):
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    lam = torch.tensor(np.random.beta(alpha, alpha)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y



def train_model(model, labeled_dataset, unlabeled_dataset, num_epochs=50, batch_size=64, learning_rate=0.002, alpha=1.0, temperature=0.5):
    model.to(device)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def sharpen(probabilities, temperature):
        sharpened_probs = probabilities ** (1 / temperature)
        sharpened_probs /= sharpened_probs.sum(dim=1, keepdim=True)
        return sharpened_probs

    for epoch in range(num_epochs):
        model.train()
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        total_loss = 0.0
        total_samples = 0

        for i, (labeled_images, labeled_labels) in enumerate(labeled_iter):
            try:
                unlabeled_images, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_images, _ = next(unlabeled_iter)

            labeled_images = labeled_images.to(device)
            labeled_labels = labeled_labels.to(device)
            unlabeled_images = unlabeled_images.to(device)

            mixed_images, mixed_labels = mixup(unlabeled_images, labeled_labels, alpha)

            optimizer.zero_grad()

            logits = model(mixed_images)
            labeled_logits = logits[:labeled_images.size(0)]
            unlabeled_logits = logits[labeled_images.size(0):]

            labeled_loss = criterion(labeled_logits, mixed_labels)
            unlabeled_probs = torch.softmax(unlabeled_logits.detach() / temperature, dim=1)
            unlabeled_probs_sharpened = sharpen(unlabeled_probs, temperature)
            unlabeled_loss = torch.mean((unlabeled_probs_sharpened - unlabeled_probs) ** 2)

            loss = labeled_loss + unlabeled_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * mixed_images.size(0)
            total_samples += mixed_images.size(0)

        epoch_loss = total_loss / total_samples
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# 将训练集划分为标注数据和非标注数据
labeled_indices = np.arange(50000)[:4000]
unlabeled_indices = np.arange(50000)[4000:]
labeled_dataset = Subset(train_dataset, labeled_indices)
unlabeled_dataset = Subset(train_dataset, unlabeled_indices)

# 测试模型
def test_model(model, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    # 创建 WideResNet-28-2 模型实例
    model = WideResNet()

    # 在标注数据和非标注数据上进行 MixMatch 训练
    trained_model = train_model(model, labeled_dataset, unlabeled_dataset, num_epochs=50, batch_size=64, learning_rate=0.002, alpha=1.0, temperature=0.5)

    # 在测试集上评估模型
    test_model(trained_model, test_dataset)
