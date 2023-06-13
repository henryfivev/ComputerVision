import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader
from torchvision.transforms.functional import to_pil_image

# 模型定义
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes):
        super(WideResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16 * widen_factor, depth)
        self.layer2 = self._make_layer(32 * widen_factor, depth, stride=2)
        self.layer3 = self._make_layer(64 * widen_factor, depth, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * widen_factor, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# 数据增强函数
def data_augmentation(x):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    augmented_x = torch.stack([transform(img) for img in x])
    return augmented_x

# MixMatch 核心函数
def mixmatch(model, labeled_data, unlabeled_data, num_classes, T, K, alpha):
    labeled_x, labeled_y = labeled_data
    unlabeled_x = unlabeled_data
    
    # 扩充标记数据
    augmented_labeled_x = data_augmentation(labeled_x)
    
    # 扩充未标记数据
    augmented_unlabeled_x = data_augmentation(unlabeled_x)
    
    # 模型训练（标记数据）
    labeled_logits = model(augmented_labeled_x)
    labeled_loss = F.cross_entropy(labeled_logits, labeled_y)
    
    # 模型训练（未标记数据）
    unlabeled_logits = model(augmented_unlabeled_x)
    unlabeled_probs = F.softmax(unlabeled_logits, dim=1)
    unlabeled_max_probs, unlabeled_pseudo_labels = unlabeled_probs.max(dim=1)
    
    # MixMatch 样本生成
    mask = (unlabeled_max_probs >= threshold) # 根据阈值选择样本
    mixed_inputs = torch.cat([augmented_labeled_x, augmented_unlabeled_x[mask]])
    mixed_labels = torch.cat([labeled_y, unlabeled_pseudo_labels[mask]])
    mixed_labels = F.one_hot(mixed_labels, num_classes).float()
    
    # MixMatch 训练
    mixed_logits = model(mixed_inputs)
    mixed_probs = F.softmax(mixed_logits, dim=1)
    mixed_loss = (alpha * F.kl_div(mixed_probs.log(), mixed_labels, reduction='batchmean') +
                  (1 - alpha) * F.cross_entropy(mixed_logits, mixed_labels.argmax(dim=1)))
    
    # 计算总损失
    total_loss = labeled_loss + mixed_loss
    print("loss=", total_loss)
    
    # 优化器更新
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# 定义验证函数
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# 参数设置
num_classes = 10  # 类别数
T = 0.5  # 温度参数
K = 2  # 数据增强次数
alpha = 0.75  # 损失函数权重参数
threshold = 0.95  # 选择 MixMatch 样本的阈值
batch_size = 32
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 运行主函数
if __name__ == "__main__":
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    train_labeled_idxs = torch.randperm(len(train_dataset))[:4000]  # 选取部分标记数据
    train_unlabeled_idxs = torch.randperm(len(train_dataset))[4000:]  # 剩余未标记数据
    train_labeled_dataset = Subset(train_dataset, train_labeled_idxs)
    train_unlabeled_dataset = Subset(train_dataset, train_unlabeled_idxs)
    train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 加载验证集数据
    validation_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 加载测试集数据
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型、优化器等
    model = WideResNet(depth=28, widen_factor=2, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        labeled_data_iter = iter(train_labeled_loader)
        unlabeled_data_iter = iter(train_unlabeled_loader)
        print("epoch=", epoch)
        
        for batch_idx, (labeled_x, labeled_y) in enumerate(labeled_data_iter):
            try:
                unlabeled_x = next(unlabeled_data_iter)[0]
            except StopIteration:
                unlabeled_data_iter = iter(train_unlabeled_loader)
                unlabeled_x = next(unlabeled_data_iter)[0]
            
            labeled_x, labeled_y, unlabeled_x = labeled_x.to(device), labeled_y.to(device), unlabeled_x.to(device)
            
            mixmatch(model, (labeled_x, labeled_y), unlabeled_x, num_classes, T, K, alpha)
        
        # 在验证集上进行评估
        accuracy = evaluate(model, validation_loader)
        print(f"Validation Accuracy: {accuracy}")
        
    # 最后进行模型评估或测试
    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {accuracy}")