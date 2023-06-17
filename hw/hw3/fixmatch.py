import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from torchvision.models import wide_resnet50_2

# 设置随机种子
torch.manual_seed(42)

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

# 定义弱增强函数
def weak_augmentation(image):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image)

# 定义强增强函数
def strong_augmentation(image):
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image)

# 定义FixMatch训练函数
def fixmatch_train(model, labeled_loader, unlabeled_loader, optimizer, device):
    model.train()
    print("start train")
    
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    num_iterations = min(len(labeled_iter), len(unlabeled_iter))
    # print("get data iterator")

    for i in range(num_iterations):
        # 加载有标签数据
        labeled_images, labels = next(labeled_iter)
        labeled_images, labels = labeled_images.to(device), labels.to(device)
        # print("load labeled data")
        
        # 加载无标签数据
        unlabeled_images, _ = next(unlabeled_iter)
        unlabeled_images = unlabeled_images.to(device)
        # print("load unlabeled data")

        # 生成强增强数据和预测
        strong_aug_images = torch.stack([strong_augmentation(image) for image in unlabeled_images]).to(device)
        outputs = model(strong_aug_images)
        _, pseudo_labels = torch.max(outputs.detach(), dim=1)
        # print("get predict")
        
        # 计算伪标签的置信度
        confidence = torch.max(torch.softmax(outputs.detach(), dim=1), dim=1)[0]
        mask = confidence.ge(0.95).float()
        # print("get confidence threshold")

        # 有标签损失
        labeled_outputs = model(labeled_images)
        labeled_loss = nn.CrossEntropyLoss()(labeled_outputs, labels)
        # print("get labeled loss")
        
        # 无标签损失
        pseudo_outputs = model(strong_aug_images)
        pseudo_loss = nn.CrossEntropyLoss(reduction='none')(pseudo_outputs, pseudo_labels)
        unlabeled_loss = (mask * pseudo_loss).mean()
        # print("get unlabeled loss")
        
        # 总损失
        loss = labeled_loss + unlabeled_loss
        print("loss=", loss) 

        # 更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("update model")

# 定义验证函数
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载CIFAR-10数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 划分有标签、无标签、验证和测试数据集
    labeled_indices = torch.arange(5000)
    unlabeled_indices = torch.arange(5000, len(train_dataset))
    val_indices = torch.arange(10000, 15000)
    
    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices)
    val_dataset = Subset(train_dataset, val_indices)
    
    # 创建数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 创建模型
    model = WideResNet(depth=28, widen_factor=2, num_classes=10).to(device)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 进行FixMatch训练
    for epoch in range(25):
        fixmatch_train(model, labeled_loader, unlabeled_loader, optimizer, device)
        
        # 在每个epoch结束后评估模型性能
        val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.2f}%")
    
    # 在测试集上评估模型性能
    test_accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), './save_model/fixmatch_model.pt')

if __name__ == '__main__':
    main()

    print("记录loss")
    # 计算准确率done
    # 修改网络架构done