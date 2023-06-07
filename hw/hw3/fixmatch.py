import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

# 设置随机种子
torch.manual_seed(42)

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
    print("get data iterator")

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

        # 更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("update model")


# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # 加载CIFAR-10数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 划分有标签和无标签数据集
    labeled_indices = torch.arange(5000)
    unlabeled_indices = torch.arange(5000, len(train_dataset))
    
    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices)
    
    # 创建数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=16, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # 创建模型
    model = Model().to(device)
    print("init model")
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 进行FixMatch训练
    for epoch in range(2):
        fixmatch_train(model, labeled_loader, unlabeled_loader, optimizer, device)
    
    # 保存模型
    torch.save(model.state_dict(), './save_model/fixmatch_model.pt')

if __name__ == '__main__':
    main()

    # 记录loss
    # 计算准确率
    # 修改网络架构