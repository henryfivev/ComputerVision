import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models.segmentation import fcn_resnet50
def get_dataloader(batch_size, num_workers):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = VOCSegmentation(root='./data', year='2007', image_set='train', download=True, transform=transform)
    testset = VOCSegmentation(root='./data', year='2007', image_set='val', download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.model = fcn_resnet50(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)['out']
        return x
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)

# 运行主函数
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(0)

    # 定义超参数
    batch_size = 4
    num_workers = 4
    learning_rate = 0.001
    num_epochs = 10
    num_classes = 21  # VOC数据集中有20个物体类别 + 背景

    # 获取数据加载器
    trainloader, testloader = get_dataloader(batch_size, num_workers)

    # 创建模型
    model = FCN8s(num_classes)

    # 将模型移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和测试循环
    for epoch in range(num_epochs):
        train_loss = train(model, trainloader, criterion, optimizer, device)
        test_loss = test(model, testloader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "fcn8s_model.pth")
