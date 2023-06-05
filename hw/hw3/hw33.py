import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import wide_resnet

# 设置随机种子
torch.manual_seed(42)

# 定义数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载 CIFAR-10 数据集
labeled_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
unlabeled_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# 设置超参数
batch_size = 64
labeled_ratio = 0.1
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 划分标记和未标记数据集
labeled_size = int(labeled_ratio * len(labeled_dataset))
unlabeled_size = len(unlabeled_dataset) - labeled_size
labeled_dataset, _ = torch.utils.data.random_split(labeled_dataset, [labeled_size, len(labeled_dataset) - labeled_size])
unlabeled_dataset, _ = torch.utils.data.random_split(unlabeled_dataset, [unlabeled_size, len(unlabeled_dataset) - unlabeled_size])

# 创建数据加载器
labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 定义模型
model = wide_resnet(pretrained=False, num_classes=10).to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# MixMatch 算法
def mixmatch_loss(inputs_x, targets_x, inputs_u, targets_u, T=0.5, K=2, alpha=0.75):
    batch_size = inputs_x.size(0)
    logits_x = model(inputs_x)
    probs_x = torch.softmax(logits_x, dim=1).unsqueeze(1)

    logits_u = [model(inputs_u[i]) for i in range(K)]
    probs_u = [torch.softmax(logit, dim=1).unsqueeze(1) for logit in logits_u]
    avg_probs_u = torch.mean(torch.cat(probs_u, dim=1), dim=1)

    loss_x = -torch.mean(torch.sum(probs_x * torch.log(probs_x), dim=1))
    loss_u = torch.mean(torch.sum((avg_probs_u - targets_u) ** 2, dim=1))

    loss = loss_x + alpha * loss_u
    return loss

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, ((inputs_x, targets_x), (inputs_u, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):
        inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
        inputs_u = [inputs.to(device) for inputs in inputs_u]

        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1, 1), 1)

        optimizer.zero_grad()

        # 在这里计算 logits_x
        logits_x = model(inputs_x)
        loss = mixmatch_loss(inputs_x, targets_x, inputs_u, targets_x, logits_x)  # 将 logits_x 传递给 mixmatch_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = logits_x.max(1)
        total += targets_x.size(0)
        correct += predicted.eq(targets_x).sum().item()

        print('Epoch [%d/%d] Iter [%d/%d] Loss: %.4f | Acc: %.4f%%'
              % (epoch + 1, epochs, batch_idx + 1, len(labeled_loader), train_loss / (batch_idx + 1),
                 100. * correct / total))

    scheduler.step()


# 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test Loss: %.4f | Acc: %.4f%%' % (test_loss / (batch_idx + 1), 100. * correct / total))

# 主训练循环
for epoch in range(epochs):
    train(epoch)
    test()
