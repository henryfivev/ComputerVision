import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np


class WideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=2):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.in_planes = channels[0]
        self.conv1 = nn.Conv2d(
            3, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._wide_layer(WideBasicBlock, channels[1], depth, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, channels[2], depth, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, channels[3], depth, stride=2)
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.9)
        self.linear = nn.Linear(channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(WideBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def get_dataloader(dataset, batch_size, num_workers):
    data_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def get_cifar10():
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    return trainset, testset


def mixmatch_loss(
    pred_labeled, targets_labeled, pred_unlabeled, targets_unlabeled, alpha=0.75, T=0.5
):
    labeled_loss = F.cross_entropy(pred_labeled, targets_labeled)
    unlabeled_loss = (
        -torch.mean(
            torch.sum(F.log_softmax(pred_unlabeled, dim=1) * targets_unlabeled, dim=1)
        )
    ) / pred_unlabeled.shape[0]
    total_loss = labeled_loss + alpha * unlabeled_loss
    return total_loss


def fixmatch_loss(
    pred_labeled,
    targets_labeled,
    pred_unlabeled,
    targets_unlabeled,
    lambda_u=1.0,
    threshold=0.95,
    T=0.5,
):
    labeled_loss = F.cross_entropy(pred_labeled, targets_labeled)
    mask = torch.max(torch.softmax(pred_unlabeled, dim=1), dim=1)[0] >= threshold
    pseudo_labels = torch.softmax(pred_unlabeled / T, dim=1)
    pseudo_labels = pseudo_labels[mask]
    targets_unlabeled = targets_unlabeled[mask]
    unlabeled_loss = F.kl_div(
        torch.log_softmax(pred_unlabeled[mask] / T, dim=1),
        pseudo_labels,
        reduction="batchmean",
    )
    total_loss = labeled_loss + lambda_u * unlabeled_loss
    return total_loss


def train(
    model, dataloader_labeled, dataloader_unlabeled, optimizer, criterion, device
):
    model.train()
    for (inputs_labeled, targets_labeled), (inputs_unlabeled, targets_unlabeled) in zip(
        dataloader_labeled, dataloader_unlabeled
    ):
        inputs_labeled, targets_labeled = inputs_labeled.to(device), targets_labeled.to(
            device
        )
        inputs_unlabeled, targets_unlabeled = inputs_unlabeled.to(
            device
        ), targets_unlabeled.to(device)

        optimizer.zero_grad()

        outputs_labeled = model(inputs_labeled)
        outputs_unlabeled = model(inputs_unlabeled)

        loss = criterion(
            outputs_labeled, targets_labeled, outputs_unlabeled, targets_unlabeled
        )
        loss.backward()
        optimizer.step()


def test(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            total_correct += predicted.eq(targets).sum().item()

    accuracy = total_correct / total_samples
    return accuracy


# 运行主函数
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)

    # 定义超参数
    batch_size = 64
    num_workers = 4
    learning_rate = 0.001
    num_epochs = 100

    # 获取数据集
    trainset, testset = get_cifar10()

    # 创建数据加载器
    labeled_dataset = data.Subset(trainset, np.arange(1000))  # 使用1000个标记样本
    unlabeled_dataset = data.Subset(trainset, np.arange(1000, len(trainset)))  # 使用未标记样本
    dataloader_labeled = get_dataloader(labeled_dataset, batch_size, num_workers)
    dataloader_unlabeled = get_dataloader(unlabeled_dataset, batch_size, num_workers)
    dataloader_test = get_dataloader(testset, batch_size, num_workers)

    # 创建模型
    model = WideResNet(num_classes=10, depth=28, widen_factor=2)

    # 将模型移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = mixmatch_loss  # 可以根据需要选择mixmatch_loss或fixmatch_loss

    # 训练和测试循环
    for epoch in range(num_epochs):
        train(
            model,
            dataloader_labeled,
            dataloader_unlabeled,
            optimizer,
            criterion,
            device,
        )
        test_accuracy = test(model, dataloader_test, device)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {test_accuracy * 100:.2f}%"
        )

    # 最终测试
    final_test_accuracy = test(model, dataloader_test, device)
    print(f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%")
