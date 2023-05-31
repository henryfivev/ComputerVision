import torch
from torch import nn
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchssl.utils.data import SSLDataLoader
from torchssl.utils.transforms import cifar10_transform
from torchssl.utils.ssl_trainer import MixMatchTrainer, FixMatchTrainer

num_labeled = 1000  # 标记样本数量
num_unlabeled = 4000  # 未标记样本数量
batch_size = 64
num_epochs = 100
learning_rate = 0.002
temperature = 0.5  # MixMatch的温度参数
lambda_u = 1.0  # FixMatch的权重参数
threshold = 0.95  # FixMatch的伪标签阈值
trainset = CIFAR10(
    root="./data", train=True, download=True, transform=cifar10_transform()
)
testset = CIFAR10(
    root="./data", train=False, download=True, transform=cifar10_transform()
)

# 按照标记和未标记样本数量划分数据集
labeled_indices = torch.arange(num_labeled)
unlabeled_indices = torch.arange(num_labeled, num_labeled + num_unlabeled)

labeled_dataset = torch.utils.data.Subset(trainset, labeled_indices)
unlabeled_dataset = torch.utils.data.Subset(trainset, unlabeled_indices)

# 创建数据加载器
labeled_loader = torch.utils.data.DataLoader(
    labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
unlabeled_loader = torch.utils.data.DataLoader(
    unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=4
)
model_mixmatch = MixMatchTrainer(
    backbone="WideResNet28_2",
    num_classes=10,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    temperature=temperature,
)

model_mixmatch.fit(labeled_loader, unlabeled_loader)
model_fixmatch = FixMatchTrainer(
    backbone="WideResNet28_2",
    num_classes=10,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    lambda_u=lambda_u,
    threshold=threshold,
)

model_fixmatch.fit(labeled_loader, unlabeled_loader)
test_accuracy_mixmatch = model_mixmatch.evaluate(test_loader)
print(f"MixMatch Test Accuracy: {test_accuracy_mixmatch * 100:.2f}%")

test_accuracy_fixmatch = model_fixmatch.evaluate(test_loader)
print(f"FixMatch Test Accuracy: {test_accuracy_fixmatch * 100:.2f}%")
