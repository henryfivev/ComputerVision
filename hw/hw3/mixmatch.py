import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据增强函数
def data_augmentation(x):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
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
    
    # 优化器更新
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# 参数设置
num_classes = 10  # 类别数
T = 0.5  # 温度参数
K = 2  # 数据增强次数
alpha = 0.75  # 损失函数权重参数
threshold = 0.95  # 选择 MixMatch 样本的阈值
batch_size = 64
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_labeled_idxs = torch.randperm(len(train_dataset))[:1000]  # 选取部分标记数据
train_unlabeled_idxs = torch.randperm(len(train_dataset))[1000:]  # 剩余未标记数据
train_labeled_dataset = torch.utils.data.Subset(train_dataset, train_labeled_idxs)
train_unlabeled_dataset = torch.utils.data.Subset(train_dataset, train_unlabeled_idxs)
train_labeled_loader = torch.utils.data.DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
train_unlabeled_loader = torch.utils.data.DataLoader(train_unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# 初始化模型、优化器等
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    labeled_data_iter = iter(train_labeled_loader)
    unlabeled_data_iter = iter(train_unlabeled_loader)
    
    for batch_idx, (labeled_x, labeled_y) in enumerate(labeled_data_iter):
        try:
            unlabeled_x = next(unlabeled_data_iter)[0]
        except StopIteration:
            unlabeled_data_iter = iter(train_unlabeled_loader)
            unlabeled_x = next(unlabeled_data_iter)[0]
        
        labeled_x, labeled_y, unlabeled_x = labeled_x.to(device), labeled_y.to(device), unlabeled_x.to(device)
        
        mixmatch(model, (labeled_x, labeled_y), unlabeled_x, num_classes, T, K, alpha)
    
    # 在验证集上进行评估等操作

# 最后进行模型评估或测试
