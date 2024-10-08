import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack([(label) for label in labels])
    return images, labels

def get_dataloader(batch_size, num_workers, image_size):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform2 = Compose([
        Resize((224, 224)),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = VOCSegmentation(
        root="./data",
        year="2007",
        image_set="train",
        download=False,
        transform=transform,
        target_transform=transform2  # 新增的标签转换步骤
    )
    testset = VOCSegmentation(
        root="./data",
        year="2007",
        image_set="val",
        download=False,
        transform=transform,
        target_transform=transform2  # 新增的标签转换步骤
    )
    # img, tgt =trainset[0]
    # img, tgt = testset[0]
    # print(img)
    # print(tgt)
    # transforms.ToPILImage()(tgt).show()

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

    return trainloader, testloader


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.model = fcn_resnet50(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)["out"]
        return x


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # transforms.ToPILImage()(images[0]).show()
        # transforms.ToPILImage()(labels[0]).show()
        outputs = model(images)
        # print(outputs[0])
        loss = criterion(outputs, labels.long().squeeze(1))
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
            loss = criterion(outputs, labels.long().squeeze(1))

            running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def save_segmentation_result(image, labels, output_path, num_classes):
    # 创建调色板，这里使用随机颜色作为示例
    palette = [0] * 256 * 3  # 初始化调色板列表
    for i in range(num_classes):
        # 为每个类别设置随机颜色
        palette[i * 3: (i + 1) * 3] = [random.randint(0, 255) for _ in range(3)]

    # 将调色板转换为字节类型
    palette = bytes(palette)

    # 将原始图像和标签图像都转换为"RGBA"模式
    image = image.convert("RGBA")
    labels = labels.convert("RGBA")

    # 将标签图像调整为与原始图像相同的大小
    labels = labels.resize(image.size)

    # 将标签图像的模式转换为"P"模式
    labels = labels.convert("P")

    # 将调色板应用于标签图像
    labels.putpalette(palette)

    # 使用alpha通道将原始图像与标签图像混合
    result = Image.alpha_composite(image, labels)

    # 保存结果图像
    result.save(output_path)


# 运行主函数
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(0)

    # 定义超参数
    batch_size = 4
    num_workers = 0
    learning_rate = 0.0001
    num_epochs = 1
    num_classes = 21  # VOC数据集中有20个物体类别 + 背景

    # 获取数据加载器
    trainloader, testloader = get_dataloader(batch_size, num_workers, (500, 333))

    # 创建模型
    model = FCN8s(num_classes)
    print("create model")

    # 将模型移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("use", device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和测试循环
    for epoch in range(num_epochs):
        train_loss = train(model, trainloader, criterion, optimizer, device)
        test_loss = test(model, testloader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    # 保存模型
    torch.save(model.state_dict(), "fcn8s_model.pth")
    j = 0
    # 在测试循环中保存语义分割结果
    for images, _ in testloader:
        images = images.to(device)
        outputs = model(images)
        # print(outputs.shape)
        # print(outputs[0])
        _, predicted = torch.max(outputs, 1)
        print("shape", predicted.shape)
        # np.savetxt("./result/outputs.txt", outputs.cpu().detach().numpy())
        
        for i in range(images.size(0)):
            image = images[i].cpu()
            label = predicted[i].cpu()
            print(label.unique())
            # transforms.ToPILImage()(label).save(f"./result/result_{j}.png")
            j += 1

            # transform = transforms.ToPILImage()
            # image = transform(image)
            # image.show()
            # label = transform(label)
            # label.show()


            # 保存语义分割结果
            # save_segmentation_result(image, label, f"./result/result_{i}.png", num_classes)
