from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import random

# Define the helper function
def decode_segmap(image, nc=21):

    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_dataloader(batch_size, num_workers):
    transform = Compose(
        [
            Resize((256, 256)),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    transform2 = Compose(
        [
            Resize((256, 256)),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

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
        target_transform=transform2,  # 新增的标签转换步骤
    )
    # img, tgt = trainset[0]
    # img, tgt = testset[0]
    # print(img)
    # print(tgt)
    # transforms.ToPILImage()(tgt).show()

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return trainloader, testloader


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
    trainloader, testloader = get_dataloader(batch_size, num_workers)

    # 创建模型并将模型移至GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fcn_resnet101(pretrained=True).to(device).eval()

    # 在测试循环中保存语义分割结果
    for images, _ in testloader:
        images = images.to(device)
        outputs = model(images)["out"]
        print("outputs", outputs.shape)

        for i in range(images.size(0)):
            image = images[i].cpu()
            label = torch.argmax(outputs[i], dim=0).detach().cpu().numpy()
            label_rgb = decode_segmap(label)
            print("label", label.shape)

            transforms.ToPILImage()(image).show()
            plt.imshow(label_rgb); 
            plt.show()
