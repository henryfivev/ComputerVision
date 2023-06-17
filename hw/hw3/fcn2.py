from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
from torchvision.models.segmentation import fcn_resnet50
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


def segment(net, path):
    img = Image.open(path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    # Comment the Resize and CenterCrop for better inference results
    trf = transforms.Compose(
        [
            Resize(256),
            #    CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    inp = trf(img).unsqueeze(0)
    out = net(inp)["out"]
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack([(label) for label in labels])
    return images, labels


def get_dataloader(batch_size, num_workers, image_size):
    transform = Compose(
        [  
            Resize((224, 224)),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    transform2 = Compose(
        [
            Resize((224, 224)),
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
        collate_fn=custom_collate_fn,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )

    return trainloader, testloader


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.model = fcn_resnet50(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)["out"]
        return x

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
# VOC_COLORMAP相当于一个调色板

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label): # 将colormap 是channel * height * width
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]
    return colormap2label[idx]

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        labels = decode_segmap(labels)
        labels = torch.from_numpy(labels)[0][0]
        print(labels.shape)
        labels = voc_label_indices(labels, voc_colormap2label())
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # transforms.ToPILImage()(images[0]).show()
        # transforms.ToPILImage()(labels[0]).show()
        print(images.shape)
        print(labels.shape)
        outputs = model(images)
        # print(outputs[0])
        loss = criterion(outputs, labels.long().unsqueeze(0))
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
        palette[i * 3 : (i + 1) * 3] = [random.randint(0, 255) for _ in range(3)]

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
    batch_size = 1
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
    # 在测试循环中保存语义分割结果
    for images, _ in testloader:
        images = images.to(device)
        outputs = model(images)
        print("outputs", outputs.shape)

        for i in range(images.size(0)):
            image = images[i].cpu()
            label = torch.argmax(outputs[i], dim=0).detach().cpu().numpy()
            label_rgb = decode_segmap(label)
            print("label", label.shape)

            transforms.ToPILImage()(image).show()
            plt.imshow(label_rgb); 
            plt.show()