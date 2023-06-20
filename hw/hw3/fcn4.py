import time
import copy
import torch
from torch import optim, nn
import torchvision
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.append("..")
from tqdm import tqdm
import warnings
from torchvision.models.segmentation import fcn_resnet50
warnings.filterwarnings("ignore") # 忽略警告

def read_voc_images(root="./data/VOCdevkit/VOC2007", is_train=True, max_num=None):
    if is_train == False:
        root = "./data/VOCdevkit/VOC2007_test"
    txt_fname = '%s/ImageSets/Segmentation/%s' % (root, 'train.txt' if is_train else 'test.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split() # 拆分成一个个名字组成list
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        # 读入数据并且转为RGB的 PIL image
        features[i] = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert("RGB")
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert("RGB")
    return features, labels # PIL image 0-255

def show_images(imgs, num_rows, num_cols, scale=2):
    # a_img = np.asarray(imgs)
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes

# 根据自己存放数据集的路径修改voc_dir
voc_dir = r"./data/VOCdevkit/VOC2007"
train_features, train_labels = read_voc_images(voc_dir, max_num=10)

# 标签中每个RGB颜色的值
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
# 标签其标注的类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormap2label = torch.zeros(256**3, dtype=torch.uint8) # torch.Size([16777216])
for i, colormap in enumerate(VOC_COLORMAP):
    # 每个通道的进制是256，这样可以保证每个 rgb 对应一个下标 i
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

# 构造标签矩阵
def voc_label_indices(colormap, colormap2label):
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]) 
    return colormap2label[idx] # colormap 映射 到colormaplabel中计算的下标

y = voc_label_indices(train_labels[0], colormap2label)
print(y[100:110, 130:140]) #打印结果是一个int型tensor，tensor中的每个元素i表示该像素的类别是VOC_CLASSES[i]

def voc_rand_crop(feature, label, height, width):
    """
    随机裁剪feature(PIL image) 和 label(PIL image).
    为了使裁剪的区域相同，不能直接使用RandomCrop，而要像下面这样做
    Get parameters for ``crop`` for a random crop.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    i,j,h,w = torchvision.transforms.RandomCrop.get_params(feature, output_size=(height, width))
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)
    return feature, label



class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        """
        crop_size: (h, w)
        """
        # 对输入图像的RGB三个通道的值分别做标准化
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)])
        self.crop_size = crop_size # (h, w)
        features, labels = read_voc_images(root=voc_dir, is_train=is_train,  max_num=max_num)
# 由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本需要通过自定义的filter函数所移除
        self.features = self.filter(features) # PIL image
        self.labels = self.filter(labels)     # PIL image
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' valid examples')

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
                                # float32 tensor           uint8 tensor (b,h,w)
        return (self.tsf(feature), voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

batch_size = 4
crop_size = (320, 480)
max_num = 20000

# 创建训练集和测试集的实例
voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)

# 设批量大小为32，分别定义【训练集】和【测试集】的数据迭代器
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(voc_test, batch_size, drop_last=True,
                             num_workers=num_workers)

# 方便封装，把训练集和验证集保存在dict里
dataloaders = {'train':train_iter, 'val':test_iter}
dataset_sizes = {'train':len(voc_train), 'val':len(voc_test)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 21 # 21分类，1个背景，20个物体
model_ft = fcn_resnet50(pretrained=True) # 设置True，表明要加载使用训练好的参数
model_ft = model_ft.to(device)

def train_model(model:nn.Module, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # 每个epoch都有一个训练和验证阶段
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            runing_loss = 0.0
            runing_corrects = 0.0
            # 迭代一个epoch
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad() # 零参数梯度
				# 前向，只在训练时跟踪参数
                with torch.set_grad_enabled(phase=='train'):
                    logits = model(inputs)  # [5, 21, 320, 480]
                    logits = list(logits.values())[0] 
                    loss = criterion(logits, labels.long())
                    # 后向，只在训练阶段进行优化
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
				# 统计loss和correct
                runing_loss += loss.item()*inputs.size(0)
                runing_corrects += torch.sum((torch.argmax(logits.data,1))==labels.data)/(480*320)

            epoch_loss = runing_loss / dataset_sizes[phase]
            epoch_acc = runing_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			# 深度复制model参数
            if phase=='val' and epoch_acc>best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

epochs = 5 # 训练5个epoch
criteon = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)
# 每3个epochs衰减LR通过设置gamma=0.1
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 开始训练
model_ft = train_model(model_ft, criteon, optimizer, exp_lr_scheduler, num_epochs=epochs)

def label2image(pred):
    # pred: [320,480]
    colormap = torch.tensor(VOC_COLORMAP,device=device,dtype=int)
    x = pred.long()
    return (colormap[x,:]).data.cpu().numpy()

# 预测前将图像标准化，并转换成(b,c,h,w)的tensor
def predict(img, model):
    tsf = transforms.Compose([
            transforms.ToTensor(), # 好像会自动转换channel
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    x = tsf(img).unsqueeze(0).to(device) # (3,320,480) -> (1,3,320,480)
    pred = torch.argmax(list(model(x).values())[0], dim=1) # 每个通道选择概率最大的那个像素点 -> (1,320,480)
    return pred.reshape(pred.shape[1],pred.shape[2]) # reshape成(320,480)

def evaluate(model:nn.Module):
    model.eval()
    test_images, test_labels = read_voc_images(voc_dir, is_train=False) 
    n, imgs = 4, []
    for i in range(10):
        xi, yi = voc_rand_crop(test_images[i], test_labels[i], 224, 224) # Image
        pred = label2image(predict(xi, model))
        imgs += [xi, pred, yi]
        plt.imshow(pred)
        plt.savefig(f'result/trained/{i}.png')
        # plt.show()
    print("finished")

# 开始测试
evaluate(model_ft)
