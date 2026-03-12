import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

# 1. 定义 FlexibleImageDataset 类
"""
自定义数据集类，这样可以在构建时传入不同的transform，从而灵活控制训练集和验证集的预处理。
__getitem__ 返回图像和标签，PyTorch 的 DataLoader 会自动批量化
"""
class FlexibleImageDataset(Dataset):
    """根据文件列表和标签动态加载图像，支持自定义transform"""
    def __init__(self, file_list, labels, transform=None):
        """
        file_list: 图像文件路径列表
        labels: 对应的标签列表
        transform: 要应用的变换
        """
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        # 用PIL打开图像，并确保RGB格式
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


"""
2-3
定义了两个变换：basic_transform 用于验证和测试（只做必要的尺寸调整和标准化），train_transform 在需要时加入数据增强。
增强策略:因为自然场景照片可能存在左右镜像（例如拍山时可能横过来）、轻微旋转（手持抖动）、光照变化（颜色抖动）
这些变换不会改变场景类别，却能增加数据多样性
4-6
用 ImageFolder 临时加载训练集，目的是获得所有样本的路径和标签，以及类别映射。这里用了 basic_transform，但实际加载图片时我们会用正确的 transform。
随机打乱并划分 80% 训练，20% 验证。固定随机种子保证每次运行划分一致，便于复现

shuffle=True 只在训练集使用，打乱顺序有助于收敛。
num_workers 可以并行加载数据，加快训练。
"""
def get_data_loaders(data_dir, batch_size=64, train_augment=False, seed=42):
    """
    负责从文件夹中读取所有样本，划分训练/验证集
    # 返回 train_loader, val_loader, test_loader, idx_to_class
    data_dir: 数据集根目录，包含 seg_train 和 seg_test
    batch_size: 批次大小
    train_augment: 是否对训练集使用数据增强
    seed: 随机种子，保证可重复性
    """
    train_dir = os.path.join(data_dir, 'seg_train')
    test_dir = os.path.join(data_dir, 'seg_test')

    # 2. 定义基础变换（所有阶段验证和测试用，阶段一、二训练也用）
    basic_transform = transforms.Compose([
        transforms.Resize((150, 150)),          # 统一尺寸，Intel图片原大小150x150
        transforms.ToTensor(),                  # 转为Tensor，并缩放到[0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet均值
                             std=[0.229, 0.224, 0.225])    # ImageNet标准差
    ])

    # 3. 定义增强变换（阶段三、四训练用）
    if train_augment:
        train_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomHorizontalFlip(p=0.5),        # 随机水平翻转
            transforms.RandomRotation(10),                  # 随机旋转±10度
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1), # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = basic_transform
    # 4. 获取所有训练样本的路径和标签（使用 basic_transform 暂时加载，只是为了获取文件列表）
    temp_dataset = datasets.ImageFolder(train_dir, transform=basic_transform)
    class_to_idx = temp_dataset.class_to_idx  # 类别到索引的映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # 索引到类别名称
    samples = temp_dataset.samples  # list of (path, target)

    # 5. 随机打乱并划分训练/验证集
    random.seed(seed)
    random.shuffle(samples)
    train_ratio = 0.8
    split = int(len(samples) * train_ratio)
    train_samples = samples[:split]
    val_samples = samples[split:]

    # 6. 分离文件路径和标签
    train_files = [s[0] for s in train_samples]
    train_labels = [s[1] for s in train_samples]
    val_files = [s[0] for s in val_samples]
    val_labels = [s[1] for s in val_samples]

    # 7. 构建训练集和验证集数据集对象
    train_dataset = FlexibleImageDataset(train_files, train_labels, train_transform)
    val_dataset = FlexibleImageDataset(val_files, val_labels, basic_transform)

    # 8. 构建测试集（直接使用 ImageFolder，因为不需要划分）
    test_dataset = datasets.ImageFolder(test_dir, transform=basic_transform)

    # 9. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, idx_to_class