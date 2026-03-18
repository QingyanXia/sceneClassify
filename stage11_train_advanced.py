import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from torch.utils.data import DataLoader
import time
import numpy as np
import os

from models.se_resnet import SEResNet
from utils.train_eval import validate, test_model
from utils.losses import LabelSmoothingCrossEntropy

# 自定义 MixUp 训练函数（与阶段十相同）
def train_one_epoch_mixup(model, train_loader, criterion, optimizer, device, alpha=0.2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(inputs.size(0)).to(device)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        labels_a, labels_b = labels, labels[index]

        outputs = model(mixed_inputs)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def get_advanced_data_loaders(data_dir, batch_size=64):
    """返回带有 RandAugment 和 MixUp 的数据加载器（与阶段十相同）"""
    train_dir = os.path.join(data_dir, 'seg_train')
    test_dir = os.path.join(data_dir, 'seg_test')

    basic_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=basic_transform)

    # 划分训练/验证集
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )

    # 获取划分后的索引
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # 重新创建数据集，确保验证集使用 basic_transform
    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(train_dir, transform=train_transform), train_indices
    )
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(train_dir, transform=basic_transform), val_indices
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, full_train_dataset.classes

def main():
    data_dir = './data'
    batch_size = 64
    epochs = 80                     # 增加到80 epoch
    lr = 0.001
    weight_decay = 5e-4
    label_smoothing = 0.1
    mixup_alpha = 0.2
    patience = 10                    # 早停容忍度增加到10
    warmup_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("加载数据（RandAugment + 弱化基础增强）...")
    train_loader, val_loader, test_loader, class_names = get_advanced_data_loaders(data_dir, batch_size)
    print("类别名称:", class_names)

    # 模型：添加 Dropout(0.3) 在分类层前
    model = SEResNet(num_classes=6, reduction=16).to(device)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 6)
    )
    model = model.to(device)        # 确保所有参数都在同一设备

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")

    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度：warmup + cosine
    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    print("开始训练阶段十一（80 epoch + 早停10）...")
    start_time = time.time()
    best_val_acc = 0.0
    best_model_wts = model.state_dict().copy()
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_mixup(model, train_loader, criterion, optimizer, device, alpha=mixup_alpha)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"  -> 新的最佳验证准确率: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  -> 连续 {epochs_no_improve} 个epoch未提升")

        if epochs_no_improve >= patience:
            print(f"早停触发！在第 {epoch+1} 个epoch停止训练")
            break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'checkpoints/stage11_80epoch_best.pth')
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}秒")

    # 测试
    print("在测试集上评估...")
    test_acc, cm, _, _ = test_model(model, test_loader, device)
    print(f"测试准确率: {test_acc:.4f}")

    # 保存结果
    np.savez('experiments/stage11_80epoch_results.npz',
             history=history,
             test_acc=test_acc,
             cm=cm,
             train_time=train_time,
             params=total_params)

    print("阶段十一实验完成，结果已保存。")

if __name__ == '__main__':
    main()