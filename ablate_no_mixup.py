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
from utils.train_eval import validate, test_model, train_one_epoch
from utils.losses import LabelSmoothingCrossEntropy

def main():
    data_dir = './data'
    batch_size = 64
    epochs = 40
    lr = 0.001
    weight_decay = 5e-4
    label_smoothing = 0.1
    warmup_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据增强（保留 RandAugment，去掉 MixUp）
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

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'seg_train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'seg_test'), transform=basic_transform)

    # 划分验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    # 修正验证集transform
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(os.path.join(data_dir, 'seg_train'), transform=basic_transform),
        val_dataset.indices
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    model = SEResNet(num_classes=6, reduction=16).to(device)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 6)
    )
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_model_wts = model.state_dict().copy()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 使用普通训练（无 MixUp）
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'checkpoints/ablate_no_mixup_best.pth')

    test_acc, cm, _, _ = test_model(model, test_loader, device)
    print(f"测试准确率: {test_acc:.4f}")
    np.savez('experiments/ablate_no_mixup_results.npz', history=history, test_acc=test_acc, cm=cm)

if __name__ == '__main__':
    main()