# stage8_train_weighted.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import numpy as np

from utils.data_loader import get_data_loaders
from models.se_resnet import SEResNet
from utils.train_eval import train_one_epoch, validate, test_model

def compute_class_weights(train_loader, num_classes, device):
    """计算类别权重（基于样本数倒数）"""
    class_counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for l in labels:
            class_counts[l] += 1
    # 使用 inverse frequency，并归一化使平均权重为1
    weights = 1.0 / class_counts
    weights = weights / weights.mean()  # 使平均权重为1，避免损失尺度过大
    return weights.to(device)

def main():
    data_dir = './data'
    batch_size = 64
    epochs = 20
    lr = 0.001
    weight_decay = 1e-4
    label_smoothing = 0.1  # 可选，若用加权交叉熵则不用平滑
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据（启用增强）
    print("加载数据（启用增强）...")
    train_loader, val_loader, test_loader, idx_to_class = get_data_loaders(
        data_dir, batch_size=batch_size, train_augment=True
    )
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print("类别名称:", class_names)

    # 计算类别权重
    class_weights = compute_class_weights(train_loader, len(class_names), device)
    print("类别权重:", class_weights.cpu().numpy())

    model = SEResNet(num_classes=6, reduction=16).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")

    # 使用加权交叉熵损失（可替换为加权FocalLoss）
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("开始训练阶段八（类别加权）...")
    start_time = time.time()
    best_val_acc = 0.0
    best_model_wts = model.state_dict().copy()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'checkpoints/stage8_weighted_best.pth')
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}秒")

    # 测试
    print("在测试集上评估...")
    test_acc, cm, _, _ = test_model(model, test_loader, device)
    print(f"测试准确率: {test_acc:.4f}")

    # 保存结果
    np.savez('experiments/stage8_weighted_results.npz',
             history=history,
             test_acc=test_acc,
             cm=cm,
             train_time=train_time,
             params=total_params)

    print("阶段八实验完成，结果已保存。")

if __name__ == '__main__':
    main()