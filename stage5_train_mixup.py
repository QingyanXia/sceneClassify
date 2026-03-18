import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import os

from utils.data_loader import get_data_loaders
from models.simple_resnet import SimpleResNet
from utils.train_eval import train_model, test_model, train_one_epoch_mixup, validate

def main():
    data_dir = './data'
    batch_size = 64
    epochs = 15  # 可适当增加，MixUp 需要更多 epoch 发挥效果
    lr = 0.001
    alpha_mixup = 0.2  # MixUp 参数，常用 0.2 或 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 使用数据增强（阶段三的增强）
    print("加载数据（启用增强）...")
    train_loader, val_loader, test_loader, idx_to_class = get_data_loaders(
        data_dir, batch_size=batch_size, train_augment=True
    )
    print("类别映射:", idx_to_class)

    model = SimpleResNet(num_classes=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 自定义训练循环，支持 MixUp
    print("开始训练阶段五（+MixUp）...")
    start_time = time.time()
    best_val_acc = 0.0
    best_model_wts = model.state_dict().copy()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_mixup(model, train_loader, criterion, optimizer, device, alpha=alpha_mixup)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'checkpoints/stage5_mixup_best.pth')
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}秒")

    # 测试
    print("在测试集上评估...")
    test_acc, cm, _, _ = test_model(model, test_loader, device)
    print(f"测试准确率: {test_acc:.4f}")

    # 保存结果
    np.savez('experiments/stage5_mixup_results.npz',
             history=history,
             test_acc=test_acc,
             cm=cm,
             train_time=train_time,
             params=total_params)

    print("阶段五实验完成，结果已保存。")

if __name__ == '__main__':
    main()