import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import numpy as np

from utils.data_loader import get_data_loaders
from models.se_resnet import SEResNet
from utils.train_eval import train_one_epoch, validate, test_model
from utils.losses import LabelSmoothingCrossEntropy

def main():
    data_dir = './data'
    batch_size = 64
    epochs = 40               # 增加到40
    lr = 0.001
    weight_decay = 1e-4
    label_smoothing = 0.1
    patience = 7               # 早停容忍次数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据（启用增强，与阶段七相同）
    print("加载数据（启用增强）...")
    train_loader, val_loader, test_loader, idx_to_class = get_data_loaders(
        data_dir, batch_size=batch_size, train_augment=True
    )
    print("类别映射:", idx_to_class)

    # 创建带SE模块的ResNet
    model = SEResNet(num_classes=6, reduction=16).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")

    # 标签平滑损失
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    # AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 余弦退火学习率调度
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("开始训练阶段九（延长训练 + 早停）...")
    start_time = time.time()
    best_val_acc = 0.0
    best_model_wts = model.state_dict().copy()
    epochs_no_improve = 0
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

        # 早停检查
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
    torch.save(model.state_dict(), 'checkpoints/stage9_extended_best.pth')
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}秒")

    # 测试
    print("在测试集上评估...")
    test_acc, cm, _, _ = test_model(model, test_loader, device)
    print(f"测试准确率: {test_acc:.4f}")

    # 保存结果
    np.savez('experiments/stage9_extended_results.npz',
             history=history,
             test_acc=test_acc,
             cm=cm,
             train_time=train_time,
             params=total_params)

    print("阶段九实验完成，结果已保存。")

if __name__ == '__main__':
    main()