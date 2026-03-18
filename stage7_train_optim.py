import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import numpy as np

from utils.data_loader import get_data_loaders
from models.se_resnet import SEResNet
from utils.train_eval import train_model, test_model
from utils.losses import LabelSmoothingCrossEntropy  # 需创建此文件

def main():
    data_dir = './data'
    batch_size = 64
    epochs = 20  # 增加轮次以配合余弦退火
    lr = 0.001
    weight_decay = 1e-4  # 添加权重衰减
    label_smoothing = 0.1  # 标签平滑系数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据（启用增强，与阶段六相同）
    print("加载数据（启用增强）...")
    train_loader, val_loader, test_loader, idx_to_class = get_data_loaders(
        data_dir, batch_size=batch_size, train_augment=True
    )
    print("类别映射:", idx_to_class)

    # 创建带SE模块的ResNet（与阶段六相同）
    model = SEResNet(num_classes=6, reduction=16).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")

    # 使用标签平滑损失替代交叉熵
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    # 使用 AdamW 优化器（内置权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 余弦退火学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("开始训练阶段七（优化策略）...")
    start_time = time.time()
    best_val_acc = 0.0
    best_model_wts = model.state_dict().copy()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # 训练一个epoch（使用原有的train_one_epoch，但传入修改后的criterion）
        # 这里直接复用 train_model 中的 train_one_epoch 逻辑，但需要单独实现或调用函数
        # 为简化，我们可以使用之前定义的 train_one_epoch 函数（需确保它使用 criterion）
        from utils.train_eval import train_one_epoch, validate

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()  # 更新学习率

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
    torch.save(model.state_dict(), 'checkpoints/stage7_optim_best.pth')
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}秒")

    # 测试
    print("在测试集上评估...")
    test_acc, cm, _, _ = test_model(model, test_loader, device)
    print(f"测试准确率: {test_acc:.4f}")

    # 保存结果
    np.savez('experiments/stage7_optim_results.npz',
             history=history,
             test_acc=test_acc,
             cm=cm,
             train_time=train_time,
             params=total_params)

    print("阶段七实验完成，结果已保存。")

if __name__ == '__main__':
    main()