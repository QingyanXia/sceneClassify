import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

from utils.data_loader import get_data_loaders
from models.cnn_bn_dropout import CNNWithBNDropout   # 模型不变
from utils.train_eval import train_model, test_model

def main():
    data_dir = './data'
    batch_size = 64
    epochs = 10
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 关键改动：启用数据增强
    print("加载数据（启用增强）...")
    train_loader, val_loader, test_loader, idx_to_class = get_data_loaders(
        data_dir, batch_size=batch_size, train_augment=True
    )
    print("类别映射:", idx_to_class)

    model = CNNWithBNDropout(num_classes=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("开始训练阶段三...")
    start_time = time.time()
    model, history = train_model(
        model, train_loader, val_loader, epochs,
        criterion, optimizer, device,
        save_path='checkpoints/stage3_best.pth'
    )
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}秒")

    test_acc, cm, _, _ = test_model(model, test_loader, device)
    print(f"测试准确率: {test_acc:.4f}")

    np.savez('experiments/stage3_results.npz',
             history=history,
             test_acc=test_acc,
             cm=cm,
             train_time=train_time,
             params=total_params)

    print("阶段三实验完成，结果已保存。")

if __name__ == '__main__':
    main()