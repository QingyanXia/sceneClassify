import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.data_loader import get_data_loaders
from models.simple_resnet import SimpleResNet
from models.plain_resnet import PlainNet
from utils.train_eval import train_model, test_model

def train_ablation(model_class, name, use_augment=True, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== 消融实验: {name} ===")
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        './data/intel_image', batch_size=64, train_augment=use_augment
    )
    model = model_class(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    _, _ = train_model(model, train_loader, val_loader, epochs,
                       criterion, optimizer, device, save_path=None)
    test_acc, _, _, _ = test_model(model, test_loader, device)
    print(f"{name} 测试准确率: {test_acc:.4f}")
    return test_acc

def main():
    # 基准：阶段四（完整模型+数据增强）
    baseline_acc = train_ablation(SimpleResNet, "Baseline (Stage4)", use_augment=True, epochs=5)

    # 消融1: 去掉残差 (PlainNet + 增强)
    no_residual_acc = train_ablation(PlainNet, "No Residual", use_augment=True, epochs=5)

    # 消融2: 去掉BN (在SimpleResNet中删除所有BN层，需自定义新模型，或手动修改)
    # 简单起见，我们这里先不做去掉BN，因为需要修改模型定义。或者可以写一个NoBN_ResNet。
    # 你可以手动创建 NoBN_ResNet，但时间有限，可暂时跳过，在PPT中说明计划。

    # 消融3: 去掉数据增强 (SimpleResNet + no augment)
    no_augment_acc = train_ablation(SimpleResNet, "No Augment", use_augment=False, epochs=5)

    # 汇总
    results = {
        'Baseline (Stage4)': baseline_acc,
        'No Residual': no_residual_acc,
        'No Augment': no_augment_acc
    }
    print("\n=== 消融实验结果 ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()