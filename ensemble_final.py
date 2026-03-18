import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from models.se_resnet import SEResNet

# 原始 SEResNet（无额外 Dropout，用于 stage7, stage8）
class SEResNetOriginal(SEResNet):
    def __init__(self, num_classes=6, reduction=16):
        super().__init__(num_classes=num_classes, reduction=reduction)
        # 父类中 fc 已经是 Linear(512, num_classes)，无需修改

# 带 Dropout 的 SEResNet（用于 stage10, stage11）
class SEResNetWithDropout(SEResNet):
    def __init__(self, num_classes=6, reduction=16):
        super().__init__(num_classes=num_classes, reduction=reduction)
        # 替换 fc 为 Sequential + Dropout（与训练时一致）
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

def load_model(model_class, checkpoint_path, device='cuda'):
    model = model_class(num_classes=6, reduction=16).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    data_dir = './data'
    test_dir = os.path.join(data_dir, 'seg_test')

    # 测试集预处理
    basic_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=basic_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    class_names = test_dataset.classes
    print("类别名称:", class_names)

    # 模型路径
    model_paths = {
        'stage7': 'checkpoints/stage7_optim_best.pth',
        'stage8': 'checkpoints/stage8_weighted_best.pth',
        'stage10': 'checkpoints/stage10_advanced_best.pth',
        'stage11': 'checkpoints/stage11_80epoch_best.pth'
    }

    # 加载模型（不同阶段使用不同类）
    models = {}
    print("加载 stage7 模型...")
    models['stage7'] = load_model(SEResNetOriginal, model_paths['stage7'], device)
    print("加载 stage8 模型...")
    models['stage8'] = load_model(SEResNetOriginal, model_paths['stage8'], device)
    print("加载 stage10 模型...")
    models['stage10'] = load_model(SEResNetWithDropout, model_paths['stage10'], device)
    print("加载 stage11 模型...")
    models['stage11'] = load_model(SEResNetWithDropout, model_paths['stage11'], device)

    # 集成预测
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            batch_probs = []
            for name, model in models.items():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())
            # 平均概率
            avg_probs = np.mean(batch_probs, axis=0)
            all_probs.extend(avg_probs)
            preds = np.argmax(avg_probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 计算准确率
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n集成模型测试准确率: {acc:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(f'Ensemble (Stage7+8+10+11) Confusion Matrix (Acc: {acc:.2%})')
    plt.tight_layout()
    plt.savefig('experiments/ensemble_final_confusion.png', dpi=150)
    plt.show()

    # 保存集成结果
    np.savez('experiments/ensemble_final_results.npz',
             test_acc=acc,
             cm=cm,
             preds=all_preds,
             labels=all_labels,
             probs=all_probs)

    print("\n集成结果已保存至 experiments/ensemble_final_results.npz")

if __name__ == '__main__':
    main()