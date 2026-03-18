import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 导入模型定义
from models.se_resnet import SEResNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    data_dir = './data'
    test_dir = data_dir + '/seg_test'

    # 测试集预处理（与训练时验证/测试一致）
    basic_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=basic_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    class_names = test_dataset.classes
    print("类别名称:", class_names)

    # 加载两个模型
    model7 = SEResNet(num_classes=6, reduction=16).to(device)
    model7.load_state_dict(torch.load('checkpoints/stage7_optim_best.pth', map_location=device))
    model7.eval()

    model8 = SEResNet(num_classes=6, reduction=16).to(device)
    model8.load_state_dict(torch.load('checkpoints/stage8_weighted_best.pth', map_location=device))
    model8.eval()
    print("模型加载完成。")

    # 集成预测
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs7 = model7(inputs)
            outputs8 = model8(inputs)

            # 计算softmax概率并平均
            probs7 = torch.softmax(outputs7, dim=1)
            probs8 = torch.softmax(outputs8, dim=1)
            avg_probs = (probs7 + probs8) / 2
            _, predicted = torch.max(avg_probs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算准确率和混淆矩阵
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n集成模型测试准确率: {acc:.4f}")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(f'Ensemble (Stage7+Stage8) Confusion Matrix (Acc: {acc:.2%})')
    plt.tight_layout()
    plt.savefig('experiments/ensemble_stage7_8_confusion.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    main()