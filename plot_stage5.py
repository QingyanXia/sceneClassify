import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 加载结果
data = np.load('experiments/stage5_mixup_results.npz', allow_pickle=True)
history = data['history'].item()
test_acc = data['test_acc'].item()
cm = data['cm']
params = data['params'].item()

# 类别名称（与之前一致）
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

print(f"阶段五测试准确率: {test_acc:.4f}")
print(f"模型参数量: {params}")

# 计算各类别准确率（召回率）
recall = cm.diagonal() / cm.sum(axis=1)
for i, name in enumerate(class_names):
    print(f"{name:12s} 召回率: {recall[i]:.4f}")

# 绘制训练曲线
epochs = range(1, len(history['train_acc']) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_acc'], 'o-', label='Train Acc')
plt.plot(epochs, history['val_acc'], 's-', label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Stage5 (+MixUp) Training Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
plt.plot(epochs, history['val_loss'], 's-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Stage5 (+MixUp) Loss Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('experiments/stage5_curves.png', dpi=150)
plt.show()

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Stage5 (+MixUp) Confusion Matrix')
plt.tight_layout()
plt.savefig('experiments/stage5_confusion.png', dpi=150)
plt.show()