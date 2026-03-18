import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 加载阶段九结果
data = np.load('experiments/stage9_extended_results.npz', allow_pickle=True)
history = data['history'].item()
cm = data['cm']
test_acc = data['test_acc'].item()

# 类别名称（与你的数据集顺序一致）
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

epochs = range(1, len(history['train_acc']) + 1)

# 创建1行3列的子图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图1：准确率曲线
axes[0].plot(epochs, history['train_acc'], 'o-', label='Train Acc', color='blue')
axes[0].plot(epochs, history['val_acc'], 's-', label='Val Acc', color='orange')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Stage9: Training and Validation Accuracy')
axes[0].legend()
axes[0].grid(True)

# 子图2：损失曲线
axes[1].plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='blue')
axes[1].plot(epochs, history['val_loss'], 's-', label='Val Loss', color='orange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Stage9: Training and Validation Loss')
axes[1].legend()
axes[1].grid(True)

# 子图3：混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[2], cmap='Blues', xticks_rotation='vertical', colorbar=False)
axes[2].set_title(f'Stage9: Confusion Matrix (Test Acc: {test_acc:.2%})')

plt.tight_layout()
plt.savefig('experiments/stage9_combined.png', dpi=150)
print("图像已保存至 experiments/stage9_combined.png")
plt.show()