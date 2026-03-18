import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

data = np.load('experiments/stage10_advanced_results.npz', allow_pickle=True)
history = data['history'].item()
cm = data['cm']
test_acc = data['test_acc'].item()
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

epochs = range(1, len(history['train_acc']) + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(epochs, history['train_acc'], 'o-', label='Train Acc')
axes[0].plot(epochs, history['val_acc'], 's-', label='Val Acc')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Stage10: Training and Validation Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, history['train_loss'], 'o-', label='Train Loss')
axes[1].plot(epochs, history['val_loss'], 's-', label='Val Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Stage10: Training and Validation Loss')
axes[1].legend()
axes[1].grid(True)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=axes[2], cmap='Blues', xticks_rotation='vertical', colorbar=False)
axes[2].set_title(f'Stage10: Confusion Matrix (Test Acc: {test_acc:.2%})')

plt.tight_layout()
plt.savefig('experiments/stage10_combined.png', dpi=150)
plt.show()