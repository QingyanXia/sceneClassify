import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

stages = ['stage1', 'stage2', 'stage3', 'stage4']
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, stage in enumerate(stages):
    data = np.load(f'experiments/{stage}_results.npz', allow_pickle=True)
    cm = data['cm']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[i], colorbar=False)
    axes[i].set_title(f'{stage.upper()} Confusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('True')

plt.tight_layout()
plt.savefig('experiments/confusion_matrices.png', dpi=150)
plt.show()