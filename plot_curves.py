import numpy as np
import matplotlib.pyplot as plt

stages = ['stage1', 'stage2', 'stage3', 'stage4']
colors = ['blue', 'green', 'orange', 'red']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, stage in enumerate(stages):
    data = np.load(f'experiments/{stage}_results.npz', allow_pickle=True)
    history = data['history'].item()
    epochs = range(1, len(history['train_acc'])+1)

    ax = axes[i]
    ax.plot(epochs, history['train_acc'], 'o-', label='Train Acc', color=colors[i], alpha=0.7)
    ax.plot(epochs, history['val_acc'], 's-', label='Val Acc', color=colors[i], linestyle='--')
    ax.set_title(f'{stage.upper()} Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/accuracy_curves.png', dpi=150)
plt.show()