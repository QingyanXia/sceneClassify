import numpy as np
import pandas as pd

stages = ['stage1', 'stage2', 'stage3', 'stage4']
results = []

for stage in stages:
    try:
        data = np.load(f'experiments/{stage}_results.npz', allow_pickle=True)
        test_acc = data['test_acc'].item()
        train_time = data['train_time'].item()
        params = data['params'].item()
        results.append({
            'Stage': stage,
            'Test Accuracy': test_acc,
            'Training Time (s)': train_time,
            'Parameters': params
        })
    except FileNotFoundError:
        print(f"Warning: {stage}_results.npz not found. Skipping.")

df = pd.DataFrame(results)
print(df.to_string(index=False))

# 保存为CSV
df.to_csv('experiments/summary.csv', index=False)