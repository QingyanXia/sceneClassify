import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('experiments/summary.csv')

# 设置风格
plt.style.use('seaborn-v0_8-darkgrid')

# 柱状图
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(df['Stage'], df['Test Accuracy'], color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.2%}', ha='center', va='bottom', fontsize=12)

ax.set_ylim(0.7, 0.9)
ax.set_ylabel('Test Accuracy', fontsize=14)
ax.set_xlabel('Model Stage', fontsize=14)
ax.set_title('Comparison of Test Accuracy Across Stages', fontsize=16)

plt.tight_layout()
plt.savefig('experiments/accuracy_comparison.png', dpi=150)
plt.show()