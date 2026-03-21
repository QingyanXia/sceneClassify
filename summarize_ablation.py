import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os

# 定义要汇总的消融实验及其结果文件路径
# 可根据实际运行的文件名修改
models = {
    "Baseline (Stage10)": "experiments/stage10_advanced_results.npz",           # 阶段十基准
    "-MixUp": "experiments/ablate_no_mixup_results.npz",
    "-RandAugment": "experiments/ablate_no_randaugment_results.npz",
    "-SE Attention": "experiments/ablate_no_se_results.npz",
    "-Dropout": "experiments/ablate_no_dropout_results.npz",
    "-Label Smoothing": "experiments/ablate_no_label_smoothing_results.npz",
    "-Cosine Annealing": "experiments/ablate_no_cosine_results.npz",
    "-MixUp -LS -WD": "experiments/stage10_no_mixup_no_ls_no_wd_results.npz",  # 组合消融
}

# 类别名称（与数据集一致）
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# 存储结果
results = []
cms = {}

# 加载每个实验
for name, path in models.items():
    if not os.path.exists(path):
        print(f"警告: {path} 不存在，跳过 {name}")
        continue
    data = np.load(path, allow_pickle=True)
    test_acc = data['test_acc'].item()
    cm = data['cm']
    # 计算分类报告（加权平均）
    report = classification_report(
        y_true=np.arange(cm.shape[0]).repeat(cm.sum(axis=1)),
        y_pred=np.repeat(np.arange(cm.shape[0]), cm.sum(axis=0)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    # 提取加权平均指标
    weighted_avg = report['weighted avg']
    results.append({
        'Model': name,
        'Accuracy': test_acc,
        'Precision': weighted_avg['precision'],
        'Recall': weighted_avg['recall'],
        'F1-score': weighted_avg['f1-score']
    })
    cms[name] = cm

# 转换为 DataFrame 并打印表格
import pandas as pd
df = pd.DataFrame(results)
print("\n========== 消融实验汇总表 ==========")
print(df.to_string(index=False, float_format='%.4f'))

# 保存表格到 CSV
df.to_csv('experiments/ablation_summary.csv', index=False)
print("\n表格已保存至 experiments/ablation_summary.csv")

# 绘制混淆矩阵图（所有模型并排）
n_models = len(cms)
if n_models == 0:
    print("没有可用的混淆矩阵数据，退出。")
    exit()

# 计算子图布局（尽量接近方形）
n_cols = min(4, n_models)  # 最多4列
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten() if n_models > 1 else [axes]

for i, (name, cm) in enumerate(cms.items()):
    ax = axes[i]
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_title(f"{name}\nAcc: {results[i]['Accuracy']:.2%}", fontsize=10)

    # 在格子中显示数值
    for i_ in range(cm.shape[0]):
        for j_ in range(cm.shape[1]):
            ax.text(j_, i_, f"{cm[i_, j_]}", ha="center", va="center",
                    color="white" if cm[i_, j_] > cm.max()/2 else "black", fontsize=8)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

# 隐藏多余的子图
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig('experiments/ablation_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n混淆矩阵图已保存至 experiments/ablation_confusion_matrices.png")