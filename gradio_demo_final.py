import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 导入模型定义
from models.simple_cnn import SimpleCNN
from models.simple_resnet import SimpleResNet
from models.se_resnet import SEResNet

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 类别名称
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- 加载模型 ----------
# 阶段1
model1 = SimpleCNN(num_classes=6).to(device)
model1.load_state_dict(torch.load('checkpoints/stage1_best.pth', map_location=device))
model1.eval()

# 阶段4
model4 = SimpleResNet(num_classes=6).to(device)
model4.load_state_dict(torch.load('checkpoints/stage4_best.pth', map_location=device))
model4.eval()

# 阶段7（原始 SEResNet）
class SEResNetOriginal(SEResNet):
    def __init__(self, num_classes=6, reduction=16):
        super().__init__(num_classes=num_classes, reduction=reduction)

model7 = SEResNetOriginal(num_classes=6, reduction=16).to(device)
model7.load_state_dict(torch.load('checkpoints/stage7_optim_best.pth', map_location=device))
model7.eval()

# 阶段12 集成模型（加载 stage7,8,10,11 并平均）
# 需要定义带 Dropout 的 SEResNet（用于 stage10,11）
class SEResNetWithDropout(SEResNet):
    def __init__(self, num_classes=6, reduction=16):
        super().__init__(num_classes=num_classes, reduction=reduction)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

# 加载集成所需的四个模型
model7_ens = SEResNetOriginal(num_classes=6, reduction=16).to(device)
model7_ens.load_state_dict(torch.load('checkpoints/stage7_optim_best.pth', map_location=device))
model7_ens.eval()

model8 = SEResNetOriginal(num_classes=6, reduction=16).to(device)
model8.load_state_dict(torch.load('checkpoints/stage8_weighted_best.pth', map_location=device))
model8.eval()

model10 = SEResNetWithDropout(num_classes=6, reduction=16).to(device)
model10.load_state_dict(torch.load('checkpoints/stage10_advanced_best.pth', map_location=device))
model10.eval()

model11 = SEResNetWithDropout(num_classes=6, reduction=16).to(device)
model11.load_state_dict(torch.load('checkpoints/stage11_80epoch_best.pth', map_location=device))
model11.eval()

ensemble_models = [model7_ens, model8, model10, model11]

print("所有模型加载完成。")

# ---------- Grad-CAM 工具函数 ----------
def grad_cam(model, img_tensor, target_layer, target_class=None):
    """生成 Grad-CAM 热力图"""
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    handle_forward.remove()
    handle_backward.remove()

    grads = gradients[0].cpu().data.numpy().squeeze()
    acts = activations[0].cpu().data.numpy().squeeze()
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[3], img_tensor.shape[2]))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam, target_class

def get_target_layer(model, stage):
    """根据阶段返回适合的 Grad-CAM 目标层"""
    if stage == 'Stage1':
        # SimpleCNN 的最后一个卷积层
        return model.conv3
    elif stage == 'Stage4':
        # SimpleResNet 的最后一个残差块
        if hasattr(model, 'layer4'):
            return model.layer4[-1] if isinstance(model.layer4, (list, tuple, nn.Sequential)) else model.layer4
        else:
            # 如果没有 layer4，尝试找最后一个卷积层
            return model.conv1  # 备选
    else:
        # 对于所有 SEResNet 模型（Stage7, Stage10, Stage12），取最后一个残差块
        if hasattr(model, 'layer4'):
            if isinstance(model.layer4, (list, tuple, nn.Sequential)):
                return model.layer4[-1]
            else:
                return model.layer4
        else:
            print(f"警告: 未找到阶段 {stage} 的目标层")
            return None

# ---------- 预测函数 ----------
def predict_with_cam(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 单个模型预测
    models_single = {
        'Stage1': model1,
        'Stage4': model4,
        'Stage7': model7
    }
    results = []
    preds = {}

    # 单模型预测
    for name, model in models_single.items():
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            results.append([name, class_names[pred.item()], f"{conf.item():.4f}"])
            preds[name] = pred.item()

    # 集成模型预测（阶段12）
    with torch.no_grad():
        probs_list = []
        for model in ensemble_models:
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            probs_list.append(probs.cpu().numpy())
        avg_probs = np.mean(probs_list, axis=0)
        conf = np.max(avg_probs)
        pred = np.argmax(avg_probs)
        results.append(['Stage12 (Ensemble)', class_names[pred], f"{conf:.4f}"])
        preds['Stage12'] = pred

    # 生成 Grad-CAM 热力图（阶段1、4、7 和阶段10作为阶段12代表）
    cam_images = []
    cam_labels = []
    stages_cam = ['Stage1', 'Stage4', 'Stage7', 'Stage10']  # 用Stage10代表Stage12的热力图
    models_cam = [model1, model4, model7, model10]

    for stage, model in zip(stages_cam, models_cam):
        target_layer = get_target_layer(model, stage)
        if target_layer is None:
            raise ValueError(f"无法为阶段 {stage} 生成 Grad-CAM，目标层不存在")
        # 获取该阶段对应的预测类别（如果不存在，用阶段12的预测）
        target_class = preds.get(stage, preds['Stage12'])
        cam, _ = grad_cam(model, img_tensor, target_layer, target_class=target_class)
        cam_images.append(cam)

    # 将热力图叠加到原图上
    img_np = np.array(image.resize((150, 150))) / 255.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    titles = ['Stage1', 'Stage4', 'Stage7', 'Stage12 (using Stage10)']
    for i, (ax, cam, title) in enumerate(zip(axes, cam_images, titles)):
        ax.imshow(img_np)
        ax.imshow(cam, cmap='jet', alpha=0.5)
        # 获取该阶段对应的预测类别显示在标题中
        stage_name = stages_cam[i]
        pred_class = class_names[preds.get(stage_name, preds['Stage12'])]
        ax.set_title(f"{title} (pred: {pred_class})")
        ax.axis('off')

    cam_path = 'cam_comparison_final.png'
    plt.tight_layout()
    plt.savefig(cam_path)
    plt.close()

    return results, cam_path

# ---------- Gradio 界面 ----------
iface = gr.Interface(
    fn=predict_with_cam,
    inputs=gr.Image(type='pil'),
    outputs=[
        gr.Dataframe(headers=["Stage", "Predicted Class", "Confidence"]),
        gr.Image(type='filepath', label="Grad-CAM (Stage1,4,7,12)")
    ],
    title="智能相册分类助手 - 最终模型对比",
    description="上传一张自然场景图片，查看阶段1、4、7和集成模型（阶段12）的预测结果，并比较四个阶段的热力图。"
)

iface.launch()