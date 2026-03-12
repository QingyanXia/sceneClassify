import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# 导入模型类
from models.simple_cnn import SimpleCNN
from models.cnn_bn_dropout import CNNWithBNDropout
from models.simple_resnet import SimpleResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model1 = SimpleCNN(num_classes=6).to(device)
model1.load_state_dict(torch.load('checkpoints/stage1_best.pth', map_location=device))
model1.eval()

model2 = CNNWithBNDropout(num_classes=6).to(device)
model2.load_state_dict(torch.load('checkpoints/stage2_best.pth', map_location=device))
model2.eval()

model3 = CNNWithBNDropout(num_classes=6).to(device)
model3.load_state_dict(torch.load('checkpoints/stage3_best.pth', map_location=device))
model3.eval()

model4 = SimpleResNet(num_classes=6).to(device)
model4.load_state_dict(torch.load('checkpoints/stage4_best.pth', map_location=device))
model4.eval()

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def grad_cam(model, img_tensor, target_layer, target_class=None):
    """
    生成 Grad-CAM 热力图
    model: 模型
    img_tensor: 输入图像 (1, C, H, W)
    target_layer: 目标卷积层 (如 model.layer4[-1])
    target_class: 指定类别索引，默认为模型预测的类别
    """
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # 前向传播
    output = model(img_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # 反向传播
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    # 计算权重和热力图
    grads = gradients[0].cpu().data.numpy().squeeze()  # [C, H, W]
    acts = activations[0].cpu().data.numpy().squeeze()  # [C, H, W]
    weights = np.mean(grads, axis=(1, 2))  # 全局平均池化

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (img_tensor.shape[3], img_tensor.shape[2]))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam, target_class

def predict(image):
    img = transform(image).unsqueeze(0).to(device)
    models = [model1, model2, model3, model4]
    names = ['SimpleCNN', '+BN+Dropout', '+Data Augmentation', '+Residual']
    results = []
    for model, name in zip(models, names):
        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            results.append([name, class_names[pred.item()], f"{conf.item():.4f}"])
    return results

def predict_with_cam(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    results = []
    cams = []

    # 阶段一
    cam1, pred1 = grad_cam(model1, img_tensor, model1.conv3)
    results.append(['SimpleCNN', class_names[pred1], f"{torch.softmax(model1(img_tensor),1).max().item():.4f}"])
    cams.append(cam1)

    # 阶段四
    cam4, pred4 = grad_cam(model4, img_tensor, model4.layer4[-1])
    results.append(['+Residual', class_names[pred4], f"{torch.softmax(model4(img_tensor),1).max().item():.4f}"])
    cams.append(cam4)

    # 可视化热力图叠加
    img_np = np.array(image.resize((150,150))) / 255.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_np)
    axes[0].imshow(cam1, cmap='jet', alpha=0.5)
    axes[0].set_title(f"SimpleCNN (pred: {class_names[pred1]})")
    axes[0].axis('off')

    axes[1].imshow(img_np)
    axes[1].imshow(cam4, cmap='jet', alpha=0.5)
    axes[1].set_title(f"ResNet (pred: {class_names[pred4]})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('cam_comparison.png')
    plt.close()

    return results, 'cam_comparison.png'

# 创建界面
iface = gr.Interface(
    fn=predict_with_cam,
    inputs=gr.Image(type='pil'),
    outputs=[
        gr.Dataframe(headers=["Model", "Predicted Class", "Confidence"]),
        gr.Image(type='filepath', label="Grad-CAM Comparison")
    ],
    title="智能相册分类助手 - 模块化改进对比",
    description="上传一张自然场景图片，查看预测结果与热力图对比。",
)
iface.launch()

# iface.launch(share=True)
iface.launch()