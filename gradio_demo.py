import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

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

def predict(image):
    # image 是 PIL Image
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

# 创建界面
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Dataframe(headers=["Model", "Predicted Class", "Confidence"]),
    title="智能相册分类助手 - 模块化改进对比",
    description="上传一张自然场景图片，查看四个阶段模型的预测结果。",
)

iface.launch()