import torch
print(torch.backends.cudnn.version())
# 能够正确返回
from torch.backends import cudnn  # 若正常则静默
if torch.cuda.is_available():
    print("GPU可用")
    print("GPU数量:", torch.cuda.device_count())
    print("GPU名称:", torch.cuda.get_device_name(0))
    print("当前GPU索引:", torch.cuda.current_device())
else:
    print("GPU不可用，将使用CPU")
a = torch.tensor(1.)
print(cudnn.is_acceptable(a.cuda()))
# 若正常返回True
