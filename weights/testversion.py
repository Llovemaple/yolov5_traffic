import torch

# 加载模型权重文件
weights_path = 'best.pt'  # 替换为你的模型权重文件路径
ckpt = torch.load(weights_path, map_location='cpu',weights_only=False)
# 查看元信息
if 'version' in ckpt:
    print(f"YOLOv5 version: {ckpt['version']}")
else:
    print("No version information found in the model weights.")