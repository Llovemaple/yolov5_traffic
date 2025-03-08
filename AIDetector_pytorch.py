import torch
import numpy as np
from models.experimental import attempt_load  # 导入加载YOLO模型的函数
from utils.general import non_max_suppression, scale_coords  # 导入非最大抑制和坐标缩放函数
from utils.BaseDetector import baseDet  # 导入基础检测器类
from utils.torch_utils import select_device  # 导入选择设备的函数
from utils.datasets import letterbox  # 导入letterbox函数，用于调整图像大小

class Detector(baseDet):  # 创建Detector类，继承自baseDet类

    def __init__(self):
        super(Detector, self).__init__()  # 调用父类的构造函数
        
        # 详细的CUDA诊断信息
        print("\n=== CUDA 诊断信息 ===")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"当前CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"当前GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
        else:
            print("警告: CUDA不可用，将使用CPU进行推理")
            print("可能的原因:")
            print("1. PyTorch未安装CUDA版本")
            print("2. NVIDIA驱动未正确安装")
            print("3. CUDA工具包未正确安装")
        
        # 设备选择
        try:
            if not torch.cuda.is_available():
                print("\n尝试强制使用CUDA...")
                torch.cuda.init()
        except Exception as e:
            print(f"强制使用CUDA失败: {e}")
        
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        print(f"\n最终使用设备: {self.device}")
        
        # 在 GPU 模式下启用半精度
        self.half = self.device != 'cpu'
        print(f"是否使用半精度: {self.half}\n")
        
        self.init_model()  # 初始化模型
        self.build_config()  # 构建配置

    def init_model(self):
        # 初始化YOLOv5模型
        self.weights = 'weights/best.pt'  # 设置模型的权重文件路径
        model = attempt_load(self.weights, device=self.device)  # 加载YOLO模型
        model.to(self.device).eval()  # 将模型移动到指定设备，并切换到评估模式
        if self.half:
            model.half()  # GPU 半精度
        self.m = model  # 保存加载的模型
        self.names = model.module.names if hasattr(model, 'module') else model.names  # 获取模型的类别名称

    def preprocess(self, img):
        # 图像预处理函数
        img0 = img.copy()  # 保存原始图像
        img = letterbox(img, new_shape=self.img_size)[0]  # 调整图像大小，使其适应模型输入要求
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR转RGB，并调整维度顺序（HWC->CHW）
        img = np.ascontiguousarray(img)  # 确保图像内存连续
        img = torch.from_numpy(img).to(self.device)  # 将图像转换为Tensor，并移动到指定设备
        img = img.half() if self.half else img.float()  # 根据模式使用半精度或单精度
        img /= 255.0  # 归一化到[0, 1]范围
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 如果是单张图像，则增加一个batch维度

        return img0, img  # 返回原始图像和处理后的图像

    def detect(self, im):
        # 检测函数
        im0, img = self.preprocess(im)  # 预处理图像
        pred = self.m(img, augment=False)[0]  # 将图像传入模型，进行预测
        pred = pred.float()  # 转为浮点数类型
        pred = non_max_suppression(pred, self.threshold, 0.4)  # 应用非最大抑制，去除重复框
        pred_boxes = []  # 存储预测框
        for det in pred:
            # 如果检测到目标
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 调整坐标到原图大小

                for *x, conf, cls_id in det:  # 遍历每个目标框
                    lbl = self.names[int(cls_id)]  # 获取类别名称
                    if not lbl in ['car','green','red','yellow']:  # 只检测特定类别：人、红绿灯
                        continue
                    x1, y1 = int(x[0]), int(x[1])  # 获取框的左上角坐标
                    x2, y2 = int(x[2]), int(x[3])  # 获取框的右下角坐标
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))  # 将框的坐标、标签、置信度添加到结果列表中

        return im, pred_boxes  # 返回原始图像和预测的目标框
