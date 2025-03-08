import torch  # PyTorch库
import torchvision.transforms as transforms  # 图像转换模块
import numpy as np  # 数组处理
import cv2  # OpenCV库，用于图像处理
import logging  # 日志记录模块

from .model import Net  # 导入自定义的神经网络模型


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        """
        构造函数，初始化Extractor类。

        参数:
        - model_path: 预训练模型的路径
        - use_cuda: 是否使用GPU
        """
        self.net = Net(reid=True)  # 创建一个ReID网络实例
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"  # 根据是否有GPU选择设备
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']  # 加载模型权重
        self.net.load_state_dict(state_dict)  # 将加载的权重应用到网络
        logger = logging.getLogger("root.tracker")  # 创建日志记录器
        logger.info("Loading weights from {}... Done!".format(model_path))  # 记录加载模型的日志
        self.net.to(self.device)  # 将模型移动到指定的设备（GPU或CPU）
        self.size = (64, 128)  # 设置图像的目标尺寸
        # 定义图像预处理操作，包括转化为Tensor和归一化
        self.norm = transforms.Compose([
            transforms.ToTensor(),  # 转换为Tensor格式
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 归一化
        ])

    def _preprocess(self, im_crops):
        """
        图像预处理方法，按要求对图像进行处理。

        参数:
        - im_crops: 输入的图像切片（列表格式）

        返回:
        - im_batch: 处理后的图像批次
        """

        def _resize(im, size):
            """
            辅助函数，将图像按给定尺寸进行缩放。

            参数:
            - im: 输入图像
            - size: 目标尺寸

            返回:
            - resized: 缩放后的图像
            """
            return cv2.resize(im.astype(np.float32) / 255., size)  # 将图像归一化到[0, 1]并缩放

        # 对每个图像切片进行预处理：缩放、转为Tensor、归一化，并合并为一个batch
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        """
        对外接口，接受图像切片并提取特征。

        参数:
        - im_crops: 输入的图像切片（列表格式）

        返回:
        - features: 提取的特征
        """
        # 预处理输入的图像切片
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():  # 不计算梯度
            im_batch = im_batch.to(self.device)  # 将图像数据移到GPU或CPU
            features = self.net(im_batch)  # 将预处理后的图像传入网络，提取特征
        return features.cpu().numpy()  # 将特征从GPU移到CPU并转为NumPy数组


# 测试代码
if __name__ == '__main__':
    img = cv2.imread("demo.png")[:, :, (2, 1, 0)]  # 读取图像并转换BGR为RGB格式
    extr = Extractor("checkpoint/ckpt.t7")  # 创建Extractor实例并加载预训练模型
    feature = extr([img])  # 提取图像特征
    print(np.linalg.norm(feature))
    print(feature.shape)  # 输出特征的形状
