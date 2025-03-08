import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入torch库，用于深度学习和张量操作
from .deep.feature_extractor import Extractor  # 导入特征提取器，用于从图像中提取深度特征
from .sort.nn_matching import NearestNeighborDistanceMetric  # 导入最近邻距离度量，常用于度量目标特征之间的相似度
from .sort.preprocessing import non_max_suppression  # 导入非最大抑制，用于去除冗余的检测框
from .sort.detection import Detection  # 导入检测对象，用于封装边界框、类别和特征等信息
from .sort.tracker import Tracker  # 导入目标追踪器，用于跟踪目标在视频帧中的位置

__all__ = ['DeepSort']  # 指定从模块中导出的类或函数，这里表示只导出DeepSort类

class DeepSort(object):  # 定义DeepSort类，包含了目标检测和追踪的功能
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        # 构造函数，初始化DeepSort的各项参数
        self.min_confidence = min_confidence  # 最小置信度，低于该置信度的检测结果会被忽略
        self.nms_max_overlap = nms_max_overlap  # 非最大抑制的最大重叠度，用于去除冗余的框

        self.extractor = Extractor(model_path, use_cuda=use_cuda)  # 初始化特征提取器，用于从图像中提取特征

        max_cosine_distance = max_dist  # 最大余弦距离，用于目标特征匹配
        nn_budget = 100  # 最近邻的缓存大小
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)  # 创建距离度量对象
        self.tracker = Tracker(  # 初始化目标追踪器
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, clss, ori_img):  # 更新函数，用于处理每一帧图像
        self.height, self.width = ori_img.shape[:2]  # 获取图像的高度和宽度

        # 生成检测对象
        features = self._get_features(bbox_xywh, ori_img)  # 提取每个目标的特征
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)  # 将边界框格式从xywh转换为tlwh（左上角坐标+宽高）
        detections = [Detection(bbox_tlwh[i], clss[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]  # 筛选出置信度大于最小值的检测结果，并封装成Detection对象

        # 更新目标追踪器
        self.tracker.predict()  # 预测当前帧中每个目标的运动状态
        self.tracker.update(detections)  # 更新追踪器的状态，基于当前的检测结果

        # 输出目标的边界框和ID
        outputs = []  # 存储最终输出的跟踪结果
        for track in self.tracker.tracks:  # 遍历所有的目标追踪
            if not track.is_confirmed() or track.time_since_update > 1:  # 如果目标未确认或已更新时间超过1帧，则跳过
                continue
            box = track.to_tlwh()  # 获取目标的边界框（左上角坐标+宽高）
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)  # 将tlwh格式转换为xyxy格式（左上和右下坐标）
            outputs.append((x1, y1, x2, y2, track.cls_, track.track_id))  # 将目标的坐标、类别和ID加入输出列表
        return outputs  # 返回所有跟踪到的目标

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):  # 将bbox从xywh格式转换为tlwh格式
        if isinstance(bbox_xywh, np.ndarray):  # 如果输入是numpy数组
            bbox_tlwh = bbox_xywh.copy()  # 复制数组
        elif isinstance(bbox_xywh, torch.Tensor):  # 如果输入是torch张量
            bbox_tlwh = bbox_xywh.clone()  # 克隆张量
        if bbox_tlwh.size(0):  # 如果输入的bbox不是空的
            bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.  # 计算左上角x坐标
            bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.  # 计算左上角y坐标
        return bbox_tlwh  # 返回转换后的边界框

    def _xywh_to_xyxy(self, bbox_xywh):  # 将xywh格式转换为xyxy格式
        x, y, w, h = bbox_xywh  # 解包bbox的xywh参数
        x1 = max(int(x - w / 2), 0)  # 计算左上角的x坐标
        x2 = min(int(x + w / 2), self.width - 1)  # 计算右下角的x坐标，确保在图像宽度范围内
        y1 = max(int(y - h / 2), 0)  # 计算左上角的y坐标
        y2 = min(int(y + h / 2), self.height - 1)  # 计算右下角的y坐标，确保在图像高度范围内
        return x1, y1, x2, y2  # 返回转换后的xyxy格式边界框

    def _tlwh_to_xyxy(self, bbox_tlwh):  # 将tlwh格式转换为xyxy格式
        x, y, w, h = bbox_tlwh  # 解包bbox的tlwh参数
        x1 = max(int(x), 0)  # 计算左上角的x坐标
        x2 = min(int(x + w), self.width - 1)  # 计算右下角的x坐标，确保在图像宽度范围内
        y1 = max(int(y), 0)  # 计算左上角的y坐标
        y2 = min(int(y + h), self.height - 1)  # 计算右下角的y坐标，确保在图像高度范围内
        return x1, y1, x2, y2  # 返回转换后的xyxy格式边界框

    def _xyxy_to_tlwh(self, bbox_xyxy):  # 将xyxy格式转换为tlwh格式
        x1, y1, x2, y2 = bbox_xyxy  # 解包xyxy格式的边界框
        t = x1  # 左上角的x坐标
        l = y1  # 左上角的y坐标
        w = int(x2 - x1)  # 计算宽度
        h = int(y2 - y1)  # 计算高度
        return t, l, w, h  # 返回转换后的tlwh格式边界框

    def _get_features(self, bbox_xywh, ori_img):  # 提取每个目标的特征
        im_crops = []  # 存储裁剪出来的目标图像
        for box in bbox_xywh:  # 遍历所有输入的边界框
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)  # 将每个目标的边界框转换为xyxy格式
            im = ori_img[y1:y2, x1:x2]  # 从原始图像中裁剪出目标区域
            im_crops.append(im)  # 将裁剪出的目标添加到im_crops列表
        if im_crops:  # 如果裁剪出了目标
            features = self.extractor(im_crops)  # 调用特征提取器提取目标特征
        else:
            features = np.array([])  # 如果没有目标，返回空的特征数组
        return features  # 返回提取的特征
