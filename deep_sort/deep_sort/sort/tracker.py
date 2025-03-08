# vim: expandtab:ts=4:sw=4
# 设置Vim编辑器的缩进和制表符设置
from __future__ import absolute_import
# 确保导入的模块是绝对导入，避免与本地模块冲突
import numpy as np
# 导入NumPy库，用于数值计算
from . import kalman_filter
# 导入自定义的卡尔曼滤波模块
from . import linear_assignment
# 导入自定义的线性分配模块
from . import iou_matching
# 导入自定义的IOU匹配模块
from .track import Track
# 导入自定义的Track类，用于表示跟踪目标
class Tracker:
    # 定义Tracker类，用于管理多个目标的跟踪
    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        # 初始化函数，设置跟踪器的参数
        self.metric = metric
        # 用于计算特征距离的度量对象
        self.max_iou_distance = max_iou_distance
        # IOU匹配的最大距离阈值
        self.max_age = max_age
        # 目标在丢失后仍保留的最大帧数
        self.n_init = n_init
        # 目标被确认所需的最小连续检测次数

        self.kf = kalman_filter.KalmanFilter()
        # 初始化卡尔曼滤波器
        self.tracks = []
        # 用于存储当前所有跟踪目标的列表
        self._next_id = 1
        # 用于分配新跟踪目标的ID

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # 预测函数，用于将每个跟踪目标的状态向前传播一步
        for track in self.tracks:
            track.predict(self.kf)
            # 对每个跟踪目标调用预测方法，更新其状态

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        # 更新函数，用于处理当前帧的检测结果并更新跟踪目标
        # 运行匹配级联
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        # 获取匹配的跟踪目标和检测结果，以及未匹配的跟踪目标和检测结果

        # 更新跟踪目标集合
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            # 对匹配的跟踪目标进行更新
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            # 对未匹配的跟踪目标标记为丢失
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
            # 对未匹配的检测结果初始化新的跟踪目标
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # 删除已被标记为删除的跟踪目标

        # 更新距离度量
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # 获取当前所有确认的跟踪目标的ID
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
            # 收集所有确认跟踪目标的特征和对应的ID
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        # 更新距离度量模型
    def _match(self, detections):
        # 内部匹配函数，用于将检测结果与现有跟踪目标进行匹配

        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 定义门控度量函数，用于计算特征距离并应用门控
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            # 计算特征距离矩阵
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            # 应用门控，过滤掉不符合条件的匹配

            return cost_matrix

        # 将跟踪目标集合分为已确认和未确认的跟踪目标
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 使用外观特征关联已确认的跟踪目标
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        # 使用级联匹配算法进行匹配

        # 使用IOU关联剩余的跟踪目标和未确认的跟踪目标
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        # 使用最小成本匹配算法进行IOU匹配

        matches = matches_a + matches_b
        # 合并两种匹配结果
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        # 合并未匹配的跟踪目标
        return matches, unmatched_tracks, unmatched_detections
        # 返回匹配结果、未匹配的跟踪目标和未匹配的检测结果

    def _initiate_track(self, detection):
        # 初始化新的跟踪目标
        mean, covariance = self.kf.initiate(detection.to_xyah())
        # 使用卡尔曼滤波器初始化目标的状态均值和协方差
        self.tracks.append(Track(
            mean, detection.cls_, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        # 创建新的Track对象并添加到跟踪目标列表中
        self._next_id += 1
        # 更新下一个跟踪目标的ID