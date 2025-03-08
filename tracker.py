from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import time
import logging

# 配色，用于绘制框的颜色
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# 获取配置文件并加载DeepSORT配置
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")

# 创建DeepSORT追踪实例
deepsort = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE,
    n_init=cfg.DEEPSORT.N_INIT,
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=True
)

class TrackerPerformance:
    def __init__(self):
        self.track_time = []  # 跟踪时间统计
        self.id_switches = 0  # ID切换次数
        self.total_tracks = 0 # 总跟踪目标数
        self.occlusion_recoveries = 0  # 遮挡恢复次数
        
    def update_metrics(self, time_cost, id_switch=False, occlusion_recovery=False):
        self.track_time.append(time_cost)
        if id_switch:
            self.id_switches += 1
        if occlusion_recovery:
            self.occlusion_recoveries += 1
            
    def get_statistics(self):
        avg_time = sum(self.track_time) / len(self.track_time) if self.track_time else 0
        return {
            "average_track_time": avg_time,
            "id_switches": self.id_switches,
            "total_tracks": self.total_tracks,
            "occlusion_recoveries": self.occlusion_recoveries
        }

# 创建性能统计实例
tracker_performance = TrackerPerformance()

def plot_bboxes(image, bboxes, line_thickness=None):
    """
    在图像上绘制目标的边界框。
    """
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 0, 255) if cls_id == 'car' else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(f'{cls_id} ID-{pos_id}', 0, fontScale=tl / 3, thickness=tf)[0]
        cv2.rectangle(image, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), color, -1, cv2.LINE_AA)
        cv2.putText(image, f'{cls_id} ID-{pos_id}', (x1, y1 - 2), 0, tl / 3, (225, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
    return image

def update_tracker(target_detector, image):
    """
    更新目标追踪器，检测新目标并更新追踪状态。
    """
    import time
    start_time = time.time()
    
    new_cars = []
    _, bboxes = target_detector.detect(image)
    
    # 记录当前跟踪的ID
    previous_ids = set([track.track_id for track in deepsort.tracker.tracks])
    
    bbox_xywh = []
    confs = []
    clss = []
    
    for x1, y1, x2, y2, cls_id, conf in bboxes:
        obj = [
            int((x1 + x2) / 2), int((y1 + y2) / 2),
            x2 - x1, y2 - y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)
    
    outputs = deepsort.update(xywhs, confss, clss, image)
    
    # 检测ID切换和遮挡恢复
    current_ids = set([track.track_id for track in deepsort.tracker.tracks])
    id_switch = len(current_ids.symmetric_difference(previous_ids)) > 0
    occlusion_recovery = len(current_ids - previous_ids) > 0
    
    # 更新性能统计
    track_time = time.time() - start_time
    tracker_performance.update_metrics(
        track_time,
        id_switch=id_switch,
        occlusion_recovery=occlusion_recovery
    )
    tracker_performance.total_tracks = len(current_ids)
    
    # 记录跟踪结果
    bboxes2draw = []
    car_bboxes = []
    current_ids = set()
    signal = None

    for value in outputs:
        x1, y1, x2, y2, cls_, track_id = value
        bboxes2draw.append((x1, y1, x2, y2, cls_, track_id))
        current_ids.add(track_id)
        if cls_ == 'car':
            if track_id not in target_detector.carTracker:
                target_detector.carTracker[track_id] = 0
                new_cars.append((image[y1:y2, x1:x2], track_id))
            car_bboxes.append((x1, y1, x2, y2, track_id))
        if cls_ in ['green', 'red', 'yellow']:
            signal = cls_
            logging.info(f"当前信号灯状态: {signal}")

    # 删除失去追踪的 ID
    for history_id in list(target_detector.carTracker.keys()):
        if history_id not in current_ids:
            target_detector.carTracker[history_id] -= 1
            if target_detector.carTracker[history_id] < -5:
                target_detector.carTracker.pop(history_id)
                logging.info(f'-[INFO] Delete track id: {history_id}')

    # 绘制边界框
    image = plot_bboxes(image, bboxes2draw)
    logging.info(tracker_performance.get_statistics())
    return image, new_cars, car_bboxes, signal, list(current_ids)

logging.basicConfig(level=logging.INFO)