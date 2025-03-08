# 全局变量存储检测区域
_detection_zone = None

def set_detection_zone(points):
    """设置检测区域"""
    global _detection_zone
    _detection_zone = points

def is_in_detection_zone(x, y):
    """检查点是否在检测区域内"""
    global _detection_zone
    if not _detection_zone or len(_detection_zone) < 3:
        print("警告: 检测区域未设置或点数不足")
        return False
        
    import cv2
    import numpy as np
    point = np.array([x, y], np.float32)
    polygon = np.array(_detection_zone, np.float32)
    result = cv2.pointPolygonTest(polygon, tuple(point), False)
    print(f"点 ({x}, {y}) 是否在区域内: {result >= 0}")
    return result >= 0

def statistics(signal, car_bboxes, detect_list, current_ids):
    """统计车辆"""
    vertical = 0
    horizontal = 0

    # 检查是否有检测区域
    if not _detection_zone or len(_detection_zone) < 3:
        print("警告: 未设置有效的检测区域")
        return 0, 0, detect_list

    # 打印调试信息
    print(f"\n当前帧检测到的车辆数: {len(car_bboxes)}")
    print(f"已记录的车辆ID: {detect_list}")
    print(f"当前帧的车辆ID: {current_ids}")
    print(f"当前信号灯状态: {signal}")

    # 遍历所有车辆的边界框
    for x1, y1, x2, y2, track_id in car_bboxes:
        # 计算车辆中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 检查车辆中心点是否在检测区域内
        if is_in_detection_zone(center_x, center_y):
            print(f"车辆 ID {track_id} 在检测区域内")
            if track_id not in detect_list:
                print(f"新车辆进入检测区域: ID {track_id}")
                detect_list.append(track_id)
                if signal == 'red':
                    vertical += 1
                    print(f"纵向车流 +1 (ID: {track_id})")
                else:
                    horizontal += 1
                    print(f"横向车流 +1 (ID: {track_id})")

    print(f"本次统计: 纵向 {vertical}, 横向 {horizontal}")
    return vertical, horizontal, detect_list

class StatisticsPerformance:
    def __init__(self):
        self.total_vehicles = 0
        self.correct_counts = 0
        self.missed_counts = 0
        self.false_counts = 0
        self.processing_times = []
        
    def update_metrics(self, actual_count, measured_count, processing_time):
        self.total_vehicles += actual_count
        self.correct_counts += min(actual_count, measured_count)
        self.missed_counts += max(0, actual_count - measured_count)
        self.false_counts += max(0, measured_count - actual_count)
        self.processing_times.append(processing_time)
        
    def get_statistics(self):
        if self.total_vehicles == 0:
            return {
                "accuracy": 0,
                "recall": 0,
                "precision": 0,
                "avg_processing_time": 0
            }
            
        accuracy = self.correct_counts / self.total_vehicles
        precision = self.correct_counts / (self.correct_counts + self.false_counts) if (self.correct_counts + self.false_counts) > 0 else 0
        recall = self.correct_counts / (self.correct_counts + self.missed_counts) if (self.correct_counts + self.missed_counts) > 0 else 0
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "avg_processing_time": avg_time,
            "total_vehicles": self.total_vehicles,
            "missed_counts": self.missed_counts,
            "false_counts": self.false_counts
        }

class VehicleCounter:
    def __init__(self):
        self.performance = StatisticsPerformance()
        self.debug_mode = True
        
    def update_count(self, vehicle_boxes, current_time):
        """更新车辆计数"""
        import time
        start_time = time.time()
        
        # 原有的计数逻辑
        count_result = statistics(None, vehicle_boxes, [], [])
        
        # 更新性能统计
        processing_time = time.time() - start_time
        if hasattr(self, 'ground_truth_count'):  # 如果有真实值用于对比
            self.performance.update_metrics(
                self.ground_truth_count,
                count_result[0] + count_result[1],
                processing_time
            )
        
        if self.debug_mode:
            stats = self.performance.get_statistics()
            print(f"车流量统计性能: {stats}")
            print(f"处理时间: {processing_time:.3f}秒")
        
        return count_result

    def set_ground_truth(self, count):
        """设置真实车流量，用于精度评估"""
        self.ground_truth_count = count
