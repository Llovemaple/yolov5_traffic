import cv2
from PIL import Image, ImageTk
from vehicle_statistics import statistics
import numpy as np
import torch


def process_frame(det, prev_time, cap, detect_list):
    ret, frame = cap.read()
    if ret:
        # 记录GPU使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0)/1024**2
            if gpu_memory > 100:  # 如果GPU显存使用超过100MB
                print(f"GPU显存使用: {gpu_memory:.1f} MB")
        
        # 缩小图像尺寸以加快处理速度
        scale = 0.5  # 缩放比例
        frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # 处理缩小后的图像
        result = det.feedCap(frame_resized)
        
        # 将结果坐标还原到原始尺寸
        if result['car_bboxes']:
            for i in range(len(result['car_bboxes'])):
                result['car_bboxes'][i] = tuple(int(x/scale) for x in result['car_bboxes'][i][:4]) + result['car_bboxes'][i][4:]
        
        frame = result['frame']
        signal = result['signal']
        car_bboxes = result['car_bboxes']
        current_ids = result['current_ids']
        
        horizontal, vertical, detect_list = statistics(signal, car_bboxes, detect_list, current_ids)
        
        # 计算 FPS
        current_time = cv2.getTickCount()
        if prev_time != 0:
            time_diff = (current_time - prev_time) / cv2.getTickFrequency()
            fps = 1 / time_diff
        else:
            fps = 0
        prev_time = current_time
        
        # 使用 numpy 操作代替 PIL 转换
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return prev_time, frame, photo, vertical, horizontal, detect_list
    return None, None, None, 0, 0, detect_list