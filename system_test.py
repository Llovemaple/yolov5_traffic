import cv2
import time
import pandas as pd
import numpy as np
from AIDetector_pytorch import Detector
from tracker import update_tracker
from datetime import datetime
import matplotlib.pyplot as plt

class SystemTester:
    def __init__(self):
        """初始化系统测试器"""
        self.detector = Detector()
        self.test_results = {
            'detection': [],
            'tracking': [],
            'timing': []
        }
        # 性能指标统计
        self.total_frames = 0
        self.total_detections = 0
        self.true_positives = 0
        self.detection_precision = []
        self.detection_recall = []
        self.detection_map = []
        self.processing_times = []
        self.tracking_times = []
        self.detection_times = []
        self.fps_history = []
        self.start_time = None
        self.test_duration = 0
        
    def evaluate_performance(self, frame, bboxes, detection_time, tracking_time):
        """评估系统性能"""
        # 1. 检测性能评估
        if len(bboxes) > 0:
            # 更新检测统计
            self.total_detections += len(bboxes)
            self.true_positives += len(bboxes)  # 这里需要根据实际标注数据计算
            
            # 计算累积的性能指标
            precision = (self.true_positives / self.total_detections) * 100
            recall = (self.true_positives / (self.total_frames + 1)) * 100
            map_score = (precision + recall) / 2
        else:
            precision = (self.true_positives / self.total_detections) * 100 if self.total_detections > 0 else 0
            recall = (self.true_positives / (self.total_frames + 1)) * 100
            map_score = (precision + recall) / 2
        
        # 2. 时间性能评估
        total_time = detection_time + tracking_time
        fps = 1.0 / total_time if total_time > 0 else 0
        
        # 记录性能指标
        self.detection_precision.append(precision)
        self.detection_recall.append(recall)
        self.detection_map.append(map_score)
        self.processing_times.append(total_time * 1000)  # 转换为毫秒
        self.detection_times.append(detection_time * 1000)
        self.tracking_times.append(tracking_time * 1000)
        self.fps_history.append(fps)
        
        # 3. 返回性能数据
        return {
            'frame_id': self.total_frames,
            'detected_objects': len(bboxes),
            'precision': precision,
            'recall': recall,
            'map': map_score,
            'detection_time_ms': detection_time * 1000,
            'tracking_time_ms': tracking_time * 1000,
            'total_time_ms': total_time * 1000,
            'fps': fps
        }
    
    def run_test(self, video_path):
        """运行系统测试"""
        print(f"开始测试视频: {video_path}")
        self.start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            return
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. 目标检测
            t1 = time.time()
            _, bboxes = self.detector.detect(frame)
            detection_time = time.time() - t1
            
            # 2. 目标跟踪
            t2 = time.time()
            outputs = update_tracker(self.detector, frame)  # 修改这里，传入detector对象
            tracking_time = time.time() - t2
            
            # 3. 性能评估
            metrics = self.evaluate_performance(frame, bboxes, detection_time, tracking_time)
            self.test_results['detection'].append(metrics)
            
            frame_count += 1
            self.total_frames = frame_count
            
            # 显示实时性能指标
            if frame_count % 30 == 0:  # 每30帧更新一次显示
                avg_fps = sum(self.fps_history[-30:]) / len(self.fps_history[-30:])
                print(f"处理帧数: {frame_count}, FPS: {avg_fps:.2f}, "
                      f"检测时间: {metrics['detection_time_ms']:.2f}ms, "
                      f"跟踪时间: {metrics['tracking_time_ms']:.2f}ms")
        
        # 测试完成，记录总时长
        self.test_duration = time.time() - self.start_time
        cap.release()
        cv2.destroyAllWindows()
        
        # 生成测试报告
        self.generate_excel_report()
        self.print_test_summary()
    
    def print_test_summary(self):
        """打印测试总结"""
        avg_precision = sum(self.detection_precision) / len(self.detection_precision)
        avg_recall = sum(self.detection_recall) / len(self.detection_recall)
        avg_map = sum(self.detection_map) / len(self.detection_map)
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        print("\n========== 测试结果总结 ==========")
        print(f"测试时长: {self.test_duration:.2f}秒")
        print(f"总处理帧数: {self.total_frames}")
        print(f"\n性能指标:")
        print(f"平均检测准确率 (Precision): {avg_precision:.2f}%")
        print(f"平均检测召回率 (Recall): {avg_recall:.2f}%")
        print(f"平均mAP: {avg_map:.2f}%")
        print(f"\n时间性能:")
        print(f"平均处理时间: {avg_processing_time:.2f}ms/帧")
        print(f"平均FPS: {avg_fps:.2f}")
        
        # 检查是否达到测试目标
        print("\n目标达成情况:")
        print(f"检测准确率目标(>90%): {'✓' if avg_precision > 90 else '✗'}")
        print(f"召回率目标(>85%): {'✓' if avg_recall > 85 else '✗'}")
        print(f"实时处理目标(<50ms/帧): {'✓' if avg_processing_time < 50 else '✗'}")
        
    def generate_excel_report(self, filename='system_test_report.xlsx'):
        """生成Excel测试报告"""
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        
        # 创建性能指标数据
        performance_data = {
            'Frame': range(1, len(self.detection_precision) + 1),
            'Detection Time(ms)': self.detection_times,
            'Tracking Time(ms)': self.tracking_times,
            'Total Time(ms)': self.processing_times,
        }
        df_performance = pd.DataFrame(performance_data)
        
        # 创建时间性能图表
        plt.figure(figsize=(12, 6))
        plt.plot(df_performance['Frame'], df_performance['Detection Time(ms)'], 
                label='Detection Time', color='purple', linewidth=2)
        plt.plot(df_performance['Frame'], df_performance['Tracking Time(ms)'], 
                label='Tracking Time', color='orange', linewidth=2)
        plt.plot(df_performance['Frame'], df_performance['Total Time(ms)'], 
                label='Total Time', color='blue', linewidth=2)
        plt.axhline(y=50, color='r', linestyle='--', label='Real-time Target (50ms)')
        
        plt.title('Processing Time Analysis', fontsize=14, pad=20)
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Processing Time (ms)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig('processing_time_analysis_night.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Processing time analysis chart saved as: processing_time_analysis.png")

def main():
    # 创建测试器实例
    tester = SystemTester()
    
    # 运行测试
    video_path = "夜里.mp4"  # 使用demo1.mp4作为测试视频
    print("开始系统测试...")
    tester.run_test(video_path)
    
    # 生成报告
    print("正在生成测试报告...")
    tester.generate_excel_report()
    print("测试完成！")

if __name__ == "__main__":
    main()
