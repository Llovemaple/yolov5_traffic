import cv2
import tkinter as tk
from tkinter import Label, Button, filedialog, messagebox, Text, Scrollbar
from PIL import Image, ImageTk
import threading
from AIDetector_pytorch import Detector
from video_processing import process_frame  # 引入处理每一帧的函数
from car_flow import car_flow_control, controller
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

class VideoGUI:
    prev_time = 0

    def __init__(self):
        self.det = Detector()
        self.signed_id = None
        self.vertical_count = None
        self.horizontal_count = None
        self.vehicle_tracks = None
        self.detect_list = []
        self.cap = None
        self.thread = None
        self.video_src = None  # 视频源路径初始化为None
        self.window = tk.Tk()
        self.time=0
        # 获取屏幕的宽度和高度
        self.window.state("zoomed")  # 最大化窗口
        self.window.title("车辆检测")
        self.window.configure(bg="#F0F0F0")  # 设置窗口背景颜色为浅灰色

        # 创建左侧主Frame，设置固定大小
        self.left_frame = tk.Frame(self.window, bg="#F0F0F0")
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.left_frame.grid_propagate(False)  # 防止自动调整大小

        # 创建右侧主Frame，设置固定大小
        self.right_frame = tk.Frame(self.window, bg="#F0F0F0")
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.right_frame.grid_propagate(False)  # 防止自动调整大小

        # 左侧视频显示区域Frame，设置最小大小
        self.frame_video = tk.Frame(
            self.left_frame, 
            bg="#FFFFFF", 
            bd=5, 
            relief="sunken",
            width=1200,   # 设置固定宽度
            height=600    # 设置固定高度
        )
        self.frame_video.grid(row=0, column=0, sticky="nsew")
        self.frame_video.grid_propagate(False)  # 防止自动调整大小

        # 左侧视频Label
        self.label = Label(
            self.frame_video, 
            bd=5, 
            relief="solid", 
            bg="#000000",
            text="等待选择视频...",  # 添加默认文本
            font=('Helvetica', 14),
            fg="#FFFFFF"
        )
        self.label.place(relwidth=1, relheight=1)  # 使用place布局填充整个Frame

        # 左侧车流量标签
        self.flow_label = Label(
            self.left_frame, 
            text="车流: 等待统计...", 
            font=('Helvetica', 16),
            bg="#F0F0F0", 
            fg="#333333"
        )
        self.flow_label.grid(row=1, column=0, sticky="w", pady=(10,5))

        # 左侧检测信息区域
        self.detection_frame = tk.Frame(self.left_frame, bg="#F0F0F0")
        self.detection_frame.grid(row=2, column=0, sticky="nsew")

        # 检测信息标签
        self.detection_label = Label(
            self.detection_frame,
            text="实时检测信息:",
            font=('Helvetica', 14),
            bg="#F0F0F0",
            fg="#333333"
        )
        self.detection_label.pack(anchor="w", pady=(0,5))

        # 检测信息文本框
        self.detection_text = Text(
            self.detection_frame,
            height=8,
            font=('Helvetica', 12),
            bg="#FFFFFF",
            fg="#333333",
            relief="solid",
            bd=1
        )
        self.detection_text.pack(side="left", fill="both", expand=True)

        # 滚动条
        scrollbar = Scrollbar(self.detection_frame, command=self.detection_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.detection_text.config(yscrollcommand=scrollbar.set)

        # 右侧配时信息Label
        self.right_label = Label(
            self.right_frame, 
            text="红绿灯配时建议\n\n等待车流量统计...", 
            font=('Helvetica', 16), 
            bg="#FFFFFF",
            fg="#333333",
            bd=2, 
            relief="solid", 
            padx=20, 
            pady=20,
            width=25,
            justify=tk.LEFT
        )
        self.right_label.grid(row=0, column=0, sticky="nsew", pady=(0,20))

        # 创建按钮Frame（放在右侧Frame底部）
        self.button_frame = tk.Frame(self.right_frame, bg="#F0F0F0")
        self.button_frame.grid(row=1, column=0, sticky="sew")

        # 按钮样式
        button_style = {
            'font': ('Helvetica', 12),
            'width': 15,
            'height': 2,
            'bg': '#4CAF50',
            'fg': 'white',
            'relief': 'raised',
            'cursor': 'hand2'
        }

        # 添加按钮
        self.select_button = Button(
            self.button_frame,
            text="选择视频",
            command=self.select_video,
            **button_style
        )
        self.select_button.pack(pady=5)

        self.draw_zone_button = Button(
            self.button_frame,
            text="绘制检测区域",
            command=self.start_drawing,
            state=tk.DISABLED,
            **button_style
        )
        self.draw_zone_button.pack(pady=5)

        self.start_button = Button(
            self.button_frame,
            text="开始检测",
            command=self.start_detection,
            state=tk.DISABLED,
            **button_style
        )
        self.start_button.pack(pady=5)

        # 添加预览图像变量
        self.preview_photo = None
        self.video_loaded = False

        # 添加绘制区域相关变量
        self.drawing = False
        self.points = []
        self.temp_points = []
        
        # 绑定鼠标事件
        self.label.bind('<Button-1>', self.on_click)
        self.label.bind('<Motion>', self.on_move)
        self.label.bind('<Button-3>', self.finish_drawing)

        # 添加图像尺寸变量
        self.display_width = 960
        self.display_height = 540
        self.original_width = None
        self.original_height = None

        # 添加性能优化相关变量
        self.frame_skip = 2  # 跳帧处理
        self.frame_count = 0
        self.process_queue = Queue(maxsize=5)  # 处理队列
        self.display_queue = Queue(maxsize=5)  # 显示队列

        # 添加计时变量
        self.time_counter = 0
        self.update_interval = 30  # 每30帧更新一次配时建议

        self.count_start_time = None  # 计数开始时间
        self.count_duration = 180  # 计数持续时间（3分钟 = 180秒）
        self.total_horizontal = 0  # 总横向车流量
        self.total_vertical = 0    # 总纵向车流量
        self.is_counting = False   # 是否正在计数

        self.init_info()

        # 设置窗口最小大小
        self.window.minsize(1600, 900)

        # 计算并设置固定大小
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        
        # 设置窗口大小和位置
        self.window.geometry(f"{window_width}x{window_height}+{int(screen_width*0.1)}+{int(screen_height*0.1)}")
        self.window.resizable(True, True)  # 允许调整大小，但内部布局保持比例

        # 设置主窗口grid权重（左右8-2分配）
        self.window.grid_columnconfigure(0, weight=8)  # 左侧占8
        self.window.grid_columnconfigure(1, weight=2)  # 右侧占2
        self.window.grid_rowconfigure(0, weight=1)

        # 设置左侧Frame的grid权重和最小大小
        self.left_frame.grid_rowconfigure(0, weight=6)  # 视频区域占6
        self.left_frame.grid_rowconfigure(1, weight=1)  # 车流量标签占1
        self.left_frame.grid_rowconfigure(2, weight=3)  # 检测信息占3
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.configure(width=window_width * 0.8, height=window_height)

        # 设置右侧Frame的grid权重和最小大小
        self.right_frame.grid_rowconfigure(0, weight=6)  # 配时信息占6
        self.right_frame.grid_rowconfigure(1, weight=4)  # 按钮区域占4
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.configure(width=window_width * 0.2, height=window_height)

    def init_info(self):
        self.vehicle_tracks = {}
        self.horizontal_count = 0
        self.vertical_count = 0
        self.signed_id = {}

    def select_video(self):
        """选择视频文件并显示预览"""
        self.video_src = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        
        if self.video_src:
            # 重置检测区域
            from vehicle_statistics import set_detection_zone
            set_detection_zone(None)  # 清空检测区域
            self.points = []  # 清空绘制点
            self.temp_points = []
            self.drawing = False
            self.draw_zone_button.config(text="绘制检测区域")
            
            # 加载视频预览
            self.load_video_preview()
            # 启用绘制按钮，禁用开始检测按钮
            self.draw_zone_button.config(state=tk.NORMAL)
            self.start_button.config(state=tk.DISABLED)

    def load_video_preview(self):
        """加载并显示视频预览"""
        try:
            cap = cv2.VideoCapture(self.video_src)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
                
            ret, frame = cap.read()
            if not ret:
                raise ValueError("无法读取视频帧")
                
            # 保存原始尺寸
            self.original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 调整预览图像大小
            frame = cv2.resize(frame, (self.display_width, self.display_height))
            
            # 转换颜色从BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.preview_photo = ImageTk.PhotoImage(image)
            
            # 更新显示
            self.label.config(image=self.preview_photo)
            self.label.image = self.preview_photo
            
            # 更新右侧信息
            self.right_label.config(
                text="红绿灯配时建议\n\n"
                     "请先绘制检测区域\n"
                     "然后点击开始检测\n\n"
                     "配时说明:\n"
                     "- 基于实时车流量\n"
                     "- 动态调整信号时长\n"
                     "- 保证最小绿灯时间"
            )
            
            # 保存预览帧用于绘制
            self.preview_frame = frame
            
            cap.release()
            self.video_loaded = True
            
        except Exception as e:
            print(f"加载预览失败: {e}")
            self.right_label.config(text=f"加载失败:\n{str(e)}")
            
    def start_detection(self):
        """开始视频检测"""
        # 检查是否已绘制检测区域
        from vehicle_statistics import _detection_zone
        if not _detection_zone or len(_detection_zone) < 3:
            messagebox.showwarning(
                "警告",
                "请先绘制检测区域！\n\n"
                "1. 点击'绘制检测区域'按钮\n"
                "2. 使用左键点击添加区域顶点\n"
                "3. 右键点击完成绘制"
            )
            return
        
        if self.video_src and self.video_loaded:
            try:
                self.cap = cv2.VideoCapture(self.video_src)
                if not self.cap.isOpened():
                    raise ValueError("无法打开视频文件")
                    
                # 获取初始配时方案并显示
                timing = car_flow_control(0, 0)  # 传入0表示还没有车流量数据
                self.right_label.config(
                    text=(
                        "红绿灯配时建议\n\n"
                        "初始配时方案:\n"
                        f"横向绿灯: {timing['horizontal_green']}秒\n"
                        f"纵向绿灯: {timing['vertical_green']}秒\n"
                        f"黄灯时间: {timing['yellow']}秒\n\n"
                        "配时说明:\n"
                        "- 基于最近3次统计数据\n"
                        "- 最小绿灯时间: 30秒\n"
                        "- 最大绿灯时间: 90秒\n"
                        "- 根据车流比动态调整\n\n"
                        "开始统计车流量..."
                    )
                )
                
                # 使用多线程处理视频帧
                self.thread = threading.Thread(target=self.update_frame)
                self.thread.daemon = True
                self.thread.start()
                
                # 禁用开始检测按钮和绘制区域按钮
                self.start_button.config(state=tk.DISABLED)
                self.draw_zone_button.config(state=tk.DISABLED)
                
                self.init_info()
                
            except Exception as e:
                print(f"错误: {e}")
                # 发生错误时重新启用按钮
                self.start_button.config(state=tk.NORMAL)
                self.draw_zone_button.config(state=tk.NORMAL)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            # 添加跳帧处理
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:  # 每 frame_skip 帧处理一次
                ret = self.cap.grab()  # 只获取帧，不解码
                if not ret:
                    self.cap.release()
                    self.start_button.config(state=tk.NORMAL)
                    self.draw_zone_button.config(state=tk.NORMAL)
                    return
                self.window.after(1, self.update_frame)
                return

            ret, frame = self.cap.read()
            if not ret:
                print("视频播放结束")
                self.cap.release()
                self.start_button.config(state=tk.NORMAL)
                self.draw_zone_button.config(state=tk.NORMAL)
                return

            # 调用处理每一帧的函数
            self.prev_time, frame, photo, vertical, horizontal, self.detect_list = process_frame(
                self.det,
                self.prev_time,
                self.cap,
                self.detect_list
            )

            # 更新车流量统计
            current_time = time.time()
            
            # 如果还没开始计数，且检测到车流，则开始计数
            if not self.is_counting and (vertical > 0 or horizontal > 0):
                self.count_start_time = current_time
                self.is_counting = True
                self.total_horizontal = 0
                self.total_vertical = 0
                print("开始计数3分钟车流量")
                
            # 如果正在计数
            if self.is_counting:
                self.total_horizontal += horizontal
                self.total_vertical += vertical
                
                # 计算剩余时间
                elapsed_time = current_time - self.count_start_time
                remaining_time = max(0, self.count_duration - elapsed_time)
                
                # 更新显示
                self.flow_label.config(
                    text=(f"车流统计(剩余{int(remaining_time)}秒): "
                          f"横向车流: {self.total_horizontal}, "
                          f"纵向车流: {self.total_vertical}, "
                          f"车流比: {self.total_horizontal/self.total_vertical if self.total_vertical else 0:.2f}")
                )
                
                # 如果计时结束
                if remaining_time <= 0:
                    self.is_counting = False
                    print(f"\n3分钟车流量统计结果:")
                    print(f"横向车流: {self.total_horizontal}")
                    print(f"纵向车流: {self.total_vertical}")
                    # 获取配时方案
                    timing = car_flow_control(self.total_horizontal, self.total_vertical)
                    
                    # 更新右侧信息显示
                    self.right_label.config(
                        text=(
                            "红绿灯配时建议\n\n"
                            f"本次3分钟车流量:\n"
                            f"横向车流: {self.total_horizontal} (每小时 {timing['h_avg']:.1f})\n"
                            f"纵向车流: {self.total_vertical} (每小时 {timing['v_avg']:.1f})\n\n"
                            f"Webster配时方案:\n"
                            f"周期长度: {timing['cycle_length']}秒\n"
                            f"横向绿灯: {timing['horizontal_green']}秒\n"
                            f"纵向绿灯: {timing['vertical_green']}秒\n"
                            f"黄灯时间: {timing['yellow']}秒\n"
                            f"全红时间: {timing['all_red']}秒\n\n"
                            f"配时计算依据:\n"
                            f"- 总流率比: {timing['Y']:.3f}\n"
                            f"- 理论最优周期: {timing['optimal_cycle']}秒\n"
                            f"- 饱和流率: 1800辆/小时\n"
                            f"- 周期损失时间: {controller.cycle_loss_time}秒\n"
                            f"- 最小绿灯时间: {controller.min_green_time}秒\n"
                            f"- 最大绿灯时间: {controller.max_green_time}秒"
                        )
                    )
                    
                    # 重置计数器，准备下一轮统计
                    self.total_horizontal = 0
                    self.total_vertical = 0

            # 更新检测信息显示
            if hasattr(self.det, 'carTracker'):
                # 清空文本框
                self.detection_text.delete(1.0, tk.END)
                # 获取当前时间戳
                current_time = time.strftime("%H:%M:%S", time.localtime())
                # 添加检测信息
                detection_info = f"[{current_time}] 当前跟踪的车辆:\n"
                for track_id, count in self.det.carTracker.items():
                    detection_info += f"ID-{track_id} "
                self.detection_text.insert(tk.END, detection_info + "\n")
                # 保持滚动到最新内容
                self.detection_text.see(tk.END)

            # 调整图像大小以适应视频显示区域
            frame_height, frame_width = frame.shape[:2]
            video_frame_width = self.frame_video.winfo_width() - 10  # 减去边框宽度
            video_frame_height = self.frame_video.winfo_height() - 10  # 减去边框宽度
            
            # 计算缩放比例
            scale = min(video_frame_width/frame_width, video_frame_height/frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            # 缩放图像
            frame = cv2.resize(frame, (new_width, new_height))
            
            # 创建黑色背景
            background = np.zeros((video_frame_height, video_frame_width, 3), dtype=np.uint8)
            
            # 计算图像在背景中的位置（居中）
            x_offset = (video_frame_width - new_width) // 2
            y_offset = (video_frame_height - new_height) // 2
            
            # 将图像放在背景中央
            background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
            
            # 转换图像格式
            image = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image=image)
            
            # 更新显示
            self.label.config(image=photo)
            self.label.image = photo

            self.window.after(30, self.update_frame)

    def start_drawing(self):
        """开始绘制检测区域"""
        self.drawing = True
        self.points = []
        self.temp_points = []
        # 显示提示框
        messagebox.showinfo(
            "绘制说明",
            "请按以下步骤绘制检测区域：\n\n"
            "1. 左键点击添加区域顶点\n"
            "2. 移动鼠标预览区域形状\n"
            "3. 右键点击完成绘制\n\n"
            "注意：至少需要3个点才能形成有效区域"
        )
        self.draw_zone_button.config(text="右键完成绘制")
        
    def get_original_coordinates(self, display_x, display_y):
        """将显示坐标转换为原始图像坐标"""
        if not self.original_width or not self.original_height:
            print("警告: 原始图像尺寸未设置")
            return display_x, display_y
            
        # 获取实际显示的图像尺寸
        label_width = self.label.winfo_width()
        label_height = self.label.winfo_height()
        
        # 计算图像在Label中的实际位置（考虑填充和边框）
        border_width = 5  # Label的边框宽度
        actual_width = label_width - 2 * border_width
        actual_height = label_height - 2 * border_width
        
        # 调整点击坐标（减去边框宽度）
        display_x = display_x - border_width
        display_y = display_y - border_width
        
        # 计算缩放比例
        scale_x = self.original_width / actual_width
        scale_y = self.original_height / actual_height
        
        # 转换坐标
        original_x = int(display_x * scale_x)
        original_y = int(display_y * scale_y)
        
        # 确保坐标在有效范围内
        original_x = max(0, min(original_x, self.original_width - 1))
        original_y = max(0, min(original_y, self.original_height - 1))
        
        print(f"转换坐标: 显示({display_x}, {display_y}) -> 原始({original_x}, {original_y})")
        return original_x, original_y
            
    def on_click(self, event):
        """处理鼠标左键点击"""
        if self.drawing:
            # 转换坐标
            x, y = self.get_original_coordinates(event.x, event.y)
            self.points.append((x, y))
            self.draw_zone()
            
    def on_move(self, event):
        """处理鼠标移动"""
        if self.drawing and self.points:
            # 转换坐标
            x, y = self.get_original_coordinates(event.x, event.y)
            self.temp_points = self.points + [(x, y)]
            self.draw_zone()
            
    def finish_drawing(self, event):
        """完成绘制"""
        if self.drawing:
            if len(self.points) >= 3:
                self.drawing = False
                self.draw_zone_button.config(text="重新绘制")
                # 添加调试信息
                print(f"设置检测区域: {self.points}")
                # 保存检测区域到 vehicle_statistics
                from vehicle_statistics import set_detection_zone
                set_detection_zone(self.points)
                # 启用开始检测按钮
                self.start_button.config(state=tk.NORMAL)
                # 显示完成提示
                messagebox.showinfo("完成", "检测区域绘制完成！")
            else:
                messagebox.showwarning("警告", "请至少添加3个点才能形成有效的检测区域！")
            
    def draw_zone(self):
        """绘制检测区域"""
        if not hasattr(self, 'preview_frame'):
            return
            
        frame = self.preview_frame.copy()
        points = self.temp_points if self.temp_points else self.points
        
        # 获取实际显示的图像尺寸
        label_width = self.label.winfo_width()
        label_height = self.label.winfo_height()
        border_width = 5
        actual_width = label_width - 2 * border_width
        actual_height = label_height - 2 * border_width
        
        # 将原始坐标转换为显示坐标进行绘制
        display_points = []
        for x, y in points:
            # 计算显示坐标
            display_x = int(x * actual_width / self.original_width) + border_width
            display_y = int(y * actual_height / self.original_height) + border_width
            display_points.append((display_x, display_y))
        
        # 调整frame大小以匹配显示尺寸
        frame = cv2.resize(frame, (actual_width, actual_height))
        
        # 绘制已确定的点
        for point in display_points:
            cv2.circle(frame, (point[0] - border_width, point[1] - border_width), 5, (0, 255, 0), -1)
            
        # 绘制连线
        if len(display_points) > 1:
            # 调整点坐标以匹配实际绘制区域
            draw_points = [(p[0] - border_width, p[1] - border_width) for p in display_points]
            pts = np.array(draw_points, np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
        # 更新显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        self.label.config(image=photo)
        self.label.image = photo
        
    def start(self):
        # 运行Tkinter事件循环
        self.window.mainloop()


if __name__ == '__main__':
    app = VideoGUI()
    app.start()
