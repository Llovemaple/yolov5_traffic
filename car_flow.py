import logging
import time

class WebsterPerformance:
    def __init__(self):
        self.cycle_times = []  # 周期时间记录
        self.computation_times = []  # 计算耗时记录
        self.flow_accuracy = []  # 流量统计准确率
        self.phase_adjustments = []  # 相位调整记录
        
    def record_cycle(self, cycle_time, computation_time):
        self.cycle_times.append(cycle_time)
        self.computation_times.append(computation_time)
        
    def record_flow_accuracy(self, actual, measured):
        accuracy = abs(actual - measured) / actual if actual > 0 else 0
        self.flow_accuracy.append(accuracy)
        
    def record_phase_adjustment(self, old_phase, new_phase):
        self.phase_adjustments.append(abs(new_phase - old_phase))
        
    def get_statistics(self):
        return {
            "avg_cycle_time": sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0,
            "avg_computation_time": sum(self.computation_times) / len(self.computation_times) if self.computation_times else 0,
            "avg_flow_accuracy": sum(self.flow_accuracy) / len(self.flow_accuracy) if self.flow_accuracy else 0,
            "avg_phase_adjustment": sum(self.phase_adjustments) / len(self.phase_adjustments) if self.phase_adjustments else 0
        }

class TrafficController:
    def __init__(self):
        # 基础配置
        self.min_green_time = 30  # 最小绿灯时间(秒)
        self.max_green_time = 90  # 最大绿灯时间(秒)
        self.yellow_time = 3      # 黄灯时间(秒)
        self.all_red_time = 2     # 全红时间(秒)
        
        # Webster算法参数
        self.cycle_loss_time = 2 * (self.yellow_time + self.all_red_time)  # 周期损失时间
        self.saturation_flow = 1800  # 饱和流率（辆/小时）
        self.critical_flow_ratio = 0.9  # 关键流率比
        
        # 存储历史车流量数据
        self.horizontal_history = []  # 存储横向车流量历史数据
        self.vertical_history = []    # 存储纵向车流量历史数据
        self.history_size = 3         # 保存最近3次的数据
        
        # 初始配时（默认配时）
        self.default_horizontal_green = 45  # 默认横向绿灯时间
        self.default_vertical_green = 45    # 默认纵向绿灯时间
        
        # 添加性能监控
        self.performance = WebsterPerformance()
        self.debug_mode = True  # 调试模式开关
        
    def calculate_optimal_cycle(self, horizontal_flow, vertical_flow):
        """计算最优周期时间"""
        import time
        start_time = time.time()
        
        # 原有的周期计算逻辑
        Y = (horizontal_flow + vertical_flow) / self.saturation_flow
        if Y >= 1:
            Y = 0.95  # 防止饱和度过高
        
        optimal_cycle = int((1.5 * self.cycle_loss_time + 5) / (1 - Y))
        optimal_cycle = max(min(optimal_cycle, self.max_green_time), self.min_green_time)
        
        # 记录性能数据
        computation_time = time.time() - start_time
        self.performance.record_cycle(optimal_cycle, computation_time)
        
        if self.debug_mode:
            logging.info(f"计算最优周期: {optimal_cycle}秒, 计算耗时: {computation_time:.3f}秒")
            logging.info(f"当前饱和度Y: {Y:.2f}")
        
        return optimal_cycle, Y
        
    def get_timing_plan(self, horizontal_flow, vertical_flow):
        """获取配时方案"""
        import time
        start_time = time.time()
        
        # 记录原始相位时间
        old_horizontal = self.default_horizontal_green
        old_vertical = self.default_vertical_green
        
        optimal_cycle, Y = self.calculate_optimal_cycle(horizontal_flow, vertical_flow)
        
        # 计算绿信比
        total_flow = horizontal_flow + vertical_flow
        if total_flow > 0:
            horizontal_ratio = horizontal_flow / total_flow
            vertical_ratio = vertical_flow / total_flow
        else:
            horizontal_ratio = vertical_ratio = 0.5
        
        # 计算绿灯时间
        effective_green_time = optimal_cycle - self.cycle_loss_time
        horizontal_green = int(effective_green_time * horizontal_ratio)
        vertical_green = int(effective_green_time * vertical_ratio)
        
        # 记录相位调整
        self.performance.record_phase_adjustment(old_horizontal, horizontal_green)
        self.performance.record_phase_adjustment(old_vertical, vertical_green)
        
        timing_plan = {
            'horizontal_green': horizontal_green,
            'vertical_green': vertical_green,
            'yellow': self.yellow_time,
            'all_red': self.all_red_time,
            'cycle_length': optimal_cycle,
            'computation_time': time.time() - start_time
        }
        
        if self.debug_mode:
            logging.info(f"配时方案生成完成: {timing_plan}")
            logging.info(f"性能统计: {self.performance.get_statistics()}")
        
        return timing_plan

    def get_default_timing(self):
        """获取默认配时方案"""
        return {
            'horizontal_green': self.default_horizontal_green,
            'vertical_green': self.default_vertical_green,
            'yellow': self.yellow_time,
            'all_red': self.all_red_time,
            'cycle_length': 0,
            'h_avg': 0,
            'v_avg': 0,
            'Y': 0,
            'optimal_cycle': 0
        }

    def add_flow_data(self, horizontal, vertical):
        """添加新的车流量数据"""
        self.horizontal_history.append(horizontal)
        self.vertical_history.append(vertical)
        
        # 保持最近3次的数据
        if len(self.horizontal_history) > self.history_size:
            self.horizontal_history.pop(0)
        if len(self.vertical_history) > self.history_size:
            self.vertical_history.pop(0)
            
    def get_average_flow(self):
        """计算平均车流量"""
        h_avg = sum(self.horizontal_history) / len(self.horizontal_history) if self.horizontal_history else 0
        v_avg = sum(self.vertical_history) / len(self.vertical_history) if self.vertical_history else 0
        return h_avg, v_avg
        
    def webster_timing(self, horizontal_flow, vertical_flow):
        """
        使用Webster方法计算信号配时
        :param horizontal_flow: 横向车流量（辆/小时）
        :param vertical_flow: 纵向车流量（辆/小时）
        :return: 配时方案
        """
        # 计算各个方向的流率比
        y1 = horizontal_flow / self.saturation_flow  # 横向流率比
        y2 = vertical_flow / self.saturation_flow    # 纵向流率比
        Y = y1 + y2  # 总流率比

        # 检查是否超过饱和度阈值
        if Y >= self.critical_flow_ratio:
            print("警告: 交通流量接近或超过饱和状态")
            Y = self.critical_flow_ratio

        # 计算最优周期长度
        L = self.cycle_loss_time
        Co = (1.5 * L + 5) / (1 - Y)  # Webster最优周期公式
        
        # 限制周期长度在合理范围内
        C = min(max(Co, 60), 120)  # 周期长度限制在60-180秒之间
        
        # 计算有效绿灯时间
        total_effective_green = C - L
        
        # 按流率比分配有效绿灯时间
        if Y > 0:
            h_green = total_effective_green * (y1 / Y)
            v_green = total_effective_green * (y2 / Y)
        else:
            h_green = v_green = total_effective_green / 2

        # 转换为实际绿灯时间（考虑启动损失时间和黄灯利用时间）
        h_green = max(min(int(h_green + 2), self.max_green_time), self.min_green_time)
        v_green = max(min(int(v_green + 2), self.max_green_time), self.min_green_time)

        return {
            'horizontal_green': h_green,
            'vertical_green': v_green,
            'yellow': self.yellow_time,
            'all_red': self.all_red_time,
            'cycle_length': int(C),
            'h_avg': horizontal_flow,
            'v_avg': vertical_flow,
            'Y': Y,
            'optimal_cycle': int(Co)
        }

    def get_signal_timing(self, horizontal_flow, vertical_flow):
        """计算信号配时"""
        # 添加新的流量数据
        self.add_flow_data(horizontal_flow, vertical_flow)
        
        # 获取平均车流量
        h_avg, v_avg = self.get_average_flow()
        
        # 转换为小时流量（从每分钟转换为每小时）
        h_hourly = h_avg * 60
        v_hourly = v_avg * 60
        
        # 使用Webster方法计算配时
        timing = self.webster_timing(h_hourly, v_hourly)
        
        return timing

# 创建全局控制器实例
controller = TrafficController()

def car_flow_control(horizontal, vertical):
    """
    根据车流量控制红绿灯
    :param horizontal: 3分钟横向车流量
    :param vertical: 3分钟纵向车流量
    :return: 信号配时方案
    """
    # 如果没有车流量数据，返回默认配时
    if horizontal == 0 and vertical == 0:
        return controller.get_default_timing()
        
    # 获取信号配时（转换为每分钟平均车流量）
    timing = controller.get_signal_timing(horizontal/3, vertical/3)
    
    # 打印当前配时方案
    print(f"\n当前3分钟车流量:")
    print(f"横向车流: {horizontal} (每小时 {timing['h_avg']:.1f})")
    print(f"纵向车流: {vertical} (每小时 {timing['v_avg']:.1f})")
    print(f"\nWebster配时计算结果:")
    print(f"总流率比(Y): {timing['Y']:.3f}")
    print(f"理论最优周期长度: {timing['optimal_cycle']}秒")
    print(f"实际周期长度: {timing['cycle_length']}秒")
    print(f"横向绿灯时间: {timing['horizontal_green']}秒")
    print(f"纵向绿灯时间: {timing['vertical_green']}秒")
    print(f"黄灯时间: {timing['yellow']}秒")
    print(f"全红时间: {timing['all_red']}秒")
    
    return timing