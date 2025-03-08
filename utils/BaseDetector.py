from tracker import update_tracker  # 从tracker模块导入update_tracker函数，用于更新目标跟踪器
import cv2  # 导入OpenCV库，用于图像处理

class baseDet(object):  # 定义一个基础检测类，作为其他检测类的基类

    def __init__(self):  # 初始化函数
        self.img_size = 640  # 设置输入图像的大小，默认值为640
        self.threshold = 0.3  # 设置目标检测的置信度阈值，默认值为0.3
        self.stride = 1  # 设置步幅，默认值为1（步幅是卷积操作中的一个参数）

    def build_config(self):  # 配置初始化函数
        self.carTracker = {}  # 初始化人脸追踪器字典，用于存储每个人脸的追踪信息
        self.carClasses = {}  # 初始化人脸类别字典，用于存储每个追踪到的面部ID的类别
        self.carLocation1 = {}  # 初始化一个字典，用于存储人脸的位置1（可能是第一次检测到的位置）
        self.carLocation2 = {}  # 初始化一个字典，用于存储人脸的位置2（可能是第二次检测到的位置）
        self.frameCounter = 0  # 帧计数器，用于统计当前处理到第几帧
        self.currentCarID = 0  # 当前车辆ID计数器，初始化为0
        self.recorded = []  # 用于存储已记录的信息，可能用于保存已检测到的目标
        self.font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体，用于显示文本（如目标ID）在图像上

    def feedCap(self, im):  # 处理每一帧图像的函数
        retDict = {  # 返回一个字典，包含处理后的信息
            'frame': None,  # 当前帧图像
            'cars': None,  # 检测到的面部
            'list_of_ids': set(),  # 目标ID列表
            'car_bboxes': [],  # 存储面部边界框
            'signal':None
        }
        self.frameCounter += 1  # 帧计数器加1
        # 调用update_tracker函数来更新跟踪器并获取最新结果
        im, cars, car_bboxes,signal,current_ids = update_tracker(self, im)
        # 将更新后的图像、面部和边界框结果存储在字典中
        retDict['current_ids'] = current_ids
        retDict['frame'] = im
        retDict['cars'] = cars
        retDict['car_bboxes'] = car_bboxes
        retDict['signal'] = signal
        return retDict  # 返回包含处理结果的字典

    def init_model(self):  # 初始化模型的函数
        raise EOFError("Undefined model type.")  # 抛出EOFError异常，提示未定义模型类型

    def preprocess(self):  # 预处理函数
        raise EOFError("Undefined model type.")  # 抛出EOFError异常，提示未定义预处理方法

    def detect(self):  # 检测函数
        raise EOFError("Undefined model type.")  # 抛出EOFError异常，提示未定义检测方法
