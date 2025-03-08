from video_gui import VideoGUI  # 导入你封装的GUI界面
from ultralytics import YOLO

def main():
    # 创建 VideoGUI 实例
    gui = VideoGUI()
    # 启动GUI应用
    gui.start()

if __name__ == "__main__":
    main()
