import cv2
import torch
from PIL import Image
import numpy as np
from AIDetector_pytorch import Detector
# 加载 YOLOv5 模型

# 对图片进行检测
def detect_image(image_path, output_path):
    # 读取图片
    image = Image.open(image_path)  # 使用 PIL 读取图片
    image_np = np.array(image)  # 转换为 NumPy 数组
    det = Detector()
    # 使用 YOLOv5 进行检测
    im,bboxes = det.detect(image_np) # 获取检测框信息 (x1, y1, x2, y2, confidence, class_id)

    # 绘制检测框
    tl =  round(0.002 * (image_np.shape[0] + image_np.shape[1]) / 2) + 1  # 计算线条厚度
    for (x1, y1, x2, y2, lbl, conf) in bboxes:
        if lbl in ['car']:  # 如果是车，使用红色
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)  # 否则使用绿色
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image_np, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # 绘制矩形框
        tf = max(tl - 1, 1)  # 字体厚度
        t_size = cv2.getTextSize(lbl, 0, fontScale=tl / 3, thickness=tf)[0]  # 计算文本大小
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 调整文本框的位置
        cv2.rectangle(image_np, c1, c2, color, -1, cv2.LINE_AA)  # 填充背景
        conf = f"{conf:.2f}"
        cv2.putText(image_np, '{}CONF-{}'.format(lbl, conf), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    # 在框上方写上标签
    # 保存结果
    output_image = Image.fromarray(image_np)  # 转换回 PIL 格式
    output_image.save(output_path)  # 保存图片
    print(f"检测结果已保存到: {output_path}")

    # 显示结果
    cv2.imshow("Detection Result", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))  # 使用 OpenCV 显示图片
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主函数
def main():
    # 模型权重路径

    # 图片路径
    image_path = "demo.png"  # 替换为你的图片路径
    output_path = "result.jpg"  # 检测结果保存路径

    # 加载模型

    # 对图片进行检测
    detect_image( image_path, output_path)

if __name__ == "__main__":
    main()