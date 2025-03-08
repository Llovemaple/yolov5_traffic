import argparse  # 用于处理命令行参数
import os  # 用于操作文件和目录
import time  # 用于计算时间

import numpy as np  # 用于数组操作
import matplotlib.pyplot as plt  # 用于绘图
import torch  # PyTorch库
import torch.backends.cudnn as cudnn  # 加速cuDNN后端
import torchvision  # PyTorch的计算机视觉模块

from model import Net  # 导入自定义的神经网络模型

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='data', type=str)  # 数据集目录
parser.add_argument("--no-cuda", action="store_true")  # 是否使用GPU
parser.add_argument("--gpu-id", default=0, type=int)  # GPU ID
parser.add_argument("--lr", default=0.1, type=float)  # 学习率
parser.add_argument("--interval", '-i', default=20, type=int)  # 每20个批次输出一次信息
parser.add_argument('--resume', '-r', action='store_true')  # 是否从检查点恢复训练
args = parser.parse_args()  # 解析命令行参数

# 选择设备（GPU或CPU）
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True  # 如果输入数据的尺寸是固定的，启用CuDNN优化

# 数据加载
root = args.data_dir  # 数据根目录
train_dir = os.path.join(root, "train")  # 训练集目录
test_dir = os.path.join(root, "test")  # 测试集目录
transform_train = torchvision.transforms.Compose([  # 定义训练数据预处理
    torchvision.transforms.RandomCrop((128, 64), padding=4),  # 随机裁剪
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.ToTensor(),  # 转为Tensor
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 图像归一化
])
transform_test = torchvision.transforms.Compose([  # 定义测试数据预处理
    torchvision.transforms.Resize((128, 64)),  # 缩放图像
    torchvision.transforms.ToTensor(),  # 转为Tensor
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 图像归一化
])

# 使用ImageFolder读取训练和测试数据集
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),  # 加载训练数据
    batch_size=64, shuffle=True  # 批量大小为64，打乱数据
)
testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),  # 加载测试数据
    batch_size=64, shuffle=True  # 批量大小为64，打乱数据
)
num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))  # 类别数目

# 网络定义
start_epoch = 0  # 初始化起始epoch
net = Net(num_classes=num_classes)  # 初始化网络模型
if args.resume:  # 如果从检查点恢复训练
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')  # 加载检查点
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    net_dict = checkpoint['net_dict']  # 加载模型参数
    net.load_state_dict(net_dict)  # 加载模型状态
    best_acc = checkpoint['acc']  # 加载最佳准确率
    start_epoch = checkpoint['epoch']  # 恢复训练的epoch
net.to(device)  # 将模型移到指定设备（GPU或CPU）

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)  # 定义SGD优化器
best_acc = 0.  # 最佳准确率初始化为0


# 训练函数
def train(epoch):
    print("\nEpoch : %d" % (epoch + 1))  # 输出当前训练轮次
    net.train()  # 设置网络为训练模式
    training_loss = 0.
    train_loss = 0.
    correct = 0  # 正确预测的数量
    total = 0  # 总样本数
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):  # 遍历训练数据
        # 正向传播
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到设备
        outputs = net(inputs)  # 网络预测
        loss = criterion(outputs, labels)  # 计算损失

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 累计训练信息
        training_loss += loss.item()  # 累计损失
        train_loss += loss.item()  # 累计损失
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()  # 统计正确预测数量
        total += labels.size(0)  # 累计总样本数

        # 打印每interval个batch的训练信息
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total,
                100. * correct / total
            ))
            training_loss = 0.  # 重置损失
            start = time.time()

    return train_loss / len(trainloader), 1. - correct / total  # 返回训练损失和训练错误率


# 测试函数
def test(epoch):
    global best_acc
    net.eval()  # 设置网络为评估模式
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():  # 不计算梯度
        for idx, (inputs, labels) in enumerate(testloader):  # 遍历测试数据
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)  # 计算损失

            test_loss += loss.item()  # 累计损失
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()  # 统计正确预测数量
            total += labels.size(0)  # 累计总样本数

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(testloader), end - start, test_loss / len(testloader), correct, total,
            100. * correct / total
        ))

    # 保存最佳模型
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc  # 更新最佳准确率
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')  # 如果没有检查点目录，创建目录
        torch.save(checkpoint, './checkpoint/ckpt.t7')  # 保存模型

    return test_loss / len(testloader), 1. - correct / total  # 返回测试损失和错误率


# 绘制训练和测试曲线
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")


# 学习率衰减
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1  # 学习率衰减
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


# 主函数
def main():
    for epoch in range(start_epoch, start_epoch + 40):  # 训练40轮
        train_loss, train_err = train(epoch)  # 训练
        test_loss, test_err = test(epoch)  # 测试
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)  # 绘制曲线
        if (epoch + 1) % 20 == 0:
            lr_decay()  # 每20轮衰减学习率
if __name__ == '__main__':
    main()  # 启动主程序
