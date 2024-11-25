# 导入所需的库和模块
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
from models.MobileNetv3 import mobilenetv3  # 从models模块导入MobileNetv3


# 定义设置随机数种子的函数，以确保实验的可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，还需要设置
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 定义一个用于监控训练的类
class TrainingMonitor:
    def __init__(self, file_path='training_performance_mobilenet.png'):
        self.train_losses = []  # 存储训练损失
        self.val_losses = []  # 存储验证损失
        self.train_acc = []  # 存储训练准确率
        self.val_acc = []  # 存储验证准确率
        self.file_path = file_path  # 图像保存路径

    # 更新训练和验证的损失与准确率
    def update(self, train_loss, val_loss, train_accuracy, val_accuracy):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_acc.append(train_accuracy)
        self.val_acc.append(val_accuracy)

    # 绘制训练和验证过程中的损失与准确率图，并保存为文件
    def plot(self):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 5))

        # 绘制损失图
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'bo-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制准确率图
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_acc, 'bo-', label='Training Accuracy')
        plt.plot(epochs, self.val_acc, 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()

        plt.savefig(self.file_path)  # 保存图像到文件
        plt.close()  # 关闭图形，避免在Jupyter等环境中显示


if __name__ == '__main__':
    # 设置随机种子以确保实验的可重复性
    set_seed(2035)

    # 根据设备的可用性选择使用CPU或GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 定义用于训练集的图像预处理操作
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪为224x224大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转±10度
        transforms.ColorJitter(brightness=0.2),  # 随机调整亮度
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    # 定义用于测试集的图像预处理操作
    test_transforms = transforms.Compose([
        transforms.Resize(224),  # 调整图像大小至256x256
        # transforms.CenterCrop(224),  # 从中心裁剪出224x224大小的图像
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    # 指定数据集的路径
    data_directory = './datasets/fer2013_images'
    # 加载数据集，以文件夹结构组织的图像数据集
    dataset = datasets.ImageFolder(root=data_directory)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))  # 80%用于训练
    test_size = len(dataset) - train_size  # 剩余20%用于测试
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 应用预处理操作
    train_dataset.dataset.transform = train_transforms
    test_dataset.dataset.transform = test_transforms

    # 定义训练参数
    num_epochs = 100  # 训练轮数
    batch_size = 8  # 每批处理的图像数量
    n_worker = 1  # DataLoader使用的工作线程数量

    # 创建DataLoader以在训练和测试时加载数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_worker,
                              pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_worker,
                             pin_memory=True)

    # 加载MobileNetV3模型，并将其转移到指定的设备（CPU或GPU）
    model = mobilenetv3(mode='small').to(device)

    # 加载预训练的权重
    state_dict = torch.load('./models/mobilenetv3_small_67.4.pth.tar', map_location=device)
    model.load_state_dict(state_dict)

    # 调整模型的分类器部分以适应当前任务（假设有7个类别）
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)

    # 再次将模型转移到指定的设备，以确保所有的模型参数都在同一个设备上
    model.to(device)

    # 定义损失函数为交叉熵损失，这在分类任务中非常常见
    criterion = nn.CrossEntropyLoss()

    # 选择Adam优化器，并设置学习率
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 设置早停策略的参数：容忍度（patience）和最小验证损失（min_val_loss）
    # 如果经过一定数量的epoch后验证损失没有改善，则提前停止训练
    patience = 20
    min_val_loss = np.inf
    patience_counter = 0

    # 初始化训练监控器，用于记录和绘制训练过程中的性能变化
    monitor = TrainingMonitor(file_path='runs/training_performance.png')

    # 开始训练周期，遍历设定的训练轮数
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        # 初始化用于累计的变量：运行损失、正确分类的训练样本数和总训练样本数
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 初始化训练阶段的进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]", leave=False)

        # 遍历训练数据集
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)  # 将图像和标签数据移至指定设备

            optimizer.zero_grad()  # 清空之前的梯度

            outputs = model(images)  # 获取模型的预测输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播求梯度
            optimizer.step()  # 根据梯度更新模型参数

            # 累计损失和正确分类的数量
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_pbar.close()  # 关闭训练阶段的进度条

        # 计算训练准确率
        train_accuracy = 100 * correct_train / total_train

        # 验证模型性能
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # 初始化验证阶段的进度条
        val_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", leave=False)
        with torch.no_grad():  # 关闭梯度计算
            # 遍历验证数据集
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_pbar.close()  # 关闭验证阶段的进度条

        # 计算验证准确率
        val_accuracy = 100 * correct_val / total_val

        # 打印每轮的训练损失、验证损失和验证准确率
        print(f'\nEpoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}, '
              f'Val Loss: {val_loss / len(test_loader)}, Accuracy: {val_accuracy}%')

        # 实现早停机制以避免过拟合
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), 'runs/best_mobilenet_model.pt')  # 保存表现最好的模型
            patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:  # 如果验证损失不再下降，则提前停止训练
        #         print("Early stopping triggered")
                # break

        # 更新训练监控器数据
        monitor.update(running_loss / len(train_loader), val_loss / len(test_loader), train_accuracy, val_accuracy)

    # 训练完成后，绘制训练和验证的性能变化图
    monitor.plot()
