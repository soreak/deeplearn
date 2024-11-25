from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from models.MobileNetv3 import mobilenetv3


def set_seed(seed=42):
    # 设置随机种子以确保实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 对于使用多GPU的情况
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_seed(2024)  # 初始化随机种子

    # 确定使用的设备是GPU还是CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 加载模型，并将其移动到指定的设备
    model = mobilenetv3(mode='small').to(device)
    # 替换分类器以适应目标数据集的类别数
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    # 加载训练好的模型权重
    state_dict = torch.load('models/best_mobilenet_model_f1_72.pt', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 定义数据预处理步骤
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像尺寸以匹配模型输入
        transforms.ToTensor(),  # 转换图像为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化处理
    ])

    # 加载并处理数据集
    data_directory = './datasets/fer2013_images'
    dataset = datasets.ImageFolder(root=data_directory, transform=transform)

    # 分割数据集为训练集和验证集
    val_size = int(0.2 * len(dataset))  # 验证集大小
    train_size = len(dataset) - val_size  # 训练集大小
    _, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建DataLoader来批量加载数据
    batch_size = 8
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # 定义类别名称
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # 用于收集预测结果和真实标签的列表
    y_true = []  # 真实标签
    y_pred = []  # 预测标签
    y_scores = []  # 模型输出的分数（用于ROC、PRC等计算）

    with torch.no_grad():  # 不计算梯度，以加速和减少内存消耗
        for images, labels in tqdm(val_loader, desc="Evaluating", leave=True):  # 进度条显示
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 进行预测
            _, preds = torch.max(outputs, 1)  # 获取最大概率的预测结果
            scores = torch.softmax(outputs, dim=1)  # 计算softmax概率分布

            y_true.extend(labels.cpu().numpy())  # 收集真实标签
            y_pred.extend(preds.cpu().numpy())  # 收集预测标签
            y_scores.extend(scores.cpu().detach().numpy())  # 收集预测分数

    y_scores = np.array(y_scores)  # 将预测分数转换为NumPy数组，便于后续处理

    # 绘制并保存混淆矩阵
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # 使用真实标签和预测标签计算混淆矩阵，并进行归一化
    plt.figure(figsize=(10, 8))  # 设置图像大小
    # 使用Seaborn的heatmap函数绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')  # 设置标题
    plt.ylabel('True Label')  # 设置y轴标签
    plt.xlabel('Predicted Label')  # 设置x轴标签
    plt.savefig('./runs/confusion_matrix.png')  # 保存混淆矩阵的图像
    plt.show()  # 显示混淆矩阵的图像

    # 计算并绘制所有类别的PR曲线和整体AP
    y_true_binary = label_binarize(y_true, classes=np.arange(len(class_names)))  # 将真实标签二值化

    precision = dict()  # 初始化精确度字典
    recall = dict()  # 初始化召回率字典
    average_precision = dict()  # 初始化平均精确度字典

    for i in range(len(class_names)):  # 对于每个类别
        # 计算每个类别的精确度和召回率
        precision[i], recall[i], _ = precision_recall_curve(y_true_binary[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true_binary[:, i], y_scores[:, i])  # 计算平均精确度

    # 计算微平均PR曲线和AP
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_binary.ravel(), np.array(y_scores).ravel())
    average_precision["micro"] = average_precision_score(y_true_binary, y_scores, average="micro")  # 计算微平均精确度

    # 绘制PR曲线
    plt.figure(figsize=(8, 8))  # 设置图像大小
    plt.plot(recall['micro'], precision['micro'],
             label=f'Micro-average PR curve (area = {average_precision["micro"]:0.2f})', linestyle=':', linewidth=4)

    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])  # 定义颜色循环
    for i, color in zip(range(len(class_names)), colors):  # 为每个类别绘制PR曲线
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR for class {class_names[i]} (area = {average_precision[i]:0.2f})')

    plt.xlabel('Recall')  # 设置x轴标签
    plt.ylabel('Precision')  # 设置y轴标签
    plt.title('Precision-Recall Curve')  # 设置标题
    plt.legend(loc='best')  # 显示图例
    plt.savefig('./runs/precision_recall_curve.png')  # 保存PR曲线图像
    plt.show()  # 显示PR曲线图像

    # 为每个类别计算F1 Score
    f1_scores = dict()  # 初始化存储每个类别F1分数的字典
    best_f1_scores = dict()  # 初始化存储每个类别最佳F1分数的字典
    best_thresholds = dict()  # 初始化存储每个类别对应最佳阈值的字典

    for i in range(len(class_names)):  # 遍历所有类别
        # 对不同的阈值计算F1分数
        f1_scores[i] = [f1_score(y_true_binary[:, i], y_scores[:, i] > threshold) for threshold in
                        np.linspace(0, 1, 100)]
        best_idx = np.argmax(f1_scores[i])  # 找到最佳F1分数的索引
        best_f1_scores[i] = f1_scores[i][best_idx]  # 获取最佳F1分数
        best_thresholds[i] = np.linspace(0, 1, 100)[best_idx]  # 获取对应的最佳阈值

    # 计算微平均F1分数
    thresholds = np.linspace(0, 1, 100)  # 定义阈值范围
    micro_f1_scores = []  # 初始化微平均F1分数列表

    for threshold in thresholds:  # 遍历所有阈值
        y_pred_binary = y_scores > threshold  # 应用阈值，生成二值预测
        micro_f1 = f1_score(y_true_binary, y_pred_binary, average='micro')  # 计算微平均F1分数
        micro_f1_scores.append(micro_f1)  # 添加到列表

    best_micro_f1 = np.max(micro_f1_scores)  # 找到最佳微平均F1分数
    best_threshold = thresholds[np.argmax(micro_f1_scores)]  # 找到对应的最佳阈值

    # 绘制F1分数曲线
    plt.figure(figsize=(10, 6))  # 设置图像大小
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'maroon', 'darkgreen'])  # 设置颜色循环
    for i, color in zip(range(len(class_names)), colors):  # 为每个类别绘制F1分数曲线
        plt.plot(np.linspace(0, 1, 100), f1_scores[i], color=color, lw=2, label=f'Class {class_names[i]}')

    # 绘制微平均F1分数曲线
    plt.plot(thresholds, micro_f1_scores, color='black', lw=2, linestyle='--',
             label=f'Overall Micro-average (best={best_micro_f1:.2f} at threshold={best_threshold:.2f})')

    plt.xlabel('Threshold')  # 设置x轴标签
    plt.ylabel('F1 Score')  # 设置y轴标签
    plt.title('F1 Score by Class and Overall Micro-average')  # 设置标题
    plt.legend(loc='lower left')  # 显示图例
    plt.grid(True)  # 显示网格

    plt.savefig('./runs/f1_score_curve.png')  # 保存F1分数曲线图
    plt.show()  # 显示F1分数曲线图

    # 计算模型的准确率
    correct_predictions = np.sum(np.array(y_true) == np.array(y_pred))  # 计算正确预测的数量
    total_predictions = len(y_true)  # 总预测数量
    accuracy = correct_predictions / total_predictions  # 计算准确率

    print(f"Accuracy: {accuracy * 100:.2f}%")  # 打印准确率
