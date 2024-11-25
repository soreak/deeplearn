import os.path

import cv2  # 导入OpenCV库，用于图像处理
import torch  # 导入PyTorch库，用于深度学习模型
import numpy as np  # 导入NumPy库，用于数学运算
from torch import nn  # 从torch中导入nn模块，包含构建神经网络的类和函数
from torchvision import transforms  # 导入transforms，用于图像预处理
from QtFusion.models import Detector  # 从QtFusion.models导入Detector类，用于检测任务
from datasets.fer2013.label_name import Chinese_name  # 从datasets.fer2013.label_name导入Chinese_name，包含表情类别的中文名称
from models.MobileNetv3 import mobilenetv3  # 从models.MobileNetv3导入mobilenetv3模型
from models.XceptionModel import mini_XCEPTION  # 从models.XceptionModel导入mini_XCEPTION模型

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 判断是否支持CUDA，以决定使用GPU还是CPU

ini_params = {
    'device': device,  # 设备类型，这里根据CUDA支持情况自动选择GPU或CPU
    'conf': 0.25,  # 定义物体检测的置信度阈值
    'iou': 0.5,  # 定义用于非极大值抑制的交并比(IOU)阈值
}


def count_classes(det_info, class_names):
    # 初始化一个字典，键为类别名称，值为计数（初始为0）
    count_dict = {name: 0 for name in class_names}
    for info in det_info:  # 遍历每个检测到的信息
        class_name = info['class_name']  # 获取检测到的类别名称
        if class_name in count_dict:  # 如果该类别在字典中
            count_dict[class_name] += 1  # 对应类别的计数加1

    # 将字典转换成列表，列表元素的顺序与class_names一致
    count_list = [count_dict[name] for name in class_names]
    return count_list  # 返回类别计数的列表


def non_max_suppression(boxes, scores, iou_threshold):
    if scores.size == 0:  # 如果没有得分，则直接返回空列表
        return []

    # 将边界框坐标从[x, y, width, height]格式转换为[x1, y1, x2, y2]格式
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算每个边界框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照得分从高到低排序边界框的索引
    order = scores.argsort().flip(dims=[0])

    keep = []  # 初始化保留列表
    while order.numel() > 0:  # 当还有边界框时
        i = order[0]  # 取出得分最高的边界框索引
        keep.append(i)  # 将该边界框索引添加到保留列表中

        # 计算当前边界框与其余边界框的交集坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算交集区域的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 计算交集区域的面积
        inter = w * h
        # 计算交并比(IOU)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到IOU小于阈值的边界框，并更新order以进行下一次迭代
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep  # 返回保留边界框的索引列表


class EmotionDetector(Detector):
    def __init__(self, params=None):
        super().__init__(params)  # 调用父类的构造函数
        self.face_model = None  # 初始化面部检测模型变量
        self.emotion_model = None  # 初始化表情识别模型变量
        self.emotion_model_type = None  # 初始化表情模型类型变量
        # 定义图像预处理流程
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # 将tensor或ndarray转换为PIL Image
            transforms.Resize((48, 48)),  # 重新调整图像大小到48x48
            transforms.ToTensor(),  # 将PIL Image或ndarray转换为tensor，并归一化到[0,1]
            transforms.Normalize((0.5,), (0.5,))  # 归一化，使像素值范围为[-1, 1]
        ])
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # 定义表情类别
        self.names = list(Chinese_name.values())  # 从Chinese_name字典中提取所有类别的中文名称
        self.emotion_conf = None  # 初始化表情置信度变量
        # 定义默认参数，包括置信度阈值、IOU阈值和计算设备
        self.params = params if params else {
            'conf': 0.5,
            'iou': 0.4,
            'device': "cuda:0" if torch.cuda.is_available() else "cpu"
        }

    def load_model(self, model_path):
        pass  # 该函数留空，具体实现在子类中

    def load_models(self, face_model_path, emotion_model_path):
        # 加载面部检测模型
        model_file, pretrained_weights = face_model_path
        self.face_model = cv2.dnn.readNetFromCaffe(model_file, pretrained_weights)  # 使用OpenCV加载Caffe模型

        # 将表情类别转换为中文名称
        self.names = [Chinese_name[v] if v in Chinese_name else v for v in self.emotions]

        # 根据模型路径判断是使用哪种表情识别模型
        if "xception" in os.path.basename(emotion_model_path):
            # 如果是Xception模型
            self.emotion_model = mini_XCEPTION(input_channels=1, num_classes=7)  # 实例化Xception模型
            self.emotion_model_type = "xception"
        else:
            # 如果是MobileNet模型
            self.emotion_model = mobilenetv3(mode='small').to(self.params['device'])  # 实例化MobileNet模型
            self.emotion_model.classifier[1] = nn.Linear(self.emotion_model.classifier[1].in_features,
                                                         7)  # 调整分类器以匹配表情类别数量
            self.emotion_model_type = "mobilenet"

        # 加载模型权重
        state_dict = torch.load(emotion_model_path, map_location=self.params['device'])  # 加载模型状态字典
        self.emotion_model.load_state_dict(state_dict)  # 将状态字典应用到模型
        self.emotion_model.to(self.params['device'])  # 将模型移动到指定设备
        self.emotion_model.eval()  # 将模型设置为评估模式

        # 预热模型
        with torch.no_grad():  # 在此模式下，不计算梯度，节省计算资源
            if self.emotion_model_type == "xception":
                dummy_input = torch.rand(1, 1, 48, 48).to(self.params['device'])  # 为Xception模型创建虚拟输入
            else:
                dummy_input = torch.rand(1, 3, 224, 224).to(self.params['device'])  # 为MobileNet模型创建虚拟输入

            for _ in range(4):  # 运行模型4次以预热
                _ = self.emotion_model(dummy_input)

    def preprocess(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式
        return img_rgb  # 返回RGB格式的图像

    def predict(self, img):
        self.emotion_model.to(self.params['device'])  # 确保模型在设置的计算设备上运行

        h, w = img.shape[:2]  # 获取图像的高度和宽度
        # 将图像转换为网络需要的尺寸和格式
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_model.setInput(blob)  # 将处理后的图像作为输入
        detections = self.face_model.forward()  # 进行前向传播，得到检测结果

        results = []  # 初始化结果列表，用于存储最终的检测结果
        configs = []  # 初始化配置列表，用于存储每个检测到的表情的置信度信息
        for i in range(detections.shape[2]):  # 遍历检测到的每个对象
            confidence = detections[0, 0, i, 2]  # 获取当前检测到的对象的置信度
            if confidence > self.params['conf']:  # 如果置信度高于阈值
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # 将边界框坐标缩放回原图尺寸
                (startX, startY, endX, endY) = box.astype("int")  # 获取边界框的整数坐标

                face_region = img[startY:endY, startX:endX]  # 从原图中裁剪出人脸区域

                # 根据不同的模型类型对人脸图像进行预处理
                if self.emotion_model_type == "xception":
                    # 针对Xception模型的预处理步骤
                    face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
                    face_gray = cv2.resize(face_gray, (48, 48))  # 调整尺寸
                    face_gray = np.expand_dims(face_gray, axis=0)  # 增加批处理维度
                    face_gray = np.expand_dims(face_gray, axis=0)  # 增加通道维度
                    face_tensor = torch.tensor(face_gray).float() / 255.0  # 转换为tensor并归一化
                else:
                    # 针对MobileNet模型的预处理步骤
                    face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)  # 转换为灰度图
                    face_gray = cv2.resize(face_gray, (224, 224))  # 调整尺寸
                    face_gray_3channel = np.stack((face_gray,) * 3, axis=-1)  # 将灰度图复制到3个通道
                    face_tensor = torch.tensor(face_gray_3channel).float() / 255.0  # 转换为tensor并归一化
                    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)  # 调整维度顺序并增加批处理维度
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化处理
                    face_tensor = normalize(face_tensor)  # 应用归一化

                face_tensor = face_tensor.to(self.params['device'])  # 将数据移动到指定的计算设备

                with torch.no_grad():  # 不计算梯度，以减少计算资源消耗
                    emotion_preds = self.emotion_model(face_tensor)  # 对人脸区域进行表情预测

                    if self.emotion_model_type == "mobilenet":
                        emotion_preds = torch.softmax(emotion_preds, dim=1)  # 如果是MobileNet模型，需要计算softmax获取置信度
                    emotion_probs = emotion_preds.squeeze().tolist()  # 将预测结果转换为Python列表

                _, predicted_emotion = torch.max(emotion_preds, 1)  # 获取最高置信度的表情索引

                emotion_label = self.names[predicted_emotion]  # 获取表情的中文名称
                # 更新结果列表
                results.append({
                    "class_name": emotion_label,
                    "bbox": [startX, startY, endX, endY],  # 边界框坐标
                    "score": confidence,  # 人脸检测置信度
                    "emotion_confidence": emotion_probs[predicted_emotion],  # 表情置信度
                    "class_id": predicted_emotion  # 表情类别索引
                })

                # 更新配置列表，记录每个表情的置信度
                face_configs = {self.names[i]: prob for i, prob in enumerate(emotion_probs)}
                configs.append(face_configs)

        self.emotion_conf = configs  # 更新类变量，存储所有检测到的表情置信度
        return results, configs  # 返回检测结果和配置信息

    def postprocess(self, preds):
        # 提取预测结果中的边界框和置信度
        boxes = [torch.tensor(pred['bbox']) for pred in preds]  # 边界框
        scores = [pred['score'] for pred in preds]  # 人脸检测的置信度
        boxes = torch.stack(boxes)  # 将边界框列表转换为张量
        scores = torch.tensor(scores)  # 将置信度列表转换为张量

        # 应用非极大值抑制来去除重叠的边界框
        keep = non_max_suppression(boxes, scores, self.params['iou'])
        # 保留非极大值抑制后的结果
        nms_preds = [preds[i] for i in keep if scores[i] >= self.params['conf']]
        # 更新emotion_conf列表以反映非极大值抑制后的结果
        self.emotion_conf = [self.emotion_conf[i] for i in keep if scores[i] >= self.params['conf']]

        return nms_preds  # 返回非极大值抑制后的结果

    def set_param(self, params):
        self.params.update(params)  # 更新检测器参数


# 当直接运行这个文件时执行以下代码
if __name__ == '__main__':
    detector = EmotionDetector()  # 创建表情检测器实例
    # 加载面部检测模型和表情识别模型
    detector.load_models(("./models/deploy.prototxt", "./models/res10_300x300_ssd_iter_140000.caffemodel"),
                         "./models/best_mobilenet_model_f1_72.pt")
    img = cv2.imread("test_media/surprised1.png")  # 读取测试图像
    processed_img = detector.preprocess(img)  # 预处理图像
    predictions, configs = detector.predict(processed_img)  # 进行预测
    final_results = detector.postprocess(predictions)  # 应用非极大值抑制进行后处理

    # 打印最终检测结果和每个检测到的表情的配置信息
    print(final_results)
    print(configs)
