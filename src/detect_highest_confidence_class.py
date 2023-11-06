from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import argparse
import cv2
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import numpy as np

def predict_classes(image_path, model, classes, device, img_size, dynamic_conf_thres, nms_thres, true_class):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape

    x = torch.from_numpy(img.transpose(2, 0, 1))
    x = x.unsqueeze(0).float()  # x = (1, 3, H, W)

    # Apply letterbox resize
    _, _, h, w = x.size()
    ih, iw = (416, 416)
    dim_diff = abs(h - w)
    pad1, pad2 = int(dim_diff // 2), int(dim_diff - dim_diff // 2)
    pad = (pad1, pad2, 0, 0) if w <= h else (0, 0, pad1, pad2)
    x = F.pad(x, pad=pad, mode='constant', value=127.5) / 255.0
    x = F.upsample(x, size=(ih, iw), mode='bilinear')  # x = (1, 3, 416, 416)
    x = x.to(device)

    dynamic_conf_thres = 0.01  # ここを調整してください
    with torch.no_grad():
        detections = model(x)
        detections = non_max_suppression(detections, dynamic_conf_thres, nms_thres)  # 動的に設定したスコアを使用

    detections = detections[0]
    if detections is not None:
        # detections = rescale_boxes(detections[0], img_size, [height, width])
        # for detection in detections:
        #     max_conf_idx = torch.argmax(detection[:, 4])
        #     cls_pred = int(detection[max_conf_idx, -1])
        #     predicted_classes.append(classes[cls_pred])
        detections = rescale_boxes(detections, img_size, [height, width])
        max_conf_idx = torch.argmax(detections[:, 4])  # 最も高い信頼度のインデックスを見つける
        x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[max_conf_idx]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_name = classes[int(cls_pred)]
        print(f"認識物体: {class_name}, 信頼度: {conf * cls_conf:.2f}, {true_class}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return class_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_file", type=str, default='/content/yolo-class/src/data/custom/valid.txt', help="path to valid.txt")
    parser.add_argument("--model_def", type=str, default="/content/yolo-class/src/config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="/content/yolo-class/src/weights/yolov3_ckpt_50.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="/content/yolo-class/src/data/custom/classes.names", help="path to class label file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights_path))
    model.eval()

    classes = load_classes(opt.class_path)

    with open(opt.valid_file, 'r') as file:
        lines = file.readlines()

    true_classes = []  # 正解クラスのリスト
    predicted_classes = []  # 予測クラスのリスト

    for line in lines:
        line = line.strip()  # 改行文字を削除
        image_path = line
        true_class = line.split('_')[0]
        last_slash_index = true_class.rfind("/")  # 最後の"/"の位置を取得
        true_class =true_class[last_slash_index + 1:] 
        predicted_class = predict_classes(image_path, model, classes, device, opt.img_size, 0.8, 0.01, true_class)

        true_classes.append(true_class)
        if predicted_class:
            predicted_classes.append(predicted_class)
        else:
            predicted_classes.append('No Prediction')

    # 混合行列を生成
    confusion = confusion_matrix(true_classes, predicted_classes, labels=classes)

    # 混合行列をファイルに出力
    np.savetxt('/content/confusion_matrix.txt', confusion, fmt='%d', delimiter=' ')

if __name__ == "__main__":
    main()
