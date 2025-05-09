from __future__ import division

from models import Darknet
from utils.utils import non_max_suppression, non_max_suppression_output, xywh2xyxy, get_batch_statistics, ap_per_class, load_classes
from utils.datasets import ListDataset
from utils.parse_config import parse_data_config

import numpy as np
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    # デフォルトの空メトリクスを設定
    default_metrics = ([], [], [])

    # sample_metrics にデフォルト値を含むリストを初期化
    sample_metrics = [default_metrics] * len(dataloader)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3_mask.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/mask_dataset.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_36.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/mask_dataset.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()

    print("")
    for arg in vars(opt):
        print(str(arg) +":\t\t\t"+ str(getattr(opt, arg)))
    print("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")
    print("")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )
    print("")
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print("")
    print(f"mAP: {AP.mean()}")
