# object_detection/plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import warnings
import os
from pathlib import Path

from object_detection.external_libraries.review_object_detection_metrics.src.evaluators.pascal_voc_evaluator import plot_precision_recall_curve, plot_precision_recall_curves
from object_detection.external_libraries.review_object_detection_metrics.src.utils import converter
from object_detection.external_libraries.review_object_detection_metrics.src.bounding_box import BoundingBox
from object_detection.external_libraries.review_object_detection_metrics.src.utils.enumerators import BBType, BBFormat, CoordinatesType


def box_iou(box1, box2, eps=1e-7):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
        eps

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        is_norm = 'norm' if normalize else ''
        fig.savefig(Path(save_dir) / f'confusion_matrix_{is_norm}.png', dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))

def plot_bb_per_classes(dict_bbs_per_class,
                        output_dir,
                        horizontally=True,
                        rotation=0,
                        show=False,
                        extra_title='',
                        box_type=''):
    plt.close()
    if horizontally:
        ypos = np.arange(len(dict_bbs_per_class.keys()))
        plt.barh(ypos, dict_bbs_per_class.values(), align='edge')
        plt.yticks(ypos, dict_bbs_per_class.keys(), rotation=rotation)
        plt.xlabel('amount of bounding boxes')
        plt.ylabel('classes')
    else:
        plt.bar(dict_bbs_per_class.keys(), dict_bbs_per_class.values())
        plt.xlabel('classes')
        plt.ylabel('amount of bounding boxes')
    plt.xticks(rotation=rotation)
    title = f'Distribution of bounding boxes per class {extra_title}'
    plt.title(title)
    if show:
        plt.tick_params(axis='x', labelsize=10) # Set the x-axis label size
        plt.show(block=True)
    plt.savefig(os.path.join(output_dir, f"bb_per_class_{box_type}.png"))
    return plt

def plot_bb_distributions(output_dir, groundtruth_bbs, det_boxes, index_to_name):

    # # Leave only the annotations of 'x' class
    # groundtruth_bbs = [bb for bb in groundtruth_bbs if bb.get_class_id() == '16']
    # det_boxes = [bb for bb in det_boxes if bb.get_class_id() == '16']

    dict_gt = BoundingBox.get_amount_bounding_box_all_classes(groundtruth_bbs, reverse=True)
    if index_to_name:
        dict_gt = {index_to_name[class_id]: count for class_id, count in dict_gt.items()}
    plot_bb_per_classes(dict_gt, horizontally=False, rotation=90, show=False, extra_title=' (groundtruths)', output_dir=output_dir, box_type="gts")
    clases_gt = [b.get_class_id() for b in groundtruth_bbs]
    dict_det = BoundingBox.get_amount_bounding_box_all_classes(det_boxes, reverse=True)
    if index_to_name:
        dict_det = {index_to_name[class_id]: count for class_id, count in dict_det.items()}
    plot_bb_per_classes(dict_det, horizontally=False, rotation=90, show=False, extra_title=' (detections)', output_dir=output_dir, box_type="dets")

def plot_confusion_matrix(output_dir, cm, num_classes, class_names=None):
    """
    Compute and plot the confusion matrix based on ground truth and detection bounding boxes.

    Parameters
    ----------
    output_dir : str
        Directory where the confusion matrix plot will be saved.
    groundtruth_bbs : list
        List of BoundingBox objects for ground truth.
    det_boxes : list
        List of BoundingBox objects for detections.
    num_classes : int
        Number of object classes.
    class_names : list, optional
        List of class names (strings) for labeling the confusion matrix. If not provided,
        class labels 0,1,...,num_classes-1 will be used.
    """

    # Prepare class names if not provided.
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Create the output directory if it does not exist.
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save the confusion matrix.
    cm.plot(normalize=True, save_dir=output_dir, names=class_names)
    cm.plot(normalize=False, save_dir=output_dir, names=class_names)

    # Optionally, you could print the raw confusion matrix:
    # cm.print()

def get_groundtruth_and_detections(dir_gts, dir_dets):
    # Get annotations (ground truth and detections)
    ground_truth_img_dir = os.path.join(os.path.dirname(dir_gts), "images")
    det_boxes = converter.text2bb(
         annotations_path=dir_dets,
         bb_type=BBType.DETECTED,
         bb_format=BBFormat.XYX2Y2,
         type_coordinates=CoordinatesType.ABSOLUTE,
         img_dir=ground_truth_img_dir
    )

    # For ground truth, we assume YOLO format (relative coordinates).
    groundtruth_bbs = converter.text2bb(
         annotations_path=dir_gts,
         bb_type=BBType.GROUND_TRUTH,
         bb_format=BBFormat.YOLO,
         type_coordinates=CoordinatesType.RELATIVE,
         img_dir=ground_truth_img_dir
    )

    return groundtruth_bbs, det_boxes

def plot_all(voc_metrics, cm, output_dir, dir_dets, dir_gts, config, is_superclass=False,idx_to_name=True):

    index_to_name = None
    if idx_to_name:
        index_to_name = {str(v): k for k, v in config["name_to_index"].items()}
        voc_metrics["per_class"] = {index_to_name[key]: value for key, value in voc_metrics["per_class"].items()}
    
    plot_precision_recall_curve(voc_metrics.get("per_class"), mAP=voc_metrics.get("mAP"), savePath=output_dir, showGraphic=False)

    os.makedirs(os.path.join(output_dir, "all_classes"), exist_ok=True)
    plot_precision_recall_curves(voc_metrics['per_class'], showInterpolatedPrecision=True, showAP=True, savePath=os.path.join(output_dir, "all_classes"), showGraphic=False)

    groundtruth_bbs, det_boxes = get_groundtruth_and_detections(dir_gts, dir_dets)

    if not is_superclass:
        num_classes = len(config["name_to_index"])
        class_names = list(config["name_to_index"].values())
    else:
        num_classes = len(config["mappings"].keys())
        class_names = [config["name_to_index"][key] for key in config["mappings"].keys()]

    if idx_to_name:
        class_names = [index_to_name[str(class_id)] for class_id in class_names]


    plot_confusion_matrix(output_dir, cm, num_classes, class_names)

    plot_bb_distributions(output_dir, groundtruth_bbs, det_boxes, index_to_name)

