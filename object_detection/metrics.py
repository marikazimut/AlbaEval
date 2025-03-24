# object_detection/metrics.py
import os
import pandas as pd
from collections import defaultdict
import torch
import numpy as np

from object_detection.plot_utils import ConfusionMatrix

from object_detection.external_libraries.review_object_detection_metrics.src.evaluators.coco_evaluator import get_coco_summary
from object_detection.external_libraries.review_object_detection_metrics.src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics
from object_detection.external_libraries.review_object_detection_metrics.src.utils.enumerators import BBType, BBFormat, CoordinatesType, CoordinatesType
from object_detection.external_libraries.review_object_detection_metrics.src.utils import converter

def compute_pascalvoc_metrics(predictions_dir, ground_truth_dir, iou_threshold=0.5):
    """
    Compute object detection metrics using the Pascal VOC evaluator.

    Parameters:
        predictions_dir (str): Directory containing the prediction text files.
        ground_truth_dir (str): Directory containing the ground-truth label files.
        iou_threshold (float): IOU threshold for a detection to be considered a TP.
        method: Method to calculate AP (e.g., EVERY_POINT_INTERPOLATION or ELEVEN_POINT_INTERPOLATION).

    Returns:
        A dictionary containing per-class metrics and mAP.
    """
    # Convert text files into bounding box objects.
    # For predictions, we use absolute coordinates (assuming your detector outputs these).
    ground_truth_img_dir = os.path.join(os.path.dirname(ground_truth_dir), "images")
    det_boxes = converter.text2bb(
         annotations_path=predictions_dir,
         bb_type=BBType.DETECTED,
         bb_format=BBFormat.XYX2Y2,
         type_coordinates=CoordinatesType.ABSOLUTE,
         img_dir=ground_truth_img_dir
    )

    # For ground truth, we assume YOLO format (relative coordinates).
    groundtruth_bbs = converter.text2bb(
         annotations_path=ground_truth_dir,
         bb_type=BBType.GROUND_TRUTH,
         bb_format=BBFormat.YOLO,
         type_coordinates=CoordinatesType.RELATIVE,
         img_dir=ground_truth_img_dir
    )

    # Compute VOC metrics.
    voc_metrics = get_pascalvoc_metrics(
        gt_boxes=groundtruth_bbs,
        det_boxes=det_boxes,
        iou_threshold=iou_threshold,
        generate_table=False  # Change to True if you want a detailed table.
    )
    
    return voc_metrics


def save_voc_metrics_csv(voc_metrics, output_csv_path):
    """
    Save VOC metrics (per class and mAP) to a CSV file.
    """
    import pandas as pd
    # Flatten the per-class results.
    rows = []
    for class_id, metrics_dict in voc_metrics.get("per_class", {}).items():
        row = {"class": class_id}
        row.update(metrics_dict)
        rows.append(row)
    # Append overall mAP.
    overall = {"class": "mAP", "AP": voc_metrics.get("mAP")}
    rows.append(overall)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    

def compute_confusion_matrix(groundtruth_bbs, det_boxes, num_classes):
    # Compute and plot the confusion matrix based on ground truth and detection bounding boxes.
    cm = ConfusionMatrix(nc=num_classes, conf=0.25, iou_thres=0.45)

    # Group bounding boxes by image name so that IoU is computed per image.
    gt_by_image = defaultdict(list)
    det_by_image = defaultdict(list)
    for gt in groundtruth_bbs:
        gt_by_image[gt._image_name].append(gt)
    for det in det_boxes:
        det_by_image[det._image_name].append(det)

    # Process images that have at least one ground truth box.
    for image, gt_list in gt_by_image.items():
        # Convert ground truth boxes into a tensor of shape (M, 5):
        # [class, x1, y1, x2, y2]
        labels = []
        for gt in gt_list:
            # Convert class id to int if necessary.
            labels.append([int(gt._class_id), gt._x, gt._y, gt._x2, gt._y2])
        labels = torch.tensor(labels)

        # Get detections for this image, if any.
        if image in det_by_image:
            det_list = det_by_image[image]
            detections = []
            for det in det_list:
                # Format: [x1, y1, x2, y2, confidence, class]
                detections.append([det._x, det._y, det._x2, det._y2, det._confidence, int(det._class_id)])
            detections = torch.tensor(detections)
        else:
            detections = torch.empty((0, 6))

        # Update the confusion matrix for this image.
        cm.process_batch(detections, labels)

    # Also process images that have detections but no ground truth.
    for image, det_list in det_by_image.items():
        if image not in gt_by_image:
            # No ground truth boxes for this image: we create an empty ground truth tensor.
            labels = torch.empty((0, 5))
            detections = []
            for det in det_list:
                detections.append([det._x, det._y, det._x2, det._y2, det._confidence, int(det._class_id)])
            detections = torch.tensor(detections)
            cm.process_batch(detections, labels)

    return cm

def get_class_names(config, is_superclass):
    """
    Get the class names for a given configuration.
    """
    if not is_superclass:
        num_classes = len(config["name_to_index"])
        class_names = list(config["name_to_index"].values())
    else:
        num_classes = len(config["mappings"].keys())
        class_names = [config["name_to_index"][key] for key in config["mappings"].keys()]
    return num_classes, class_names

def compute_detection_metrics(predictions_dir, ground_truth_dir, img_size, config, use_superclasses=False, avg_inference_speed=None):
    """
    Compute object detection metrics by reading predictions from saved text files.
    
    Parameters:
      predictions_dir: str
          Directory containing the prediction text files.
      ground_truth_dir: str
          Directory where ground truth label files are stored.
      use_superclasses: bool
          Flag to adjust evaluation when predictions are for superclasses.
      avg_inference_speed: float
          Average inference time (in seconds) per image.
    
    Returns:
      A dictionary containing evaluation metrics (e.g., AP, AR) along with the inference speed.
    """

    ground_truth_img_dir = os.path.join(os.path.dirname(ground_truth_dir), "images")
    detected_bbs = converter.text2bb(
            annotations_path=predictions_dir,    # directory containing your prediction .txt files
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.XYX2Y2,                   # telling the converter that tokens are x1,y1,x2,y2
            type_coordinates=CoordinatesType.ABSOLUTE,
            img_dir=ground_truth_img_dir
    )

    groundtruth_bbs = converter.text2bb(
            annotations_path=ground_truth_dir,    # directory containing your prediction .txt files
            bb_type=BBType.GROUND_TRUTH,
            bb_format=BBFormat.YOLO,                   # telling the converter that tokens are x1,y1,x2,y2
            type_coordinates=CoordinatesType.RELATIVE,
            img_dir=ground_truth_img_dir
    )

    # Compute metrics using the external COCO-based evaluator.
    coco_metrics = get_coco_summary(groundtruth_bbs, detected_bbs)

    num_classes, _ = get_class_names(config, use_superclasses)
    # Compute the confusion matrix.
    cm = compute_confusion_matrix(groundtruth_bbs, detected_bbs, num_classes)

    # Compute FN_det and FP_det from the confusion matrix.
    # FN_det: Sum over the "background" row (index num_classes) for all object classes.
    FN_det = np.sum(cm.matrix[cm.nc, :cm.nc])
    # FP_det: Sum over the "background" column (index num_classes) for all predicted classes.
    FP_det = np.sum(cm.matrix[:cm.nc, cm.nc])

    # Add these additional metrics to the results.
    coco_metrics["FN_det"] = int(FN_det)
    coco_metrics["FP_det"] = int(FP_det)
    coco_metrics["FN_det_norm"] = FN_det / len(groundtruth_bbs)
    coco_metrics["FP_det_norm"] = FP_det / len(groundtruth_bbs)
    coco_metrics["gt_num"] = len(groundtruth_bbs)

    # Add the average inference speed (runtime) to the metrics.
    if avg_inference_speed is not None:
        coco_metrics["inference_speed"] = avg_inference_speed

    return coco_metrics, cm

def save_metrics_csv(metrics_results, output_csv_path):
    df = pd.DataFrame([metrics_results])
    df.to_csv(output_csv_path, index=False)
