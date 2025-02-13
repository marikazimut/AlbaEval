# object_detection/metrics.py
import os
import pandas as pd

# Import the COCO evaluator from your external library.
from object_detection.external_libraries.review_object_detection_metrics.src.evaluators.coco_evaluator import get_coco_summary
from object_detection.external_libraries.review_object_detection_metrics.src.bounding_box import BoundingBox, BBType, BBFormat
from object_detection.external_libraries.review_object_detection_metrics.src.utils.enumerators import CoordinatesType
from object_detection.external_libraries.review_object_detection_metrics.src.utils.general_utils import convert_to_absolute_values
from object_detection.external_libraries.review_object_detection_metrics.src.utils import converter


def compute_detection_metrics(predictions_dir, ground_truth_dir, img_size, use_superclasses=False, avg_inference_speed=None):
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

    ground_truth_img_dir = ground_truth_dir.replace("labels", "images")
    detected_bbs = converter.text2bb(
            annotations_path=predictions_dir,    # directory containing your prediction .txt files
            bb_type=BBType.DETECTED,
            bb_format=BBFormat.XYX2Y2,                   # telling the converter that tokens are x1,y1,x2,y2
            type_coordinates=CoordinatesType.ABSOLUTE,
            img_dir=ground_truth_img_dir
    )

    ground_truth_img_dir = os.path.join(os.path.dirname(ground_truth_dir), "images")
    groundtruth_bbs = converter.text2bb(
            annotations_path=ground_truth_dir,    # directory containing your prediction .txt files
            bb_type=BBType.GROUND_TRUTH,
            bb_format=BBFormat.YOLO,                   # telling the converter that tokens are x1,y1,x2,y2
            type_coordinates=CoordinatesType.RELATIVE,
            img_dir=ground_truth_img_dir
    )

    # Compute metrics using the external COCO-based evaluator.
    coco_metrics = get_coco_summary(groundtruth_bbs, detected_bbs)
    
    # Add the average inference speed (runtime) to the metrics.
    if avg_inference_speed is not None:
        coco_metrics["inference_speed"] = avg_inference_speed

    return coco_metrics

def save_metrics_csv(metrics_results, output_csv_path):
    df = pd.DataFrame([metrics_results])
    df.to_csv(output_csv_path, index=False)
