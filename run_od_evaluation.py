import argparse
import os
import yaml
import pathlib
from object_detection import metrics, plot_utils, mapping

# Dynamically import detector modules
from object_detection.detectors import yolo_detector #, detr_detector  # (detr_detector might be used later)

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Evaluation Pipeline")
    parser.add_argument("--model", type=str, default="yolo11", help="Model name (e.g., 'yolo11', 'detr')")
    parser.add_argument("--weights", type=str, default="Albatross-v0.3.pt", help="Model weights file")
    parser.add_argument("--test_set", type=str, default="Azimut-Haifa-Dataset-v0.4", help="Test-set version to evaluate")
    parser.add_argument("--img_size", type=int, default=[1088, 1920], help="Input image size for the detector")
    return parser.parse_args()

def load_config(config_path="object_detection/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def select_detector(model_name):
    if model_name.lower() == "yolo11":
        return yolo_detector.YOLODetector(model_name)
    # elif model_name.lower() == "detr":
    #     return detr_detector.DETRDetector(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    args = parse_args()
    config = load_config()
    
    # Choose the right detector
    detector = select_detector(args.model)
    
    # Define paths based on the project structure
    test_images_dir = os.path.join("object_detection", "test_sets", args.test_set, "images")
    test_labels_dir = os.path.join("object_detection", "test_sets", args.test_set, "labels")
    model_weights_dir = os.path.join("object_detection", "models")
    output_dir = os.path.join("object_detection","outputs", args.model, pathlib.Path(args.weights).stem)
    
    # Ensure output subdirectories exist:
    subclass_output = os.path.join(output_dir, "subclass")
    superclass_output = os.path.join(output_dir, "superclass")
    subclass_dir_pred = os.path.join(subclass_output, "predictions")
    os.makedirs(subclass_dir_pred, exist_ok=True)

    # Run detector on test-set images
    # (The detector is expected to return predictions in a universal format: list of dicts or similar)
    predictions, avg_inference_speed = detector.run_inference(test_images_dir, args.img_size, model_weights_dir, args.weights)
    
    # Save the raw predictions (text files) for subclasses
    detector.save_predictions(predictions, subclass_dir_pred)
    # avg_inference_speed = 0.0

    # Compute evaluation metrics and graphs for subclass outputs.
    metrics_results = metrics.compute_detection_metrics(subclass_dir_pred, test_labels_dir, args.img_size, avg_inference_speed=avg_inference_speed)
    metrics.save_metrics_csv(metrics_results, os.path.join(subclass_output, "results.csv"))
    plot_utils.plot_all(metrics_results, subclass_output)
    
    # Map subclass predictions to superclasses using the mapping module.
    superclass_dir_pred = os.path.join(superclass_output, "predictions")
    if not os.path.exists(superclass_dir_pred):
        superclass_predictions  = mapping.map_to_superclasses(subclass_dir_pred, config)
        detector.save_predictions(superclass_predictions, superclass_dir_pred)
    else:
        print(f"{superclass_dir_pred} already exists. Skipping mapping.")
    superclass_labels_dir = test_labels_dir.replace("labels", "superclass_labels")
    if not os.path.exists(superclass_labels_dir):
        superclass_labels  = mapping.map_to_superclasses(test_labels_dir, config)
        detector.save_predictions(superclass_labels, superclass_labels_dir)
    else:
        print(f"{superclass_labels_dir} already exists. Skipping mapping.")
    # Compute and save metrics/plots for superclass predictions.
    superclass_metrics = metrics.compute_detection_metrics(superclass_dir_pred, superclass_labels_dir, args.img_size, avg_inference_speed=None, use_superclasses=True)
    metrics.save_metrics_csv(superclass_metrics, os.path.join(superclass_output, "results.csv"))
    plot_utils.plot_all(superclass_metrics, superclass_output)
    
if __name__ == "__main__":
    main()
