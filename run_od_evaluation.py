import argparse
import os
import yaml
import pathlib
from object_detection import metrics, plot_utils, mapping
import logging
import shutil

# Dynamically import detector modules
from object_detection.detectors import yolo_detector, yoloe_detector #, detr_detector  # (detr_detector might be used later)

def setup_logger(log_file="pipeline.log"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Evaluation Pipeline")
    parser.add_argument("--model", type=str, default="yoloe-11m", help="Model name (e.g., 'yolo11', 'yolo9', 'yoloe-11m', 'detr')")
    parser.add_argument("--weights", type=str, default="yoloe-11m-seg.pt", help="Model weights file (e.g., 'Albatross-v0.3.pt', 'Albatross-v0.2.pt','Albatross-v0.4.pt', 'Albatross-v0.4-2.pt', 'yoloe-11m-seg.pt')")
    parser.add_argument("--test_set", type=str, default="yoloe-test", help="Test-set version to evaluate (e.g., 'Azimut-Haifa-Dataset-v0.4', 'Albatross-Dataset-v0.4-test')")
    parser.add_argument("--img_size", type=int, default=[1088, 1920], help="Input image size for the detector") 
    return parser.parse_args()

def load_config(config_path="object_detection/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def select_detector(model_name):
    if model_name.lower() in ["yolo11", "yolo9"]:
        return yolo_detector.YOLODetector(model_name)
    elif model_name.lower() == "yoloe-11m":
        return yoloe_detector.YOLOEDetector(model_name)
    # elif model_name.lower() == "detr":
    #     return detr_detector.DETRDetector(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    args = parse_args()
    config = load_config()
    
    # for weights in ["Albatross-v0.3.pt", "Albatross-v0.4.pt"]:
    for weights in ["yoloe-11m-seg.pt"]:

        args.weights = weights

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
        predictions, avg_inference_speed, corrupted_images = detector.run_inference(test_images_dir, args.img_size, model_weights_dir, args.weights)
        logging.info(f"Finished running inference on {test_images_dir}")

        if len(corrupted_images) > 0:
            logging.warning(f"Corrupted images: {corrupted_images}")
            print(f"Removing corrupted images from test-set to corrupted folder")
            corrupted_dir = os.path.join("object_detection", "test_sets", "corrupted", args.test_set)
            os.makedirs(corrupted_dir, exist_ok=True)
            for image in corrupted_images:
                image_extension = os.path.splitext(image)[1]
                label_name = image.replace(image_extension, ".txt")
                shutil.move(os.path.join(test_images_dir, image), os.path.join(corrupted_dir, image))
                shutil.move(os.path.join(test_labels_dir, label_name), os.path.join(corrupted_dir, label_name))
                print(f"Corrupted image removed: {image}")

        detector.save_predictions(predictions, subclass_dir_pred)
        logging.info(f"Saved predictions to {subclass_dir_pred}")
        # avg_inference_speed = 0.0

        # Compute evaluation metrics and graphs for subclass outputs.
        logging.info(f"Computing detection metrics for {subclass_dir_pred}")
        metrics_results, cm = metrics.compute_detection_metrics(subclass_dir_pred, test_labels_dir, args.img_size, config, avg_inference_speed=avg_inference_speed)
        metrics.save_metrics_csv(metrics_results, os.path.join(subclass_output, "results.csv"))
        voc_metrics = metrics.compute_pascalvoc_metrics(subclass_dir_pred, test_labels_dir, iou_threshold=0.5)
        metrics.save_voc_metrics_csv(voc_metrics, os.path.join(subclass_output, "voc_results.csv"))
        plot_utils.plot_all(voc_metrics, cm, subclass_output, subclass_dir_pred, test_labels_dir, config)
        logging.info(f"Plotted metrics for {subclass_output}")

        # Map subclass predictions to superclasses using the mapping module.
        logging.info(f"Mapping subclass predictions to superclasses for {subclass_dir_pred}")
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
        logging.info(f"Computing detection metrics for {superclass_dir_pred}")
        superclass_metrics, cm = metrics.compute_detection_metrics(superclass_dir_pred, superclass_labels_dir, args.img_size, config, avg_inference_speed=None, use_superclasses=True)
        metrics.save_metrics_csv(superclass_metrics, os.path.join(superclass_output, "results.csv"))
        voc_metrics = metrics.compute_pascalvoc_metrics(superclass_dir_pred, superclass_labels_dir, iou_threshold=0.5)
        metrics.save_voc_metrics_csv(voc_metrics, os.path.join(superclass_output, "voc_results.csv"))
        plot_utils.plot_all(voc_metrics, cm, superclass_output, superclass_dir_pred, superclass_labels_dir, config, is_superclass=True)
        logging.info(f"Plotted metrics for {superclass_output}")

if __name__ == "__main__":
    setup_logger()
    main()
