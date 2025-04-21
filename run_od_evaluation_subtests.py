import argparse
import os
import yaml
import pathlib
import json
from object_detection import metrics, plot_utils, mapping
import logging
import shutil

# Dynamically import detector modules
from object_detection.detectors import yolo_detector  #, detr_detector  # (detr_detector might be used later)

def setup_logger(log_file="pipeline.log"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Evaluation Pipeline")
    parser.add_argument("--model", type=str, default="yolo11s", help="Model name (e.g., 'yolo11s', 'yolo9', 'detr')")
    parser.add_argument("--weights", type=str, default="Albatross-v0.4.pt", help="Model weights file")
    parser.add_argument("--test_set", type=str, default="Albatross-Dataset-v0.4-test", help="Test-set version to evaluate (e.g. 'Albatross-Dataset-v0.4-test', 'Albatross-Dataset-v0.4-test - combined_testset', 'yoloe-test')")
    # Note: img_size here is expected to be a list of two integers, e.g., [1088, 1920]
    parser.add_argument("--img_size", type=int, nargs=2, default=[1088, 1920], help="Input image size for the detector")
    # New argument for JSON folder containing the original subset JSON files
    parser.add_argument("--json_dir", type=str, default=r"C:\Users\offic\OneDrive\Desktop\Azimut-Labeling", help=r"Path to folder containing subset JSON files. (r'C:\Users\offic\OneDrive\Desktop\Azimut-Labeling')")
    return parser.parse_args()

def load_config(config_path="object_detection/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def select_detector(model_name):
    if model_name.lower() in ["yolo11s", "yolo9"]:
        return yolo_detector.YOLODetector(model_name)
    # elif model_name.lower() == "detr":
    #     return detr_detector.DETRDetector(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_avg_inference_speed(args, config, detector):
    pass

def process_subtest(subtest, args, config, detector, is_combined=False):
    # Define paths based on the project structure
    test_images_dir = os.path.join("object_detection", "test_sets", args.test_set, subtest, "images")
    test_labels_dir = os.path.join("object_detection", "test_sets", args.test_set, subtest, "labels")
    model_weights_dir = os.path.join("object_detection", "models")
    output_dir = os.path.join("object_detection", "outputs", args.model, pathlib.Path(args.weights).stem, subtest)

    # Ensure output subdirectories exist:
    subclass_output = os.path.join(output_dir, "subclass")
    superclass_output = os.path.join(output_dir, "superclass")
    subclass_dir_pred = os.path.join(subclass_output, "predictions")
    os.makedirs(subclass_dir_pred, exist_ok=True)

    predictions = None  # will hold predictions if inference is run

    if not is_combined:
        # Run detector on test-set images
        predictions, avg_inference_speed, corrupted_images = detector.run_inference(
            test_images_dir, args.img_size, model_weights_dir, args.weights)
        logging.info(f"Finished running inference on {test_images_dir}")

        if len(corrupted_images) > 0:
            logging.warning(f"Corrupted images: {corrupted_images}")
            print("Removing corrupted images from test-set to corrupted folder")
            corrupted_dir = os.path.join("object_detection", "test_sets", "corrupted", args.test_set, subtest)
            os.makedirs(corrupted_dir, exist_ok=True)
            for image in corrupted_images:
                image_extension = os.path.splitext(image)[1]
                label_name = image.replace(image_extension, ".txt")
                shutil.move(os.path.join(test_images_dir, image), os.path.join(corrupted_dir, image))
                shutil.move(os.path.join(test_labels_dir, label_name), os.path.join(corrupted_dir, label_name))
                print(f"Corrupted image removed: {image}")

        detector.save_predictions(predictions, subclass_dir_pred)
        logging.info(f"Saved predictions to {subclass_dir_pred}")
    else:
        # TODO: get avg inference speed from combined predictions
        avg_inference_speed = 0

    # Compute evaluation metrics and graphs for subclass outputs.
    logging.info(f"Computing detection metrics for {subclass_dir_pred}")
    metrics_results, cm = metrics.compute_detection_metrics(subclass_dir_pred, test_labels_dir, args.img_size, config, avg_inference_speed=avg_inference_speed)
    metrics.save_metrics_csv(metrics_results, os.path.join(subclass_output, "results.csv"))
    voc_metrics = metrics.compute_pascalvoc_metrics(subclass_dir_pred, test_labels_dir, iou_threshold=0.5)
    metrics.save_voc_metrics_csv(voc_metrics, os.path.join(subclass_output, "voc_results.csv"))
    plot_utils.plot_all(voc_metrics, cm, subclass_output, subclass_dir_pred, test_labels_dir, config)
    logging.info(f"Plotted metrics for {subclass_output}")

    if not is_combined:
        # Map subclass predictions to superclasses using the mapping module.
        logging.info(f"Mapping subclass predictions to superclasses for {subclass_dir_pred}")
        superclass_dir_pred = os.path.join(superclass_output, "predictions")
        if not os.path.exists(superclass_dir_pred):
            superclass_predictions = mapping.map_to_superclasses(subclass_dir_pred, config)
            detector.save_predictions(superclass_predictions, superclass_dir_pred)
        else:
            print(f"{superclass_dir_pred} already exists. Skipping mapping.")
        superclass_labels_dir = test_labels_dir.replace("labels", "superclass_labels")
        if not os.path.exists(superclass_labels_dir):
            superclass_labels = mapping.map_to_superclasses(test_labels_dir, config)
            detector.save_predictions(superclass_labels, superclass_labels_dir)
        else:
            print(f"{superclass_labels_dir} already exists. Skipping mapping.")
    else:
        superclass_dir_pred = os.path.join(superclass_output, "predictions")
        superclass_labels_dir = test_labels_dir.replace("labels", "superclass_labels")

    # Compute and save metrics/plots for superclass predictions.
    logging.info(f"Computing detection metrics for {superclass_dir_pred}")
    superclass_metrics, cm = metrics.compute_detection_metrics(
        superclass_dir_pred, superclass_labels_dir, args.img_size, config, avg_inference_speed=None, use_superclasses=True)
    metrics.save_metrics_csv(superclass_metrics, os.path.join(superclass_output, "results.csv"))
    voc_metrics = metrics.compute_pascalvoc_metrics(superclass_dir_pred, superclass_labels_dir, iou_threshold=0.5)
    metrics.save_voc_metrics_csv(voc_metrics, os.path.join(superclass_output, "voc_results.csv"))
    plot_utils.plot_all(voc_metrics, cm, superclass_output, superclass_dir_pred, superclass_labels_dir, config, is_superclass=True)
    logging.info(f"Plotted metrics for {superclass_output}")

    return predictions  # return predictions for use in JSON update (if inference was run)

def update_json_predictions_for_subtest(subtest, predictions, args, config):
    """
    For the given subtest:
      - Load the original JSON file from the folder provided in args.json_dir (e.g. "test-AZIMUTHAIFA.json").
      - For each image entry (assumed to be a dict with a "data" field containing "image"),
        find the corresponding prediction (by matching base filenames) from the predictions list.
      - Replace the "prediction" field (and update "model_version") with the new predictions.
      - Save the updated JSON file in the subtest's output folder.
    """
    # Define paths
    original_json_path = os.path.join(args.json_dir, f"{subtest}.json")
    output_dir = os.path.join("object_detection", "outputs", args.model, pathlib.Path(args.weights).stem, subtest)
    updated_json_path = os.path.join(output_dir, f"{subtest}_updated.json")

    unmapped_images = []

    name_to_index = config["name_to_index"]
    index_to_name = {str(v): k for k, v in name_to_index.items()}

    if not os.path.exists(original_json_path):
        logging.warning(f"JSON file for subtest '{subtest}' not found at {original_json_path}. Skipping JSON update.")
        return

    # Load original JSON content
    with open(original_json_path, "r") as f:
        json_data = json.load(f)

    # Build a lookup for predictions by base filename.
    # We assume each prediction has a "filename" key.
    pred_lookup = {}
    for pred in predictions:
        # Remove extension to get base name.
        base_name = os.path.splitext(pred["filename"])[0]
        pred_lookup[base_name] = pred

    # Update each entry in the JSON file.
    for entry in json_data:
        image_path = entry.get("data", {}).get("image", "")
        if not image_path:
            continue
        # Extract the file name from the image path and remove any added tokens (e.g. '.rf.'):
        json_filename = os.path.basename(image_path)
        # E.g., "Ashdod-Port_YUVEL-THERMAL_2024_05_23-14_06_frame_0_jpg.rf.51774a4212c1574c2df0e3545128929b.jpg"
        # We split at ".rf." to get the base.
        json_base = json_filename.split(".")[0]
        # Try to find a matching prediction using the base filename.
        if json_base in pred_lookup:
            pred = pred_lookup[json_base]
            # Get the original prediction (if any) to preserve id/task and image dimensions.
            orig_pred = entry.get("annotations", {})[0].get("prediction", {})
            # Determine original dimensions – try to read from the first result if available; otherwise use fallback.
            if orig_pred.get("result") and len(orig_pred["result"]) > 0:
                orig_width = orig_pred["result"][0].get("original_width", args.img_size[1])
                orig_height = orig_pred["result"][0].get("original_height", args.img_size[0])
            else:
                orig_width = args.img_size[1]
                orig_height = args.img_size[0]

            new_results = []
            # Convert each detection in the prediction to the JSON result format.
            for det in pred["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                # Normalize coordinates to percentages relative to the original image size.
                x_norm = (x1 / orig_width) * 100
                y_norm = (y1 / orig_height) * 100
                width_norm = ((x2 - x1) / orig_width) * 100
                height_norm = ((y2 - y1) / orig_height) * 100
                result_item = {
                    "type": "rectanglelabels",
                    "value": {
                        "x": x_norm,
                        "y": y_norm,
                        "width": width_norm,
                        "height": height_norm,
                        "rectanglelabels": [str(index_to_name[str(det["label"])])]
                    },
                    "to_name": "image",
                    "from_name": "label",
                    "original_width": orig_width,
                    "original_height": orig_height
                }
                new_results.append(result_item)

            # Build the new prediction field.
            new_prediction = {
                "id": orig_pred.get("id"),
                "task": orig_pred.get("task"),
                "model": args.weights,  # update with current model weights (version)
                "score": None,
                "result": new_results
            }
            entry["prediction"] = new_prediction
            # Also update (or add) the top-level model_version field.
            entry["model_version"] = args.weights
        
        else:
            # print(f"No prediction found for {json_base}!!!!!!!!!")
            unmapped_images.append(json_base)

    # Save the updated JSON file to the output folder.
    with open(updated_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    logging.info(f"Updated JSON file saved to {updated_json_path}")

    return unmapped_images

def combine_predictions(args, subtests):
    # copy all predictions from subtests to one folder - subclass and superclass
    combined_output_dir = os.path.join("object_detection", "outputs", args.model, pathlib.Path(args.weights).stem, "COMBINED")
    os.makedirs(combined_output_dir, exist_ok=True)
    subclass_predictions_dir = os.path.join(combined_output_dir, "subclass", "predictions")
    superclass_predictions_dir = os.path.join(combined_output_dir, "superclass", "predictions")
    os.makedirs(subclass_predictions_dir, exist_ok=True)
    os.makedirs(superclass_predictions_dir, exist_ok=True)
    for subtest in subtests:
        output_dir = os.path.join("object_detection", "outputs", args.model, pathlib.Path(args.weights).stem, subtest)
        for dir in ["subclass", "superclass"]:
            dir_pred = os.path.join(output_dir, dir, "predictions")
            for file in os.listdir(dir_pred):
                shutil.copy(os.path.join(dir_pred, file), os.path.join(combined_output_dir, dir, "predictions"))

def combine_gt_labels_and_images(args, subtests):
    combined_images_dir = os.path.join("object_detection", "test_sets", args.test_set, "COMBINED", "images")
    combined_labels_dir = os.path.join("object_detection", "test_sets", args.test_set, "COMBINED", "labels")
    combined_superclass_labels_dir = os.path.join("object_detection", "test_sets", args.test_set, "COMBINED", "superclass_labels")
    if os.path.exists(combined_labels_dir):
        print(f"Combined labels directory already exists: {combined_labels_dir}")
        return
    os.makedirs(combined_labels_dir, exist_ok=True)
    os.makedirs(combined_images_dir, exist_ok=True)
    os.makedirs(combined_superclass_labels_dir, exist_ok=True)
    for subtest in subtests:
        test_labels_dir = os.path.join("object_detection", "test_sets", args.test_set, subtest, "labels")
        for file in os.listdir(test_labels_dir):
            shutil.copy(os.path.join(test_labels_dir, file), os.path.join(combined_labels_dir, file))
        test_images_dir = os.path.join("object_detection", "test_sets", args.test_set, subtest, "images")
        for file in os.listdir(test_images_dir):
            shutil.copy(os.path.join(test_images_dir, file), os.path.join(combined_images_dir, file))
        test_superclass_labels_dir = os.path.join("object_detection", "test_sets", args.test_set, subtest, "superclass_labels")
        for file in os.listdir(test_superclass_labels_dir):
            shutil.copy(os.path.join(test_superclass_labels_dir, file), os.path.join(combined_superclass_labels_dir, file))

def main():
    args = parse_args()
    config = load_config()

    # Choose the right detector
    detector = select_detector(args.model)
    
    subtests = ['test-AZIMUTHAIFA', 'test-VTS', 'test-YUVELPTZ', 'test-YUVELRGB', 'test-YUVELTHERMAL']
    # Process each subtest individually.
    for subtest in subtests:
        preds = process_subtest(subtest, args, config, detector)
        # If a JSON directory is provided and we have predictions, update the corresponding JSON file.
        if args.json_dir and preds is not None:
            unmapped_images = update_json_predictions_for_subtest(subtest, preds, args, config)
            if len(unmapped_images) > 0:
                print(f"Unmapped images: {len(unmapped_images)}")

    # Get combined metrics.
    combine_predictions(args, subtests)
    combine_gt_labels_and_images(args, subtests)
    process_subtest("COMBINED", args, config, detector, is_combined=True)

if __name__ == "__main__":
    setup_logger()
    main()
