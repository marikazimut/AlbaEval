# object_detection/detectors/yolo_detector.py
import os
from ultralytics import YOLO  # Assuming ultralytics is installed
import json
import time
import torch

class YOLODetector:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def run_inference(self, images_dir, img_size, models_dir, weights_file):
        # Load the appropriate weights for YOLO11 from the models folder.
        model_path = os.path.join(models_dir, self.model_name, weights_file)
        model = YOLO(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        predictions = []
        inference_times = []

        # Iterate over test images and run inference.
        for image_file in sorted(os.listdir(images_dir)):
            image_path = os.path.join(images_dir, image_file)
            
            start_time = time.time()
            results = model.predict(image_path, imgsz=img_size, verbose=False)  # pseudocode: adjust as needed
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Convert the results to a universal format (e.g., a dict with keys: filename, boxes, labels, scores)
            # (Assume results[0].boxes exists; adjust if necessary.)
            results_boxes = results[0].boxes  
            universal_results = self._convert_to_universal(results_boxes, image_file)
            predictions.append(universal_results)
        
        # Compute average inference time (in seconds per image)
        avg_inference_speed = sum(inference_times) / len(inference_times) if inference_times else 0
        return predictions, avg_inference_speed  # return both predictions and inference speed
    
    
    def _convert_to_universal(self, results, image_file):
        # Convert results from YOLO to your universal format.
        # This is where you translate between YOLO's output and your “Slack” format.
        universal = {
            "filename": image_file,
            "detections": []
        }
        for det in results:  # Pseudocode: iterate over detections.
            universal["detections"].append({
                "bbox": map(int, det.xyxy.cpu().numpy().tolist()[0]),     # Example: [x1, y1, x2, y2]
                "score": det.conf.cpu().item(),    # Confidence
                "label": int(det.cls.cpu().item())      # Class label (subclass)
            })
        return universal
    
    def save_predictions(self, predictions, output_dir_pred):
        # Save predictions as text files, one per image or one consolidated file.
        os.makedirs(output_dir_pred, exist_ok=True)
        for pred in predictions:
            filename = pred["filename"].split('.')[0] + ".txt"
            output_path = os.path.join(output_dir_pred, filename)
            with open(output_path, "w") as f:
                for det in pred["detections"]:
                    # Write in a consistent format, e.g., label score x1 y1 x2 y2
                    if "score" in det:
                        f.write(f'{det["label"]} {det["score"]:.4f} ' +
                                " ".join(map(str, det["bbox"])) + "\n")
                    else:
                        f.write(f'{det["label"]} ' +
                                " ".join(map(str, det["bbox"])) + "\n")
