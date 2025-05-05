# object_detection/detectors/yolo_detector.py
import os
from ultralytics import YOLO  # Assuming ultralytics is installed
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle

class YOLODetector:
    def __init__(self, model_name, save_runtime_obj_hist=False):
        self.model_name = model_name
        self.save_runtime_obj_hist = save_runtime_obj_hist
        self.objects_per_image = []
    
    def run_inference(self, images_dir, img_size, models_dir, weights_file):
        # Load the appropriate weights for YOLO11 from the models folder.
        model_path = os.path.join(models_dir, self.model_name, weights_file)
        model = YOLO(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        predictions = []
        inference_times = []
        corrupted_images = []

        # Add this to track objects per image alongside inference times
        objects_per_image = []

        # Iterate over test images and run inference.
        for image_file in sorted(os.listdir(images_dir)):
            image_path = os.path.join(images_dir, image_file)
            
            try:
                start_time = time.time()
                results = model.predict(image_path, conf=0.05, imgsz=img_size, verbose=False)  # pseudocode: adjust as needed
                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)

                # Convert the results to a universal format (e.g., a dict with keys: filename, boxes, labels, scores)
                # (Assume results[0].boxes exists; adjust if necessary.)
                results_boxes = results[0].boxes  
                universal_results = self._convert_to_universal(results_boxes, image_file)
                predictions.append(universal_results)

                if self.save_runtime_obj_hist:
                    # Count the number of objects detected in this image
                    num_objects = len(universal_results["detections"])
                    # Store tuple of (image_file, num_objects, inference_time)
                    objects_per_image.append({
                        "filename": image_file,
                        "image_path": image_path,
                        "num_objects": num_objects,
                        "inference_time": inference_time
                    })

            except:
                print(f"Error running inference for {image_file}: corrupted image")
                # corrupted_images.append(image_file)
                continue
        
        # Compute average inference time (in seconds per image)
        avg_inference_speed = sum(inference_times) / len(inference_times) if inference_times else 0

        # Store objects_per_image for use in save_predictions
        self.objects_per_image = objects_per_image

        return predictions, avg_inference_speed, corrupted_images  # return both predictions and inference speed

    def _convert_to_universal(self, results, image_file):
        # Convert results from YOLO to your universal format.
        # This is where you translate between YOLO's output and your “Slack” format.
        universal = {
            "filename": image_file,
            "detections": []
        }
        for det in results:  # Pseudocode: iterate over detections.
            x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy().tolist()[0])
            universal["detections"].append({
                "bbox": [x1, y1, x2, y2],     # Example: [x1, y1, x2, y2]
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

        # Generate and save histogram if enabled
        output_dir = os.path.dirname(output_dir_pred)
        if self.save_runtime_obj_hist and hasattr(self, 'objects_per_image') and self.objects_per_image:
            self._generate_runtime_histogram(output_dir)

    def _generate_runtime_histogram(self, output_dir):
        if not self.objects_per_image:
            print("No data available to generate histogram")
            return
            
        # Extract data for plotting
        object_counts = [item["num_objects"] for item in self.objects_per_image]
        inference_times = [item["inference_time"] for item in self.objects_per_image]
        
        # Group inference times by object count
        object_count_groups = {}
        for item in self.objects_per_image:
            obj_count = item["num_objects"]
            if obj_count not in object_count_groups:
                object_count_groups[obj_count] = []
            object_count_groups[obj_count].append(item["inference_time"])
        
        # Calculate statistics for each group
        x_values = sorted(object_count_groups.keys())
        y_values = [np.mean(object_count_groups[count]) for count in x_values]
        error_bars = [np.std(object_count_groups[count]) for count in x_values]
        
        # Create histogram figure
        plt.figure(figsize=(12, 6))
        
        # Bar chart of average inference time by object count
        plt.bar(x_values, y_values, yerr=error_bars, alpha=0.7)
        plt.xlabel('Number of Objects Detected')
        plt.ylabel('Average Inference Time (seconds)')
        plt.title('Inference Time vs Number of Detected Objects')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the histogram
        histogram_path = os.path.join(output_dir, 'inference_time_histogram.png')
        plt.savefig(histogram_path)
        plt.close()
        
        # Save the full object data for future analysis
        self._save_objects_per_image_data(output_dir)
        
        print(f"Histogram saved to {histogram_path}")

    def _save_objects_per_image_data(self, output_dir):
        """Save the objects_per_image data in multiple formats for later analysis"""
        
        # 1. Save as CSV for easy viewing
        csv_path = os.path.join(output_dir, 'objects_inference_data.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'image_path', 'num_objects', 'inference_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in self.objects_per_image:
                writer.writerow(item)
        
        # 2. Save as pickle for easy loading in Python
        pickle_path = os.path.join(output_dir, 'objects_inference_data.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.objects_per_image, f)
            
        # 3. Save a report of outliers - images with unusually high inference times
        self._save_outlier_report(output_dir)
        
        print(f"Raw data saved as CSV to {csv_path}")
        print(f"Raw data saved as pickle to {pickle_path}")
        
    def _save_outlier_report(self, output_dir):
        """Generate a report of outliers (images with unusually high inference times)"""
        if not self.objects_per_image:
            return
            
        # Calculate mean and standard deviation of inference times
        inference_times = [item["inference_time"] for item in self.objects_per_image]
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        # Define outliers as those more than 2 standard deviations above mean
        threshold = mean_time + 2 * std_time
        
        # Find outliers
        outliers = [item for item in self.objects_per_image if item["inference_time"] > threshold]
        
        # Sort outliers by inference time (descending)
        outliers.sort(key=lambda x: x["inference_time"], reverse=True)
        
        # Save outlier report
        report_path = os.path.join(output_dir, 'inference_outliers.txt')
        with open(report_path, 'w') as f:
            f.write(f"Inference Time Analysis\n")
            f.write(f"=====================\n\n")
            f.write(f"Mean inference time: {mean_time:.4f} seconds\n")
            f.write(f"Standard deviation: {std_time:.4f} seconds\n")
            f.write(f"Outlier threshold (mean + 2*std): {threshold:.4f} seconds\n\n")
            
            f.write(f"Outliers (sorted by inference time):\n")
            f.write(f"-----------------------------------\n")
            for item in outliers:
                f.write(f"File: {item['filename']}\n")
                f.write(f"  Path: {item['image_path']}\n")
                f.write(f"  Objects detected: {item['num_objects']}\n")
                f.write(f"  Inference time: {item['inference_time']:.4f} seconds\n\n")
                
        if outliers:
            print(f"Outlier report saved to {report_path}")
