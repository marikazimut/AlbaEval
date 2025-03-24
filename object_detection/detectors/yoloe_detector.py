# object_detection/detectors/yoloe_detector.py
import os
from object_detection.external_libraries.yoloe.ultralytics import YOLOE
import json
import time
import torch
import sys

# current_dir = os.path.dirname(os.path.abspath(__file__))
# local_ultralytics_dir = os.path.join(current_dir, '..', 'external_libraries', 'yoloe')
# sys.path.insert(0, local_ultralytics_dir)

class YOLOEDetector:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def run_inference(self, images_dir, img_size, models_dir, weights_file, prompt_type="text"):
        # Load the appropriate weights for YOLO11 from the models folder.

        # unfused_model = YOLOE("yoloe-v8l.yaml")
        # unfused_model.load("object_detection\external_libraries\yoloe\pretrain\yoloe-v8l-seg.pt")
        # unfused_model.eval()
        # unfused_model.cuda()

        # with open(r'object_detection\external_libraries\yoloe\tools\ram_tag_list.txt', 'r') as f:
        #     names = [x.strip() for x in f.readlines()]
        # vocab = unfused_model.get_vocab(names)



        model_path = os.path.join(models_dir, self.model_name, weights_file)
        model = YOLOE(model_path)
        # model = YOLOE("object_detection\external_libraries\yoloe\pretrain\yoloe-v8l-seg.pt")
        # model = YOLOE(r"C:\Users\offic\OneDrive\Desktop\Evaluation-pipeline\object_detection\models\yoloe-11m\yoloe-11m-seg.pt")
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # model.set_vocab(vocab, names=names)
        # model.model.model[-1].is_fused = True
        # model.model.model[-1].conf = 0.001
        # model.model.model[-1].max_det = 1000

        predictions = []
        inference_times = []
        corrupted_images = []

        # model.args["task"] = "detect"
        if prompt_type == "text":
            # names = ['Merchant', 'Military', 'SWC', 'Support', 'Dvora', 'Motor', 'Patrol-Boat', 'Pilot', 'Bulk', 'Containers', 'General-Cargo', 'Ro-Ro', 'Tanker', 'Saar-4.5', 'Saar-5', 'Submarine', 'Buoy', 'Sailing', 'Cruise', 'Ferry', 'Supply', 'Tug', 'Yacht', 'Fishing', 'Rubber', 'Patrol', 'Saar-6', 'BWC']
            names = ["barge", "seaplane", "fish boat", "sailboat", "ferry", "yacht", "boat", "ship", "vessel", "cargo ship", "container ship", "cruise ship", "fishing boat", "military ship", "patrol boat", "patrol ship", "supply ship", "tanker", "tugboat"]
            model.set_classes(names, model.get_text_pe(names))

        elif prompt_type == "free":
            unfused_model = YOLOE(os.path.join(models_dir, self.model_name, "yoloe-11m-seg.pt"))
            # unfused_model.load()
            unfused_model.eval()
            unfused_model.cuda()
            with open(r'object_detection\external_libraries\yoloe\tools\ram_tag_list.txt', 'r') as f:
                names = [x.strip() for x in f.readlines()]
            vocab = unfused_model.get_vocab(names)

            model.set_vocab(vocab, names=names)
            # model.model.model[-1].is_fused = True
            # model.model.model[-1].conf = 0.001
            # model.model.model[-1].max_det = 1000

        # Iterate over test images and run inference.
        for image_file in sorted(os.listdir(images_dir)):
            image_path = os.path.join(images_dir, image_file)
            
            try:
                start_time = time.time()
                results = model.predict(image_path, conf=0.01, imgsz=img_size, verbose=False)  # pseudocode: adjust as needed
                end_time = time.time()
                inference_times.append(end_time - start_time)
            
                # Convert the results to a universal format (e.g., a dict with keys: filename, boxes, labels, scores)
                # (Assume results[0].boxes exists; adjust if necessary.)
                results_boxes = results[0].boxes  
                universal_results = self._convert_to_universal(results_boxes, image_file)
                predictions.append(universal_results)
            except:
                # print(f"Error running inference for {image_file}: corrupted image")
                corrupted_images.append(image_file)
                continue
        
        # Compute average inference time (in seconds per image)
        avg_inference_speed = sum(inference_times) / len(inference_times) if inference_times else 0
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
