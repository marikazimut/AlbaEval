# object_detection/mapping.py

import os

def _read_predictions_from_txt(predictions_dir):
    """
    Reads prediction text files from the given directory.
    
    Each file is expected to contain lines formatted as:
         label score x1 y1 x2 y2
    The image name is inferred from the file name.
    
    Returns:
        List[dict]: Each dict contains "filename" and "detections" (a list of detection dicts).
    """
    predictions = []
    for txt_file in sorted(os.listdir(predictions_dir)):
        if txt_file.endswith('.txt'):
            # Infer image name from the file name (adjust the extension if needed)
            image_name = os.path.splitext(txt_file)[0] + ".jpg"
            pred = {"filename": image_name, "detections": []}
            file_path = os.path.join(predictions_dir, txt_file)
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue  # skip invalid lines
                    label = parts[0]
                    if len(parts) == 6:
                        try:
                            score = float(parts[1])
                        except ValueError:
                            print(f"Warning: Invalid score in {txt_file} line: {line}")
                            continue  # skip lines with invalid score
                        bbox = list(map(float, parts[2:6]))  # [x1, y1, x2, y2]
                        pred["detections"].append({
                            "label": label,
                            "score": score,
                            "bbox": bbox
                        })
                    else:
                        bbox = list(map(float, parts[1:5]))  # [x1, y1, x2, y2]
                        pred["detections"].append({
                            "label": label,
                            "bbox": bbox
                        })
            predictions.append(pred)
    return predictions

def _read_labels_from_txt(labels_dir):
    """
    Reads label text files from the given directory.
    
    Each file is expected to contain lines formatted as:
         label score x1 y1 x2 y2
    The image name is inferred from the file name.
    
    Returns:
        List[dict]: Each dict contains "filename" and "detections" (a list of detection dicts).
    """
    labels = []
    for txt_file in sorted(os.listdir(labels_dir)):
        if txt_file.endswith('.txt'):
            # Infer image name from the file name (adjust the extension if needed)
            image_name = os.path.splitext(txt_file)[0] + ".jpg"
            label = {"filename": image_name, "detections": []}
            file_path = os.path.join(labels_dir, txt_file)
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue  # skip invalid lines
                    label = parts[0]
                    bbox = list(map(float, parts[1:5]))  # Yolo format
                    label["detections"].append({
                        "label": label,
                        "bbox": bbox
                    })
            labels.append(label)
    return labels

# def map_to_superclasses(predictions_dir, labels_dir, config):
#     """
#     Reads subclass prediction text files from a directory, maps each detection's label to its 
#     corresponding superclass label, and returns new predictions in the same universal format.
    
#     The provided mapping (class_mapping) is expected to be a dictionary where keys are 
#     superclass labels and values are lists of subclass labels that should be mapped to that superclass.
    
#     Parameters:
#         predictions_dir (str): Directory containing the subclass prediction text files.
#         class_mapping (dict): Dictionary mapping superclass labels to lists of subclass labels.
        
#     Returns:
#         List[dict]: New predictions with detections mapped to their superclasses.
#     """
#     # First, check if the predictions are stored in a subfolder named "predictions".
#     # If not, assume the provided directory already contains the text files.
#     predictions_path = os.path.join(predictions_dir, "predictions")
#     if not os.path.exists(predictions_path):
#         predictions_path = predictions_dir

#     predictions = _read_predictions_from_txt(predictions_path)
    
#     class_to_index = config["name_to_index"]
#     class_mapping = config["mappings"]

#     # Build a reverse mapping: for each subclass, know its superclass.
#     # Example: if class_mapping["Merchant"] = ["Bulk", "Containers", "Merchant", ...],
#     # then reverse_mapping["Bulk"] = "Merchant", etc.
#     reverse_mapping = {}
#     for superclass, subclasses in class_mapping.items():
#         for subclass in subclasses:
#             reverse_mapping[subclass] = superclass

#     index_to_class = {}
#     for class_name, class_index in class_to_index.items():
#         index_to_class[class_index] = class_name

#     # Create new predictions with the detection labels replaced by their superclasses.
#     superclass_predictions = []
#     for pred in predictions:
#         new_pred = {"filename": pred["filename"], "detections": []}
#         for det in pred["detections"]:
#             orig_label = det["label"]
#             orig_label = index_to_class[int(orig_label)]
#             new_label = reverse_mapping.get(orig_label)
#             if new_label is None:
#                 print(f"Warning: {orig_label} not found in reverse mapping.")
#             new_label_index = class_to_index[new_label]
#             new_det = det.copy()
#             new_det["label"] = new_label_index
#             new_pred["detections"].append(new_det)
#         superclass_predictions.append(new_pred)
    
#     # Also return the labels for comparison
#     superclass_labels = []
#     superclass_labels_dir = labels_dir.replace("labels", "superclass_labels")
#     if not os.path.exists(superclass_labels_dir):
#         os.makedirs(superclass_labels_dir)
#         labels = _read_labels_from_txt(labels_dir)  
#         superclass_labels = []
#         for label in labels:
#             new_label = {"filename": label["filename"], "detections": []}
#             for det in label["detections"]:
#                 orig_label = det["label"]
#                 orig_label = index_to_class[int(orig_label)]
#                 new_label = reverse_mapping.get(orig_label)
#                 if new_label is None:
#                     print(f"Warning: {orig_label} not found in reverse mapping.")
#                 new_label_index = class_to_index[new_label]
#                 new_det = det.copy()
#                 new_det["label"] = new_label_index
#                 new_label["detections"].append(new_det)
#             superclass_labels.append(new_label)

#     return superclass_predictions, superclass_labels

def map_to_superclasses(predictions_dir, config):
    """
    Reads subclass prediction text files from a directory, maps each detection's label to its 
    corresponding superclass label, and returns new predictions in the same universal format.
    
    The provided mapping (class_mapping) is expected to be a dictionary where keys are 
    superclass labels and values are lists of subclass labels that should be mapped to that superclass.
    
    Parameters:
        predictions_dir (str): Directory containing the subclass prediction text files.
        class_mapping (dict): Dictionary mapping superclass labels to lists of subclass labels.
        
    Returns:
        List[dict]: New predictions with detections mapped to their superclasses.
    """
    # First, check if the predictions are stored in a subfolder named "predictions".
    # If not, assume the provided directory already contains the text files.
    predictions = _read_predictions_from_txt(predictions_dir)
    
    class_to_index = config["name_to_index"]
    class_mapping = config["mappings"]

    # Build a reverse mapping: for each subclass, know its superclass.
    # Example: if class_mapping["Merchant"] = ["Bulk", "Containers", "Merchant", ...],
    # then reverse_mapping["Bulk"] = "Merchant", etc.
    reverse_mapping = {}
    for superclass, subclasses in class_mapping.items():
        for subclass in subclasses:
            reverse_mapping[subclass] = superclass

    index_to_class = {}
    for class_name, class_index in class_to_index.items():
        index_to_class[class_index] = class_name

    # Create new predictions with the detection labels replaced by their superclasses.
    superclass_predictions = []
    for pred in predictions:
        new_pred = {"filename": pred["filename"], "detections": []}
        for det in pred["detections"]:
            orig_label = det["label"]
            orig_label = index_to_class[int(orig_label)]
            new_label = reverse_mapping.get(orig_label)
            if new_label is None:
                print(f"Warning: {orig_label} not found in reverse mapping.")
            new_label_index = class_to_index[new_label]
            new_det = det.copy()
            new_det["label"] = new_label_index
            new_pred["detections"].append(new_det)
        superclass_predictions.append(new_pred)

    return superclass_predictions
