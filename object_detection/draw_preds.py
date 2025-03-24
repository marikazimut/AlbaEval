#!/usr/bin/env python
import os
import argparse
import cv2

def draw_predictions(image, predictions):
    """
    Draws bounding boxes and prediction labels on the image.
    
    Each prediction is expected in the format:
    <class> <confidence> <xmin> <ymin> <xmax> <ymax>
    """
    for line in predictions:
        parts = line.split()
        if len(parts) < 6:
            continue  # skip lines that don't have enough parts
        class_id = parts[0]
        confidence = parts[1]
        try:
            xmin, ymin, xmax, ymax = map(int, parts[2:6])
        except ValueError:
            print("Skipping line due to invalid coordinate values:", line)
            continue
        # Draw rectangle (using green color and thickness of 2)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Put text above the box: "class: confidence"
        text = f"{class_id}: {confidence}"
        cv2.putText(image, text, (xmin, max(ymin - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def process_image(image_path, prediction_path, output_path):
    """
    Loads an image, reads its corresponding prediction file,
    draws the predictions on the image, and saves the result.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    if not os.path.exists(prediction_path):
        print(f"Prediction file not found for image: {image_path}")
        return

    with open(prediction_path, "r") as file:
        lines = file.readlines()
    
    # Filter out any separator lines (e.g. lines of dashes)
    predictions = [line.strip() for line in lines if line.strip() and not line.strip().startswith("-")]
    
    image_with_predictions = draw_predictions(image, predictions)
    cv2.imwrite(output_path, image_with_predictions)
    print(f"Saved annotated image to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Annotate test set images with predictions.")
    parser.add_argument("--model", type=str, default="yoloe-11m", help="Model name (e.g. yoloe-11m)")
    parser.add_argument("--run", type=str, default="yoloe-11m-seg_text_0.05_combined_testset", help="Run name (e.g. yoloe-11m-seg-yoloe-test)")
    parser.add_argument("--test_set", type=str, default=r"Albatross-Dataset-v0.4-test - combined_testset", help="Test set name (e.g. yoloe-test)")
    parser.add_argument("--images", type=list, default=[], help="List of image filenames to process (if empty, process all images with predictions)")
    args = parser.parse_args()

    # Define folder paths relative to the current directory ("Evaluation-pipeline/object_detection")
    base_dir = os.getcwd()
    predictions_dir = os.path.join(base_dir, "object_detection", "outputs", args.model, args.run, "subclass", "predictions")
    test_images_dir = os.path.join(base_dir, "object_detection", "test_sets", args.test_set, "images")
    output_dir = os.path.join(base_dir, "object_detection", "debug_runs", args.run)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Determine which images to process
    if args.images:
        image_files = args.images
    else:
        # If no image list is provided, process all prediction files.
        # Assumes prediction files are named as <base_name>.txt and corresponding images are <base_name>.png
        image_files = []
        for file in os.listdir(predictions_dir):
            if file.endswith(".txt"):
                base_name = os.path.splitext(file)[0]
                image_files.append(base_name + ".png")  # adjust extension as needed

    # Process each image
    for image_filename in image_files:
        image_path = os.path.join(test_images_dir, image_filename)
        prediction_filename = os.path.splitext(image_filename)[0] + ".txt"
        prediction_path = os.path.join(predictions_dir, prediction_filename)
        output_image_path = os.path.join(output_dir, image_filename)
        process_image(image_path, prediction_path, output_image_path)

if __name__ == "__main__":
    main()
