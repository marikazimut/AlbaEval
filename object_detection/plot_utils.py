# object_detection/plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score
import pandas as pd

def plot_all(metrics_results, output_dir):
    """
    Generates various evaluation plots for object detection performance.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- CONFUSION MATRIX ---
    if "confusion_matrix" in metrics_results:
        cm = metrics_results["confusion_matrix"]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix (Counts)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_counts.png"))
        plt.close()

        # Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
        plt.title("Confusion Matrix (Normalized)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"))
        plt.close()

    # --- PRECISION-RECALL CURVE ---
    if "precision_recall" in metrics_results:
        precisions, recalls, thresholds = metrics_results["precision_recall"]
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='.', label="Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()

    # --- F1-SCORE CURVE ---
    if "f1_curve" in metrics_results:
        f1_scores, f1_thresholds = metrics_results["f1_curve"]
        plt.figure(figsize=(8, 6))
        plt.plot(f1_thresholds, f1_scores, marker='.', label="F1-Score Curve")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("F1-Score")
        plt.title("F1-Score Curve")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, "f1_score_curve.png"))
        plt.close()

    # --- IoU DISTRIBUTION HISTOGRAM ---
    if "iou_distribution" in metrics_results:
        ious = metrics_results["iou_distribution"]
        plt.figure(figsize=(8, 6))
        sns.histplot(ious, bins=20, kde=True)
        plt.xlabel("IoU")
        plt.ylabel("Count")
        plt.title("IoU Distribution Histogram")
        plt.savefig(os.path.join(output_dir, "iou_distribution.png"))
        plt.close()

    # --- CLASS-WISE AP & AR PLOTS ---
    if "classwise_AP" in metrics_results and "classwise_AR" in metrics_results:
        class_names = list(metrics_results["classwise_AP"].keys())
        ap_values = list(metrics_results["classwise_AP"].values())
        ar_values = list(metrics_results["classwise_AR"].values())

        df = pd.DataFrame({'Class': class_names, 'AP': ap_values, 'AR': ar_values})
        df = df.sort_values(by="AP", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="AP", y="Class", data=df, palette="Blues_r")
        plt.xlabel("Average Precision (AP)")
        plt.title("Per-Class AP")
        plt.savefig(os.path.join(output_dir, "classwise_AP.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.barplot(x="AR", y="Class", data=df, palette="Greens_r")
        plt.xlabel("Average Recall (AR)")
        plt.title("Per-Class AR")
        plt.savefig(os.path.join(output_dir, "classwise_AR.png"))
        plt.close()

    # --- BOX SIZE DISTRIBUTION ---
    if "box_sizes" in metrics_results:
        box_sizes = metrics_results["box_sizes"]
        plt.figure(figsize=(8, 6))
        sns.histplot(box_sizes, bins=30, kde=True)
        plt.xlabel("Bounding Box Area (pixels)")
        plt.ylabel("Count")
        plt.title("Bounding Box Size Distribution")
        plt.savefig(os.path.join(output_dir, "box_size_distribution.png"))
        plt.close()
