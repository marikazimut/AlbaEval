import argparse
import os
import math
import io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import matplotlib.gridspec as gridspec
import numpy as np

# --- Helper Functions (updated) ---

def find_model_subfolder(weights_name, outputs_root):
    for model_type in os.listdir(outputs_root):
        model_type_path = os.path.join(outputs_root, model_type)
        if not os.path.isdir(model_type_path):
            continue
        candidate = os.path.join(model_type_path, weights_name, "subclass")
        if os.path.exists(candidate):
            return candidate, model_type
    raise FileNotFoundError(f"Could not find metrics for weights '{weights_name}' in {outputs_root}")

def load_model_metrics(weights_name, outputs_root):
    subclass_path, model_type = find_model_subfolder(weights_name, outputs_root)
    
    results_csv = os.path.join(subclass_path, "results.csv")
    voc_csv = os.path.join(subclass_path, "voc_results.csv")
    
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results CSV not found at {results_csv}")
    
    df_results = pd.read_csv(results_csv)
    metrics = {}
    for col in df_results.columns:
        metrics[col] = df_results.iloc[0][col]
    
    voc_tp = None
    voc_fp = None
    if os.path.exists(voc_csv):
        try:
            df_voc = pd.read_csv(voc_csv)
            df_voc['precision'] = pd.to_numeric(df_voc['precision'], errors='coerce')
            df_voc['recall'] = pd.to_numeric(df_voc['recall'], errors='coerce')
            voc_precision = df_voc['precision'].mean()
            voc_recall = df_voc['recall'].mean()
            voc_tp = df_voc['total TP'].sum()
            voc_fp = df_voc['total FP'].sum()
        except Exception as e:
            print(f"Warning: Failed to load/parse {voc_csv}: {e}")
            voc_precision, voc_recall = None, None
    else:
        voc_precision, voc_recall = None, None

    if voc_precision is not None:
        metrics['voc_precision'] = voc_precision
    if voc_recall is not None:
        metrics['voc_recall'] = voc_recall
    if voc_tp is not None:
        metrics['voc_tp'] = voc_tp
    if voc_fp is not None:
        metrics['voc_fp'] = voc_fp

    confusion_img = os.path.join(subclass_path, "confusion_matrix.png")
    if os.path.exists(confusion_img):
        metrics["confusion_matrix"] = confusion_img

    metrics['model_type'] = model_type
    metrics['weights_name'] = weights_name
    return metrics

def create_summary_table(metrics_list, model_names):
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    all_keys.discard('model_type')
    all_keys.discard('weights_name')
    all_keys.discard('confusion_matrix')  # handled separately
    summary = {}
    for key in sorted(all_keys):
        summary[key] = []
        for m in metrics_list:
            summary[key].append(m.get(key, float('nan')))
    summary_df = pd.DataFrame(summary, index=model_names)
    return summary_df.T

def get_model_colors(model_names):
    """
    Returns a dictionary mapping each model name to a distinct color.
    Here we use the tab10 colormap from matplotlib.
    """
    cmap = plt.get_cmap("tab10")
    colors = {}
    for i, model in enumerate(model_names):
        colors[model] = cmap(i % 10)
    return colors

def plot_metric_bar(ax, summary_df, metric_name, model_names, model_colors):
    label_name = metric_name
    # Map metric names if necessary
    if metric_name == "AR":
        metric_key = "AR100"
    elif metric_name == "TP_num":
        metric_key = "voc_tp"
    elif metric_name == "FP_num":
        metric_key = "voc_fp"
    else:
        metric_key = metric_name

    if metric_key not in summary_df.index:
        ax.text(0.5, 0.5, f"Metric {metric_name} not found", ha='center', fontsize=10)
        return

    values = summary_df.loc[metric_key]
    for i, model in enumerate(model_names):
        val = values.get(model, float('nan'))
        bar = ax.bar(i, val, color=model_colors.get(model, "gray"), edgecolor='black')
        ax.annotate(f'{val:.2f}', xy=(i, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    ax.set_title(label_name, fontsize=12)
    ax.set_ylabel(label_name, fontsize=10)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def plot_grouped_bars(ax, summary_df, metrics, group_labels, title, ylabel, model_names, model_colors):
    """
    Plot a grouped bar chart.
    
    - metrics: list of metric keys (e.g., ["AP", "AR"] or ["voc_tp", "voc_fp"])
    - group_labels: list of labels for each group (e.g., ["AP", "AR"] or ["TP", "FP"])
    """
    n_groups = len(metrics)
    n_models = len(model_names)
    bar_width = 0.8 / n_models
    indices = np.arange(n_groups)
    
    for i, model in enumerate(model_names):
        offsets = indices - 0.4 + (i + 0.5) * bar_width
        values = []
        for metric in metrics:
            # Use the metric key directly; the summary dataframe must contain it.
            if metric in summary_df.index:
                values.append(summary_df.loc[metric][model])
            else:
                values.append(np.nan)
        ax.bar(offsets, values, width=bar_width, color=model_colors.get(model, "gray"), edgecolor='black', label=model)
        for j, v in enumerate(values):
            ax.annotate(f'{v:.2f}', xy=(offsets[j], v),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(indices)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)

# --- Updated PDF Report Generation ---

def create_pdf_report(summary_df, args, confusion_images, model_colors):
    model_names = args.models
    comparison_models = "_".join(model_names)
    output_dir = os.path.join(args.output, comparison_models)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "comparison_summary.pdf")

    # Categories definition.
    # For Overall Performance we will merge AP/AR and TP/FP together.
    categories = [
        ("Overall Performance", {
            "explanation": (
                "AP: The overall area under the precision–recall curve averaged over multiple IoU thresholds "
                "(from 0.50 to 0.95 in steps of 0.05). AR: The average recall over multiple IoU thresholds. "
                "TP: The total number of true positives based on VOC metrics. FP: The total number of false positives based on VOC metrics."
            )
        }),
        ("Object Confusion", {
            "metrics": ["confusion_matrix"],
            "explanation": "Confusion matrix based on ground truth and detection bounding boxes."
        }),
        ("BBox Localization", {
            "metrics": ["AP50", "AP75"],
            "explanation": (
                "AP50: Calculated at a fixed IoU threshold of 0.50. "
                "AP75: Calculated at a stricter IoU threshold of 0.75. "
                "A gap between AP50 and AP75 can highlight issues with localization accuracy."
            )
        }),
        ("Scale-Specific Performance", {
            "metrics": ["APsmall", "APmedium", "APlarge", "ARsmall", "ARmedium", "ARlarge"],
            "explanation": (
                "AP/AR computed separately for small, medium, and large objects (using area thresholds) "
                "help identify if your model is struggling with objects of a certain scale."
            )
        })
    ]
    
    with PdfPages(output_file) as pdf:
        # Cover Page
        fig_cover = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        cover_title = "Object Detection Model Comparison"
        subtitle = "Models: " + ", ".join(model_names)
        plt.text(0.5, 0.6, cover_title, fontsize=28, ha='center', va='center')
        plt.text(0.5, 0.5, subtitle, fontsize=20, ha='center', va='center')
        pdf.savefig(fig_cover, bbox_inches='tight', dpi=600)
        plt.close(fig_cover)

        # Summary Text Page
        fig_summary = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        summary_lines = []
        summary_lines.append("## Model Comparison Summary")
        summary_lines.append("")
        summary_lines.append("**Models Compared:** " + ", ".join(model_names))
        summary_lines.append("")
        summary_lines.append("### Key Metrics (Overall)")
        summary_lines.append("")
        summary_lines.append(summary_df.to_markdown())
        if 'AP' in summary_df.index and not summary_df.loc['AP'].isnull().all():
            best_AP = summary_df.loc['AP'].idxmax()
            summary_lines.append(f"\n* **Observation:** {best_AP} achieved the highest overall AP, indicating superior detection performance.")
        if 'voc_recall' in summary_df.index and not summary_df.loc['voc_recall'].isnull().all():
            best_recall = summary_df.loc['voc_recall'].idxmax()
            summary_lines.append(f"* **Observation:** {best_recall} exhibits the highest recall, meaning it misses fewer objects.")
        summary_text = "\n".join(summary_lines)
        wrapped_text = "\n".join(textwrap.wrap(summary_text, width=90))
        plt.text(0.05, 0.95, wrapped_text, fontsize=10, va='top')
        pdf.savefig(fig_summary, bbox_inches='tight', dpi=600)
        plt.close(fig_summary)

        # Category Pages
        for cat_name, cat_data in categories:
            explanation = cat_data["explanation"]
            
            # Create a new outer figure for the full page.
            fig = plt.figure(figsize=(8.5, 11))
            gs_outer = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[0.15, 0.70, 0.15], figure=fig)
            
            # Title Section
            ax_title = fig.add_subplot(gs_outer[0])
            ax_title.axis('off')
            ax_title.text(0.5, 0.5, cat_name, fontsize=16, ha='center', va='center')
            
            # Composite Graphs Section:
            fig_comp = None
            if cat_name == "Overall Performance":
                # Merge AP and AR in one chart and TP and FP in another, side by side.
                fig_comp, (ax_group1, ax_group2) = plt.subplots(1, 2, figsize=(12, 4))
                # Group 1: AP and AR
                plot_grouped_bars(ax_group1, summary_df, metrics=["AP", "AR"],
                                  group_labels=["AP", "AR"],
                                  title="Precision-Recall Performance", ylabel="Score",
                                  model_names=model_names, model_colors=model_colors)
                # Group 2: TP and FP (using voc_tp and voc_fp keys)
                plot_grouped_bars(ax_group2, summary_df, metrics=["voc_tp", "voc_fp"],
                                  group_labels=["TP", "FP"],
                                  title="Detection Counts", ylabel="Count",
                                  model_names=model_names, model_colors=model_colors)
            elif cat_name == "BBox Localization":
                # Merge AP50 and AP75 into one grouped bar chart.
                fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
                plot_grouped_bars(ax_comp, summary_df, metrics=["AP50", "AP75"],
                                  group_labels=["AP50", "AP75"],
                                  title="BBox Localization", ylabel="Score",
                                  model_names=model_names, model_colors=model_colors)
            elif cat_name == "Scale-Specific Performance":
                # Create two charts: one for AP and one for AR by scale.
                fig_comp, (ax_ap, ax_ar) = plt.subplots(1, 2, figsize=(12, 4))
                plot_grouped_bars(ax_ap, summary_df, metrics=["APsmall", "APmedium", "APlarge"],
                                  group_labels=["Small", "Medium", "Large"],
                                  title="AP by Scale", ylabel="AP",
                                  model_names=model_names, model_colors=model_colors)
                plot_grouped_bars(ax_ar, summary_df, metrics=["ARsmall", "ARmedium", "ARlarge"],
                                  group_labels=["Small", "Medium", "Large"],
                                  title="AR by Scale", ylabel="AR",
                                  model_names=model_names, model_colors=model_colors)
            else:
                # For Object Confusion, plot the confusion matrices.
                n_metrics = cat_data.get("metrics", [])
                fig_comp, axes = plt.subplots(1, len(n_metrics), figsize=(8, 4))
                if len(n_metrics) == 1:
                    axes = [axes]
                for idx, metric in enumerate(n_metrics):
                    ax = axes[idx]
                    if metric == "confusion_matrix":
                        ax.set_title("Confusion Matrix", fontsize=12)
                        # Create subplots for each model's confusion matrix.
                        n_models = len(model_names)
                        fig_sub, sub_axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
                        if n_models == 1:
                            sub_axes = [sub_axes]
                        for j, model_name in enumerate(model_names):
                            sub_ax = sub_axes[j]
                            img_path = confusion_images.get(model_name, None)
                            if img_path and os.path.exists(img_path):
                                img = plt.imread(img_path)
                                sub_ax.imshow(img)
                                sub_ax.set_title(model_name, fontsize=10)
                            else:
                                sub_ax.text(0.5, 0.5, f"No image for {model_name}", ha='center', va='center')
                            sub_ax.axis('off')
                        buf = io.BytesIO()
                        fig_sub.tight_layout()
                        fig_sub.savefig(buf, format='png', dpi=600)
                        buf.seek(0)
                        sub_img = plt.imread(buf)
                        buf.close()
                        plt.close(fig_sub)
                        ax.imshow(sub_img)
                        ax.axis('off')
            # Save the composite graphs figure into an in-memory buffer.
            buf_comp = io.BytesIO()
            if fig_comp is not None:
                fig_comp.tight_layout()
                fig_comp.savefig(buf_comp, format='png', dpi=300)
                buf_comp.seek(0)
                comp_img = plt.imread(buf_comp)
                buf_comp.close()
                plt.close(fig_comp)
            else:
                comp_img = None

            # Place the composite image in the outer figure.
            ax_graph = fig.add_subplot(gs_outer[1])
            if comp_img is not None:
                ax_graph.imshow(comp_img)
            ax_graph.axis('off')
            
            # Explanation Section
            ax_expl = fig.add_subplot(gs_outer[2])
            ax_expl.axis('off')
            wrapped_explanation = "\n".join(textwrap.wrap(explanation, width=90))
            ax_expl.text(0.5, 0.5, wrapped_explanation, fontsize=10, ha='center', va='center')
            
            pdf.savefig(fig, bbox_inches='tight', dpi=600)
            plt.close(fig)

        # Conclusion Page
        fig_concl = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        conclusion = (
            "### Conclusion\n\n"
            "The comparison shows that each model exhibits distinct trade-offs. For instance, while one model "
            "may achieve high recall (fewer missed objects), another may offer higher precision (fewer false positives).\n\n"
            "Furthermore, differences in bounding box localization (e.g. differences between AP50 and AP75) highlight variations in the models' "
            "ability to precisely localize objects. Scale-specific performance metrics also help identify potential weaknesses on objects of different sizes. "
            "Additional domain-specific testing is recommended to determine the optimal model for a given application."
        )
        wrapped_conclusion = "\n".join(textwrap.wrap(conclusion, width=90))
        plt.text(0.05, 0.95, wrapped_conclusion, fontsize=12, va='top')
        pdf.savefig(fig_concl, bbox_inches='tight', dpi=600)
        plt.close(fig_concl)
    
    print(f"Comparison summary saved to {output_file}")

# --- Main Script (unchanged except for color mapping) ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare object detection models and generate a summary PDF report."
    )
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=['Albatross-v0.3', 'Albatross-v0.2'],
        help="List of model names (e.g., Albatross-v0.3 detr-v0.1)"
    )
    parser.add_argument(
        '--outputs_root',
        default=os.path.join("object_detection", "outputs"),
        help="Root directory of evaluation outputs."
    )
    parser.add_argument(
        '--output',
        default=os.path.join("object_detection", "comparison_reports"),
        help="Output directory for the summary report."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    weights_names = args.models
    metrics_list = []
    for weights in weights_names:
        try:
            metrics = load_model_metrics(weights, args.outputs_root)
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Error loading metrics for {weights}: {e}")
            return
    summary_df = create_summary_table(metrics_list, weights_names)
    confusion_images = {m['weights_name']: m.get("confusion_matrix", None) for m in metrics_list}
    model_colors = get_model_colors(args.models)
    create_pdf_report(summary_df, args, confusion_images, model_colors)

if __name__ == "__main__":
    main()
