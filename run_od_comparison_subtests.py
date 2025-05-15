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

# --- Helper Functions ---

def find_model_folder(weights_name, outputs_root, folder_type):
    """
    Locate the folder (subclass or superclass) for a given weights_name 
    in the COMBINED folder. In this version, the candidate path is:
    outputs_root / <model_type> / <weights_name> / COMBINED / <folder_type>
    """
    for model_type in os.listdir(outputs_root):
        model_type_path = os.path.join(outputs_root, model_type)
        if not os.path.isdir(model_type_path):
            continue
        candidate = os.path.join(model_type_path, weights_name, "COMBINED", folder_type)
        if os.path.exists(candidate):
            return candidate, model_type
    raise FileNotFoundError(f"Could not find metrics for weights '{weights_name}' with folder type '{folder_type}' in {outputs_root}")

def load_metrics(weights_name, outputs_root, folder_type):
    folder_path, model_type = find_model_folder(weights_name, outputs_root, folder_type)
    
    results_csv = os.path.join(folder_path, "results.csv")
    voc_csv = os.path.join(folder_path, "voc_results.csv")
    
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results CSV not found at {results_csv}")
    
    df_results = pd.read_csv(results_csv)
    metrics_dict = {col: df_results.iloc[0][col] for col in df_results.columns}
    
    # Try to load VOC metrics if available
    if os.path.exists(voc_csv):
        try:
            df_voc = pd.read_csv(voc_csv)
            df_voc['precision'] = pd.to_numeric(df_voc['precision'], errors='coerce')
            df_voc['recall'] = pd.to_numeric(df_voc['recall'], errors='coerce')
            voc_precision = df_voc['precision'].mean()
            voc_recall = df_voc['recall'].mean()
            voc_tp = df_voc['total TP'].sum()
            voc_fp = df_voc['total FP'].sum()
            metrics_dict['voc_precision'] = voc_precision
            metrics_dict['voc_recall'] = voc_recall
            metrics_dict['voc_tp'] = voc_tp
            metrics_dict['voc_fp'] = voc_fp
        except Exception as e:
            print(f"Warning: Failed to load/parse {voc_csv}: {e}")
    
    # Add confusion matrix image if it exists
    confusion_img = os.path.join(folder_path, "confusion_matrix_.png")
    if os.path.exists(confusion_img):
        metrics_dict["confusion_matrix"] = confusion_img

    metrics_dict['model_type'] = model_type
    metrics_dict['weights_name'] = weights_name
    metrics_dict['folder_type'] = folder_type  # indicate whether these are subclass or superclass metrics
    return metrics_dict

def create_summary_table(metrics_list, model_names):
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    # Remove keys that we do not want to show in the table.
    for key in ['model_type', 'weights_name', 'folder_type', 'confusion_matrix']:
        all_keys.discard(key)
    summary = {}
    for key in sorted(all_keys):
        summary[key] = []
        for m in metrics_list:
            summary[key].append(m.get(key, float('nan')))
    summary_df = pd.DataFrame(summary, index=model_names)
    return summary_df.T

def get_model_colors(model_names):
    """
    Returns a dictionary mapping each model name to a distinct color using the tab10 colormap.
    """
    cmap = plt.get_cmap("tab10")
    colors = {}
    for i, model in enumerate(model_names):
        colors[model] = cmap(i % 10)
    return colors

def plot_metric_bar(ax, summary_df, metric_name, model_names, model_colors):
    label_name = metric_name
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
        ax.bar(i, val, color=model_colors.get(model, "gray"), edgecolor='black')
        ax.annotate(f'{val:.2f}', xy=(i, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    ax.set_title(label_name, fontsize=12)
    ax.set_ylabel(label_name, fontsize=10)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def plot_grouped_bars(ax, summary_df, group_keys, group_labels, title, ylabel, model_names, model_colors):
    """
    Plot a grouped bar chart.
    - summary_df: DataFrame with index=groups (e.g., specific metrics) and columns= models.
    - group_keys: list of keys from the index to use as groups.
    - group_labels: labels for each group.
    """
    n_groups = len(group_keys)
    n_models = len(model_names)
    bar_width = 0.8 / n_models
    indices = np.arange(n_groups)
    
    for i, model in enumerate(model_names):
        offsets = indices - 0.4 + (i + 0.5) * bar_width
        values = []
        for key in group_keys:
            values.append(summary_df.loc[key][model] if key in summary_df.index else float('nan'))
        ax.bar(offsets, values, width=bar_width, color=model_colors.get(model, "gray"), edgecolor='black', label=model)
        for j, v in enumerate(values):
            if title in ['TP', 'FP', 'Detection Counts', 'Background Confusion', 'Missed Detection Counts']:
                ax.annotate(f'{v:.0f}', xy=(offsets[j], v),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
            else:
                ax.annotate(f'{v:.2f}', xy=(offsets[j], v),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
            
    
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(indices)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    max_val = np.nanmax([summary_df.loc[key][model] if key in summary_df.index else 0 for key in group_keys for model in model_names])
    ax.set_ylim(0, max_val * 1.2)

def plot_composite_for_category(cat_name, summary_df, model_names, model_colors, confusion_images):
    """
    Create a composite figure (returned as an image array) for the given category.
    For each category, only the specific metrics are used.
    """
    fig_comp = None
    if cat_name == "Overall Performance":
        fig_comp, (ax_group1, ax_group2) = plt.subplots(1, 2, figsize=(6, 3))
        # For precision-recall, use only AP and AR100 (displayed as AP and AR)
        pr_keys = ["AP", "AR100"]
        pr_labels = ["AP", "AR"]
        # For detection counts, use only voc_tp and voc_fp (displayed as TP and FP)
        dc_keys = ["voc_tp", "voc_fp"]
        dc_labels = ["TP", "FP"]
        plot_grouped_bars(ax_group1, summary_df, group_keys=pr_keys,
                          group_labels=pr_labels,
                          title="Precision-Recall Performance", ylabel="Score",
                          model_names=model_names, model_colors=model_colors)
        plot_grouped_bars(ax_group2, summary_df, group_keys=dc_keys,
                          group_labels=dc_labels,
                          title="Detection Counts", ylabel="Count",
                          model_names=model_names, model_colors=model_colors)
    elif cat_name == "Overall Performance - det":
        fig_comp, (ax_group1, ax_group2) = plt.subplots(1, 2, figsize=(6, 3))
        # For precision-recall, use only AP and AR100 (displayed as AP and AR)
        pr_keys = ["AP_det", "AR_det"]
        pr_labels = ["AP Det", "AR Det"]
        # For detection counts, use only voc_tp and voc_fp (displayed as TP and FP)
        f_keys = ["FN_det", "FP_det"]
        f_labels = ["FN Det", "FP Det"]

        plot_grouped_bars(ax_group1, summary_df, group_keys=pr_keys,
                        group_labels=pr_labels,
                        title="object vs. background", ylabel="Score",
                        model_names=model_names, model_colors=model_colors)
        plot_grouped_bars(ax_group2, summary_df, group_keys=f_keys,
                        group_labels=f_labels,
                        title="Missed Detection Counts", ylabel="Count",
                        model_names=model_names, model_colors=model_colors)
    elif cat_name == "Overall Performance - cls":
        fig_comp, (ax_group1, ax_group2) = plt.subplots(1, 2, figsize=(6, 3))
        # For precision-recall, use only AP and AR100 (displayed as AP and AR)
        pr_keys = ["AP_cls", "AR_cls"]
        pr_labels = ["AP Cls", "AR Cls"]
        # For detection counts, use only voc_tp and voc_fp (displayed as TP and FP)
        f_keys = ["FN_cls", "FP_cls"]
        f_labels = ["FN Cls", "FP Cls"]

        plot_grouped_bars(ax_group1, summary_df, group_keys=pr_keys,
                        group_labels=pr_labels,
                        title="cls", ylabel="Score",
                        model_names=model_names, model_colors=model_colors)
        plot_grouped_bars(ax_group2, summary_df, group_keys=f_keys,
                        group_labels=f_labels,
                        title="Missed Detection Counts", ylabel="Count",
                        model_names=model_names, model_colors=model_colors)
        
    elif cat_name == "BBox Localization":
        fig_comp, ax_comp = plt.subplots(figsize=(6, 3))
        loc_keys = ["AP50", "AP75"]
        loc_labels = ["AP50", "AP75"]
        plot_grouped_bars(ax_comp, summary_df, group_keys=loc_keys,
                          group_labels=loc_labels,
                          title="BBox Localization", ylabel="Score",
                          model_names=model_names, model_colors=model_colors)
    elif cat_name == "Scale-Specific Performance":
        fig_comp, (ax_ap, ax_ar) = plt.subplots(1, 2, figsize=(6, 3))
        ap_keys = ["APsmall", "APmedium", "APlarge"]
        ap_labels = ["Small", "Medium", "Large"]
        ar_keys = ["ARsmall", "ARmedium", "ARlarge"]
        ar_labels = ["Small", "Medium", "Large"]
        plot_grouped_bars(ax_ap, summary_df, group_keys=ap_keys,
                          group_labels=ap_labels,
                          title="AP by Scale", ylabel="AP",
                          model_names=model_names, model_colors=model_colors)
        plot_grouped_bars(ax_ar, summary_df, group_keys=ar_keys,
                          group_labels=ar_labels,
                          title="AR by Scale", ylabel="AR",
                          model_names=model_names, model_colors=model_colors)
    elif cat_name == "Runtime Performance" and "inference_speed" in summary_df.index:
        fig_comp, (ax_time, ax_fps) = plt.subplots(1, 2, figsize=(6, 3))
        plot_metric_bar(ax_time, summary_df, "inference_speed", model_names, model_colors)
        ax_time.set_title("Inference Time (ms)", fontsize=12)
        fps_values = {}
        for model in model_names:
            speed = summary_df.loc["inference_speed"].get(model, float('nan'))
            fps_values[model] = 1000 / speed if speed and speed != 0 else float('nan')
        ax_fps.set_title("FPS", fontsize=12)
        for i, model in enumerate(model_names):
            v = fps_values[model]
            ax_fps.bar(i, v, color=model_colors.get(model, "gray"), edgecolor="black")
            ax_fps.annotate(f'{v:.2f}', xy=(i, v), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom', fontsize=8)
        ax_fps.set_xticks(range(len(model_names)))
        ax_fps.set_xticklabels(model_names, rotation=45, fontsize=9)
        ax_fps.set_ylabel("FPS", fontsize=10)
        ax_fps.grid(axis='y', linestyle='--', alpha=0.7)
        max_val = summary_df.loc["inference_speed"].max()
        ax_time.set_ylim(0, max_val * 1.2)
        max_val = max(fps_values.values())
        ax_fps.set_ylim(0, max_val * 1.2)
    elif cat_name == "Object Confusion":
        n_metrics = ["confusion_matrix"]
        fig_comp, axes = plt.subplots(1, len(n_metrics), figsize=(10, 5))
        if len(n_metrics) == 1:
            axes = [axes]
        for idx, metric in enumerate(n_metrics):
            ax = axes[idx]
            if metric == "confusion_matrix":
                ax.set_title("Confusion Matrix", fontsize=12)
                n_models = len(model_names)
                fig_sub, sub_axes = plt.subplots(1, n_models, figsize=(3 * n_models, 3))
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
    buf_comp = io.BytesIO()
    if fig_comp is not None:
        fig_comp.tight_layout()
        fig_comp.savefig(buf_comp, format='png', dpi=600)
        buf_comp.seek(0)
        comp_img = plt.imread(buf_comp)
        buf_comp.close()
        plt.close(fig_comp)
        return comp_img
    else:
        return None

def load_metrics_for_subtest(weights_name, subtest, outputs_root, model_type, folder_type="subclass"):
    """
    Load metrics for an individual subtest.
    The expected folder structure is:
    outputs_root / <model_type> / <weights_name> / <subtest> / <folder_type> / results.csv
    and similarly for voc_results.csv.
    """
    folder_path = os.path.join(outputs_root, model_type, weights_name, subtest, folder_type)
    results_csv = os.path.join(folder_path, "results.csv")
    metrics_dict = {}
    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        metrics_dict.update({col: df_results.iloc[0][col] for col in df_results.columns})
    voc_csv = os.path.join(folder_path, "voc_results.csv")
    if os.path.exists(voc_csv):
        try:
            df_voc = pd.read_csv(voc_csv)
            df_voc['precision'] = pd.to_numeric(df_voc['precision'], errors='coerce')
            df_voc['recall'] = pd.to_numeric(df_voc['recall'], errors='coerce')
            voc_tp = df_voc['total TP'].sum()
            voc_fp = df_voc['total FP'].sum()
            metrics_dict['voc_tp'] = voc_tp
            metrics_dict['voc_fp'] = voc_fp
        except Exception as e:
            print(f"Warning: Failed to load voc metrics for {weights_name} in {subtest}: {e}")

    # Add confusion matrix image if it exists
    confusion_img = os.path.join(folder_path, "confusion_matrix_.png")
    if os.path.exists(confusion_img):
        metrics_dict["confusion_matrix"] = confusion_img

    return metrics_dict

def create_pdf_report(sub_summary_df, super_summary_df, args, sub_confusion_images, super_confusion_images, model_colors):
    model_names = args.weights
    comparison_models = "_".join(model_names)
    output_dir = os.path.join(args.output, comparison_models)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "comparison_summary.pdf")

    categories = [
        ("Overall Performance", {
            "explanation": (
                "AP: The area under the precision–recall curve averaged over multiple IoU threshs "
                "(0.50 to 0.95). ;AR: The average recall over multiple IoU thresholds. "
                ";TP: The total number of true positives based on VOC metrics. ;FP: The total number of false positives based on VOC metrics."
            )
        }),
        ("Overall Performance - det", {
            "explanation": (
                "AP_det: Average Precision for object detection (object vs. background discrimination). "
                ";AR_det: Average Recall for object detection, measuring the ability to find all objects. "
                ";FN_det: Number of ground truth objects that were not detected (missed detections). "
                ";FP_det: Number of predicted objects that do not correspond to any ground truth object."
            )
        }),
        ("Overall Performance - cls", {
            "explanation": (
                "AP_cls: Average Precision for classification among detected objects (excluding background confusion). "
                ";AR_cls: Average Recall for classification, measuring correct class assignment of detected objects. "
                ";FN_cls: Cases where objects were detected but assigned the wrong class. "
                ";FP_cls: Cases where detected objects were incorrectly classified."
            )
        }),
        ("Object Confusion", {
            "explanation": "Confusion matrix based on ground truth and detection bounding boxes."
        }),
        ("BBox Localization", {
            "explanation": (
                "AP50: Calculated at a fixed IoU threshold of 0.50. "
                ";AP75: Calculated at a stricter IoU threshold of 0.75. "
                ";A gap between AP50 and AP75 can highlight issues with localization accuracy."
            )
        }),
        ("Scale-Specific Performance", {
            "explanation": (
                "AP/AR computed separately for small, medium, and large objects (using area threshs). "
                ";Helps identify if your model is struggling with objects of a certain scale."
            )
        }),
        ("Runtime Performance", {
            "explanation": (
                "Inference Time: The average inference time (in ms) measured on the test set. ;"
                "FPS: Frames per second, computed as 1000 divided by the inference time."
            )
        })
    ]
    
    with PdfPages(output_file) as pdf:
        # Cover Page
        fig_cover = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        cover_title_1 = "Object Detection"
        cover_title_2 = "Model Comparison"
        subtitle = "Models: " + ", ".join(model_names)
        plt.text(0.5, 0.7, cover_title_1, fontsize=28, ha='center', va='center')
        plt.text(0.5, 0.6, cover_title_2, fontsize=28, ha='center', va='center')
        plt.text(0.5, 0.5, subtitle, fontsize=20, ha='center', va='center')
        pdf.savefig(fig_cover, bbox_inches='tight', dpi=600)
        plt.close(fig_cover)

        # Category Pages
        for cat_name, cat_data in categories:
            explanation = cat_data["explanation"]
            
            fig = plt.figure(figsize=(8.5, 11))
            if cat_name != "Runtime Performance":
                gs_outer = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[0.15, 0.35, 0.35, 0.15], figure=fig)
            else:
                gs_outer = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[0.15, 0.35, 0.15], figure=fig)
            
            # Title Section
            ax_title = fig.add_subplot(gs_outer[0])
            ax_title.axis('off')
            ax_title.text(0.5, 0.5, cat_name, fontsize=16, ha='center', va='center')
            
            # Subclass Composite Graph Section
            ax_sub = fig.add_subplot(gs_outer[1])
            sub_img = plot_composite_for_category(cat_name, sub_summary_df, model_names, get_model_colors(model_names), sub_confusion_images)
            if sub_img is not None:
                ax_sub.imshow(sub_img)
            ax_sub.axis('off')
            if cat_name != "Runtime Performance":
                ax_sub.set_title("Subclass", fontsize=12)
            
            # Superclass Composite Graph Section
            if cat_name != "Runtime Performance":
                ax_super = fig.add_subplot(gs_outer[2])
                super_img = plot_composite_for_category(cat_name, super_summary_df, model_names, get_model_colors(model_names), super_confusion_images)
                if super_img is not None:
                    ax_super.imshow(super_img)
                ax_super.axis('off')
                ax_super.set_title("Superclass", fontsize=12)
                i = 3
            else:
                i = 2

            if cat_name != "Object Confusion":
                handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=get_model_colors(model_names)[m],
                                        markersize=10, label=m) for m in model_names]
                ax_sub.legend(handles=handles, loc='lower center', ncol=len(model_names), bbox_to_anchor=(0.5, 1.1))

            # Explanation Section
            ax_expl = fig.add_subplot(gs_outer[i])
            ax_expl.axis('off')
            lines = [line.strip() for line in explanation.split(";")]
            wrapped_explanation = "\n".join(lines)
            ax_expl.text(0.5, 0.5, wrapped_explanation, fontsize=10, ha='center', va='center')
            pdf.savefig(fig, bbox_inches='tight', dpi=600)
            plt.close(fig)

        # Summary Table Page
        fig_summary, ax_summary = plt.subplots(figsize=(8.5, 11))
        ax_summary.axis('tight')
        ax_summary.axis('off')
        filtered_summary_df = sub_summary_df.drop(index=["voc_precision", "voc_recall"]).round(3)
        table = ax_summary.table(
            cellText=filtered_summary_df.values,
            rowLabels=filtered_summary_df.index,
            colLabels=filtered_summary_df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(0.6, 1.2)
        plt.title("Model Comparison Summary", fontsize=16, y=0.9)
        pdf.savefig(fig_summary, bbox_inches='tight', dpi=600)
        plt.close(fig_summary)
        
        # --- New Page: Platform-Specific Performance ---
        # This page shows overall performance (AP, AR, voc_tp, voc_fp) broken down by sub-dataset.
        # It will only use the subclass results from each individual subtest.
        subtests = ['test-AZIMUTHAIFA', 'test-VTS', 'test-YUVELPTZ', 'test-YUVELRGB', 'test-YUVELTHERMAL']
        metrics_of_interest = ["AP", "AR", "voc_tp", "voc_fp"]
        # Structure: {metric: {subtest: {model: value}}}
        platform_metrics = {m: {} for m in metrics_of_interest}
        for sub in subtests:
            for m in metrics_of_interest:
                platform_metrics[m][sub] = {}
            for i, weight in enumerate(model_names):
                try:
                    m_data = load_metrics_for_subtest(weight, sub, args.outputs_root, args.models[i], folder_type="subclass")
                except Exception as e:
                    m_data = {}
                for m in metrics_of_interest:
                    m_name = m if m != "AR" else "AR100"
                    platform_metrics[m][sub][weight] = m_data.get(m_name, float('nan'))
        
        # Convert each metric dictionary to a DataFrame.
        df_AP = pd.DataFrame(platform_metrics["AP"]).T
        df_AR = pd.DataFrame(platform_metrics["AR"]).T
        df_voc_tp = pd.DataFrame(platform_metrics["voc_tp"]).T
        df_voc_fp = pd.DataFrame(platform_metrics["voc_fp"]).T
        
        # Create a new figure with 2 columns and 2 rows.
        fig_platform, axes = plt.subplots(2, 2, figsize=(7, 6))
        # Left column: AP (top) and AR (bottom)
        plot_grouped_bars(axes[0,0], df_AP, group_keys=list(df_AP.index), group_labels=list(df_AP.index),
                          title="AP", ylabel="Score", model_names=model_names, model_colors=model_colors)
        plot_grouped_bars(axes[1,0], df_AR, group_keys=list(df_AR.index), group_labels=list(df_AR.index),
                          title="AR", ylabel="Score", model_names=model_names, model_colors=model_colors)
        # Right column: voc_tp (top) and voc_fp (bottom)
        plot_grouped_bars(axes[0,1], df_voc_tp, group_keys=list(df_voc_tp.index), group_labels=list(df_voc_tp.index),
                          title="TP", ylabel="Count", model_names=model_names, model_colors=model_colors)
        plot_grouped_bars(axes[1,1], df_voc_fp, group_keys=list(df_voc_fp.index), group_labels=list(df_voc_fp.index),
                          title="FP", ylabel="Count", model_names=model_names, model_colors=model_colors)
        for ax in axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=45)
        fig_platform.suptitle("Platform-Specific Performance", fontsize=16)
        fig_platform.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig_platform, bbox_inches='tight', dpi=600)
        plt.close(fig_platform)
    
        # --- New Page: Platform-Specific Confusion Matrices ---
        # This page will display the confusion matrix for each sub-dataset and each model.
        n_rows = len(subtests)
        n_cols = len(model_names)
        # Adjust figure size as needed; here each cell is roughly 2.5 x 2.5 inches.
        fig_conf, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
        
        # If there's only one row or one column, axes might not be a 2D array.
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for row, sub in enumerate(subtests):
            for col, weight in enumerate(model_names):
                # Use the corresponding model name from args.models for the folder structure.
                try:
                    m_data = load_metrics_for_subtest(weight, sub, args.outputs_root, args.models[col], folder_type="subclass")
                except Exception as e:
                    m_data = {}
                ax = axes[row][col]
                img_path = m_data.get("confusion_matrix", None)
                if img_path and os.path.exists(img_path):
                    img = plt.imread(img_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, f"No image\nfor {weight}\n({sub})", ha='center', va='center', fontsize=8)
                ax.axis('off')
                # Optionally, add a small label at the top of each cell
                ax.set_title(f"{sub}\n{weight}", fontsize=8)
        
        fig_conf.suptitle("Confusion Matrices", fontsize=16)
        fig_conf.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig_conf, bbox_inches='tight', dpi=600)
        plt.close(fig_conf)

    print(f"Comparison summary saved to {output_file}")

# --- Main Script ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare object detection models (subtests combined) and generate a summary PDF report."
    )
    parser.add_argument(
        '--weights', 
        nargs='+', 
        default=['Albatross-v0.4.1', 'Albatross-v0.4.3_1088,1920', 'Albatross-v0.4.3_1280', 'Albatross-v0.4.4_1088,1920', 'Albatross-v0.4.4_1280'],
        help="List of model weights (e.g., Albatross-v0.3 Albatross-v0.4)"
    )
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=['yolo11s', 'yolo12s', 'yolo12s', 'yolo11s', 'yolo11s'],
        help="List of model names (e.g., yolo11s detr-v0.1)"
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
    weights_names = args.weights
    subclass_metrics_list = []
    superclass_metrics_list = []
    for weights in weights_names:
        try:
            sub_metrics = load_metrics(weights, args.outputs_root, "subclass")
            super_metrics = load_metrics(weights, args.outputs_root, "superclass")
            subclass_metrics_list.append(sub_metrics)
            superclass_metrics_list.append(super_metrics)
        except Exception as e:
            print(f"Error loading metrics for {weights}: {e}")
            return
    
    # Create separate summary tables for subclass and superclass metrics.
    subclass_summary_df = create_summary_table(subclass_metrics_list, weights_names)
    superclass_summary_df = create_summary_table(superclass_metrics_list, weights_names)
    
    # Build confusion image dictionaries for each.
    sub_confusion_images = {m['weights_name']: m.get("confusion_matrix", None) for m in subclass_metrics_list}
    super_confusion_images = {m['weights_name']: m.get("confusion_matrix", None) for m in superclass_metrics_list}
    model_colors = get_model_colors(args.weights)
    
    create_pdf_report(subclass_summary_df, superclass_summary_df, args, sub_confusion_images, super_confusion_images, model_colors)

if __name__ == "__main__":
    main()
