import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import matplotlib.gridspec as gridspec

# --- Helper Functions ---

def find_model_subfolder(weights_name, outputs_root):
    """
    Search inside each model folder in outputs_root for a folder that contains
    the specified weights_name and its 'subclass' subfolder.
    Returns: (subclass_path, model_type)
    """
    for model_type in os.listdir(outputs_root):
        model_type_path = os.path.join(outputs_root, model_type)
        if not os.path.isdir(model_type_path):
            continue
        candidate = os.path.join(model_type_path, weights_name, "subclass")
        if os.path.exists(candidate):
            return candidate, model_type
    raise FileNotFoundError(f"Could not find metrics for weights '{weights_name}' in {outputs_root}")

def load_model_metrics(weights_name, outputs_root):
    """
    Given a weights_name and the outputs_root, find the corresponding subclass folder
    and load the two CSV files: results.csv and voc_results.csv.
    Returns: a dict of metrics (including keys from results.csv and aggregated voc_results).
    """
    subclass_path, model_type = find_model_subfolder(weights_name, outputs_root)
    
    results_csv = os.path.join(subclass_path, "results.csv")
    voc_csv = os.path.join(subclass_path, "voc_results.csv")
    
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results CSV not found at {results_csv}")
    
    # Load overall metrics from results.csv (assume a single-row CSV)
    df_results = pd.read_csv(results_csv)
    metrics = {}
    for col in df_results.columns:
        metrics[col] = df_results.iloc[0][col]
    
    # Attempt to load voc_results.csv and compute mean precision/recall
    if os.path.exists(voc_csv):
        try:
            df_voc = pd.read_csv(voc_csv)
            df_voc['precision'] = pd.to_numeric(df_voc['precision'], errors='coerce')
            df_voc['recall'] = pd.to_numeric(df_voc['recall'], errors='coerce')
            voc_precision = df_voc['precision'].mean()
            voc_recall = df_voc['recall'].mean()
        except Exception as e:
            print(f"Warning: Failed to load/parse {voc_csv}: {e}")
            voc_precision, voc_recall = None, None
    else:
        voc_precision, voc_recall = None, None

    if voc_precision is not None:
        metrics['voc_precision'] = voc_precision
    if voc_recall is not None:
        metrics['voc_recall'] = voc_recall

    # Include additional info (e.g., model type)
    metrics['model_type'] = model_type
    metrics['weights_name'] = weights_name
    return metrics

def create_summary_table(metrics_list, model_names):
    """
    Given a list of metrics dictionaries (one per model) and their corresponding names,
    produce a summary DataFrame where rows are metric names and columns are models.
    """
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    # Remove non-numeric keys if desired.
    all_keys.discard('model_type')
    all_keys.discard('weights_name')
    
    summary = {}
    for key in sorted(all_keys):
        summary[key] = []
        for m in metrics_list:
            summary[key].append(m.get(key, float('nan')))
    summary_df = pd.DataFrame(summary, index=model_names)
    return summary_df.T  # rows = metrics, columns = models

def plot_metric_bar(summary_df, metric_name, ax):
    """
    Plot a bar chart comparing a specific metric across models.
    Adjusts fonts and spacing so labels do not overlap.
    """
    if metric_name not in summary_df.index:
        ax.text(0.5, 0.5, f"Metric {metric_name} not found", ha='center', fontsize=10)
        return
    values = summary_df.loc[metric_name]
    bars = ax.bar(range(len(values)), values, color='steelblue', edgecolor='black')
    ax.set_title(metric_name, fontsize=12)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(list(values.index), rotation=45, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Annotate each bar with its value.
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# --- PDF Report Generation Functions ---

def create_pdf_report(summary_df, args):
    """
    Create a multi-page PDF report with a cover, summary text, and category pages.
    Each category page shows (from top to bottom):
      1. A large title (the category name),
      2. A graph comparing the relevant metric,
      3. A small explanation of the metric.
    """
    model_names = args.models
    comparison_models = "_".join(model_names)
    output_dir = os.path.join(args.output, comparison_models)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "comparison_summary.pdf")

    # Define detection categories and assign relevant metrics and explanation text.
    # Here we assume one metric per category.
    categories = {
        "Missed Objects": {
            "metrics": ["voc_recall"],
            "explanation": ("This metric measures the detector’s ability to capture all objects. "
                            "Higher values indicate fewer missed detections.")
        },
        "Object Confusion": {
            "metrics": ["voc_precision"],
            "explanation": ("This metric indicates how often the detector confuses background or other objects "
                            "with the target. Higher precision means fewer false positives.")
        },
        "Bounding Box Localization Quality": {
            "metrics": ["AP75"],
            "explanation": ("This metric assesses the accuracy of the predicted bounding boxes at a strict IoU threshold (0.75). "
                            "Higher values indicate better localization.")
        }
    }
    
    with PdfPages(output_file) as pdf:
        # --- Cover Page ---
        fig_cover = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        cover_title = "Object Detection Model Comparison"
        subtitle = "Models: " + ", ".join(model_names)
        plt.text(0.5, 0.6, cover_title, fontsize=28, ha='center', va='center')
        plt.text(0.5, 0.5, subtitle, fontsize=20, ha='center', va='center')
        pdf.savefig(fig_cover, bbox_inches='tight')
        plt.close(fig_cover)

        # --- Summary Text Page ---
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
        pdf.savefig(fig_summary, bbox_inches='tight')
        plt.close(fig_summary)

        # --- Category Pages ---
        for cat_name, cat_data in categories.items():
            metric = cat_data["metrics"][0]  # assuming one metric per category
            explanation = cat_data["explanation"]

            # Create a figure with 3 vertical sections:
            # Row 0: Title (Large)
            # Row 1: Graph
            # Row 2: Explanation (Small)
            fig = plt.figure(figsize=(8.5, 11))
            gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[0.2, 0.55, 0.25], figure=fig)
            
            # Title section
            ax_title = fig.add_subplot(gs[0])
            ax_title.axis('off')
            ax_title.text(0.5, 0.5, cat_name, fontsize=15, ha='center', va='center')
            
            # Graph section
            ax_graph = fig.add_subplot(gs[1])
            plot_metric_bar(summary_df, metric, ax_graph)
            
            # Explanation section
            ax_expl = fig.add_subplot(gs[2])
            ax_expl.axis('off')
            wrapped_explanation = "\n".join(textwrap.wrap(explanation, width=90))
            ax_expl.text(0.5, 0.5, wrapped_explanation, fontsize=10, ha='center', va='center')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # --- Conclusion Page ---
        fig_concl = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        conclusion = (
            "### Conclusion\n\n"
            "The comparison shows that each model exhibits distinct trade-offs. For instance, while one model "
            "may achieve high recall (fewer missed objects), another may offer higher precision (fewer false positives).\n\n"
            "Furthermore, differences in bounding box localization (AP75) highlight variations in the models' "
            "ability to precisely localize objects. Additional domain-specific testing is recommended to determine "
            "the optimal model for a given application."
        )
        wrapped_conclusion = "\n".join(textwrap.wrap(conclusion, width=90))
        plt.text(0.05, 0.95, wrapped_conclusion, fontsize=12, va='top')
        pdf.savefig(fig_concl, bbox_inches='tight')
        plt.close(fig_concl)
    
    print(f"Comparison summary saved to {output_file}")

# --- Main Script ---

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
    weights_names = args.models  # e.g. ['Albatross-v0.3', 'detr-v0.1']
    
    # Load metrics for each model weight.
    metrics_list = []
    for weights in weights_names:
        try:
            metrics = load_model_metrics(weights, args.outputs_root)
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Error loading metrics for {weights}: {e}")
            return
    
    # Create a summary table DataFrame.
    summary_df = create_summary_table(metrics_list, weights_names)
    
    # Create and save the PDF report.
    create_pdf_report(summary_df, args)

if __name__ == "__main__":
    main()
