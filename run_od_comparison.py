
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
            # We assume the voc_results.csv contains columns named 'precision' and 'recall'
            # It might contain one row per class; we compute the mean value.
            # (If your CSV has a different structure, adjust accordingly.)
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

    # Include additional info (e.g., model type) if needed.
    metrics['model_type'] = model_type
    metrics['weights_name'] = weights_name
    return metrics

def create_summary_table(metrics_list, model_names):
    """
    Given a list of metrics dictionaries (one per model) and their corresponding names,
    produce a summary DataFrame where rows are metric names and columns are models.
    """
    # Combine all keys; then for each model fill in the value (or NaN if missing)
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    # Remove non-numeric keys if desired
    all_keys.discard('model_type')
    all_keys.discard('weights_name')
    
    summary = {}
    for key in sorted(all_keys):
        summary[key] = []
        for m in metrics_list:
            summary[key].append(m.get(key, float('nan')))
    summary_df = pd.DataFrame(summary, index=model_names)
    return summary_df.T  # so that rows = metrics, columns = models

def plot_metric_comparison(summary_df, metric_name, ax):
    """
    Plot a bar chart comparing a specific metric across models.
    """
    if metric_name not in summary_df.index:
        ax.text(0.5, 0.5, f"Metric {metric_name} not found", ha='center')
        return
    values = summary_df.loc[metric_name]
    values.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title(f"Comparison of {metric_name}")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Models")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def generate_graphs(summary_df, model_names, pdf):
    """
    Generate and save plots for a predefined list of key metrics.
    """
    # Define the list of key metrics to plot.
    # Adjust as necessary to include the metrics you consider most informative.
    key_metrics = ['AP', 'AP50', 'AP75', 'voc_precision', 'voc_recall']
    for metric in key_metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_metric_comparison(summary_df, metric, ax)
        # Add a brief explanation (customize as desired)
        explanation = ""
        if metric == 'AP':
            explanation = "Higher AP indicates overall better detection performance."
        elif metric == 'AP50':
            explanation = "AP@0.5 reflects detection performance with relaxed IoU threshold."
        elif metric == 'AP75':
            explanation = "AP@0.75 shows localization quality at a stricter IoU threshold."
        elif metric == 'voc_precision':
            explanation = "Higher VOC precision suggests fewer false positives."
        elif metric == 'voc_recall':
            explanation = "Higher VOC recall suggests fewer missed objects."
        ax.text(0.5, -0.15, explanation, transform=ax.transAxes,
                fontsize=9, ha='center', va='top')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def generate_summary_text(summary_df, model_names):
    """
    Create a text summary (in markdown format) based on the summary table.
    """
    lines = []
    lines.append("## Model Comparison Summary")
    lines.append("### Models Compared: " + ", ".join(model_names))
    lines.append("#### Key Metrics Overview:")
    lines.append(summary_df.to_markdown())
    # Sample observations (customize these observations as needed)
    if 'AP' in summary_df.index and not summary_df.loc['AP'].isnull().all():
        best_AP = summary_df.loc['AP'].idxmax()
        lines.append(f"* **Observation:** {best_AP} has the highest overall AP, indicating the best overall detection performance.")
    if 'voc_recall' in summary_df.index and not summary_df.loc['voc_recall'].isnull().all():
        best_recall = summary_df.loc['voc_recall'].idxmax()
        lines.append(f"* **Observation:** {best_recall} exhibits the highest VOC recall, suggesting it misses fewer objects.")
    summary_text = "\n".join(lines)
    return summary_text

def create_pdf_report(summary_df, args):
    """
    Create a multi-page PDF report with a cover, summary text, graphs, and conclusion.
    """
    model_names = args.models
    comparison_models = "_".join(model_names)
    output_dir = os.path.join(args.output, comparison_models)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "comparison_summary.pdf")

    with PdfPages(output_file) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        title = "Model Comparison Summary"
        subtitle = " vs. ".join(model_names)
        cover_text = f"{title}\n\nModels Compared:\n{subtitle}"
        ax.text(0.5, 0.5, cover_text, transform=ax.transAxes, fontsize=20,
                ha='center', va='center')
        pdf.savefig(fig)
        plt.close(fig)
        
        # Summary text page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        summary_text = generate_summary_text(summary_df, model_names)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, va='top')
        pdf.savefig(fig)
        plt.close(fig)
        
        # Graphs for each metric
        generate_graphs(summary_df, model_names, pdf)
        
        # Conclusion page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        conclusion_text = ("### Conclusion\n\n"
                           "Based on the comparison, each model shows its own trade-offs. "
                           "For example, while one model might achieve the highest AP and AP50, "
                           "another may excel in VOC recall. Further analysis and additional tests "
                           "could help in selecting the optimal model for a given scenario.")
        ax.text(0.05, 0.95, conclusion_text, transform=ax.transAxes,
                fontsize=10, va='top')
        pdf.savefig(fig)
        plt.close(fig)

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
        help="Output PDF file name for the summary report."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    weights_names = args.models  # e.g., ['Albatross-v0.3', 'detr-v0.1']
    
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
    print(f"Comparison summary saved to {args.output}")

if __name__ == "__main__":
    main()
