import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
import seaborn as sns

sns.set_context("paper")
sns.set(font='serif')
sns.set_style("white", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})

# Define the custom color palette for the models with eMOSAIC first
custom_palette = {"TrustAffinity": "#dd8452", "eMOSAIC": "#4c72b0"}

save_fig_dir = "/results/figures/"
os.makedirs(save_fig_dir, exist_ok=True)

def run_anomaly_detection(seeds):
    results = []
    for seed in seeds:
        command = f"python -u AnomalyDetection/anomaly_detection.py --num_clusters=50 --batch_size=256 --epochs=50 --scaling=True --data_split=scaffold --seed={seed}"
        subprocess.run(command, shell=True)
        
        result_file = f"/results/metrics_summary_test_seed_{seed}.csv"
        metrics_df = pd.read_csv(result_file)
        results.append(metrics_df)
        
    return pd.concat(results, ignore_index=True)

def aggregate_and_visualize(seeds):
    all_results = run_anomaly_detection(seeds)

    test_results = all_results[all_results['Dataset'] == 'test']

    melted_results = pd.melt(test_results, id_vars=['Seed', 'Dataset'], 
                             value_vars=['PearsonR_TrustAffinity', 'PearsonR_eMOSAIC',
                                         'SpearmanR_TrustAffinity', 'SpearmanR_eMOSAIC',
                                         'RMSE_TrustAffinity', 'RMSE_eMOSAIC',
                                         'MAE_TrustAffinity', 'MAE_eMOSAIC'],
                             var_name='Metric_Type', value_name='Metric_Value')

    melted_results[['Metric', 'Model']] = melted_results['Metric_Type'].str.split('_', expand=True)

    metric_name_mapping = {
        'PearsonR': 'Pearson Correlation',
        'SpearmanR': 'Spearman Correlation',
        'RMSE': 'RMSE',
        'MAE': 'MAE'
    }
    melted_results['Metric'] = melted_results['Metric'].map(metric_name_mapping)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Model Performance Metrics for Test Set', fontsize=16)

    metrics = ['RMSE', 'MAE', 'Pearson Correlation', 'Spearman Correlation']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for metric, pos in zip(metrics, positions):
        ax = axes[pos]
        data_to_plot = melted_results[melted_results['Metric'] == metric]

        # Ensure eMOSAIC appears first, followed by TrustAffinity
        grouped_data = data_to_plot.groupby('Model')['Metric_Value'].agg(['mean', 'std']).reset_index()
        grouped_data = grouped_data.set_index('Model').reindex(['eMOSAIC', 'TrustAffinity']).reset_index()

        models = grouped_data['Model']
        means = grouped_data['mean']
        stds = grouped_data['std']

        # Define positions and bar width
        bar_width = 0.4
        spacing = 0.07
        x_center = 0
        offsets = [-bar_width/2 - spacing/2, bar_width/2 + spacing/2]
        bar_positions = [x_center + offset for offset in offsets]

        # Create bars for the reordered models (eMOSAIC first, TrustAffinity second)
        bars = ax.bar(bar_positions, means, width=bar_width, yerr=stds, capsize=5,
                      color=[custom_palette[model] for model in models])

        ax.set_xticks([x_center])
        ax.set_xticklabels([''])

        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.grid(False)

    legend_elements = [Patch(facecolor=color, label=model) for model, color in custom_palette.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, title='Model', fontsize='large')

    plt.tight_layout(rect=[0, 0.075, 1, 0.95])
    figure_path = os.path.join(save_fig_dir, "trustaffinity_vs_emosaic_test_metrics.png")
    plt.savefig(figure_path, dpi=300)
    print(f"Figure saved to {figure_path}")
    plt.show()

    summary = test_results.groupby('Dataset').agg({col: ['mean', 'std'] for col in test_results.columns if col not in ['Seed', 'Dataset']}).reset_index()
    summary.columns = [' '.join(col).strip() for col in summary.columns.values]
    
    reshaped_summary = pd.DataFrame({
        'Metric': ['Pearson Correlation', 'Spearman Correlation', 'RMSE', 'MAE'],
        'TrustAffinity (mean ± std)': [
            f"{summary['PearsonR_TrustAffinity mean'].values[0]:.4f} ± {summary['PearsonR_TrustAffinity std'].values[0]:.4f}",
            f"{summary['SpearmanR_TrustAffinity mean'].values[0]:.4f} ± {summary['SpearmanR_TrustAffinity std'].values[0]:.4f}",
            f"{summary['RMSE_TrustAffinity mean'].values[0]:.4f} ± {summary['RMSE_TrustAffinity std'].values[0]:.4f}",
            f"{summary['MAE_TrustAffinity mean'].values[0]:.4f} ± {summary['MAE_TrustAffinity std'].values[0]:.4f}"
        ],
        'eMOSAIC (mean ± std)': [
            f"{summary['PearsonR_eMOSAIC mean'].values[0]:.4f} ± {summary['PearsonR_eMOSAIC std'].values[0]:.4f}",
            f"{summary['SpearmanR_eMOSAIC mean'].values[0]:.4f} ± {summary['SpearmanR_eMOSAIC std'].values[0]:.4f}",
            f"{summary['RMSE_eMOSAIC mean'].values[0]:.4f} ± {summary['RMSE_eMOSAIC std'].values[0]:.4f}",
            f"{summary['MAE_eMOSAIC mean'].values[0]:.4f} ± {summary['MAE_eMOSAIC std'].values[0]:.4f}"
        ]
    })

    print("\nFormatted Test Set Metrics Table (mean ± std):\n")
    col_widths = [max(len(str(value)) for value in reshaped_summary[col]) for col in reshaped_summary.columns]
    header = " | ".join(f"{col:<{col_widths[idx]}}" for idx, col in enumerate(reshaped_summary.columns))
    separator = "-+-".join('-' * col_widths[idx] for idx, col in enumerate(reshaped_summary.columns))
    print(header)
    print(separator)
    for _, row in reshaped_summary.iterrows():
        print(" | ".join(f"{str(value):<{col_widths[idx]}}" for idx, value in enumerate(row)))

if __name__ == "__main__":
    seeds = [42, 0, 11]
    aggregate_and_visualize(seeds)
