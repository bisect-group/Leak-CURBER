import torch.nn.functional as F
from datetime import datetime
from scipy import spatial
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

def mahalanobis_distance_dataset(embeddings, centroids, inverse_cov_matrices, num_clusters):
    num_clusters = len(centroids)
    distances = []
    for point in embeddings:
        point_distances = []
        for i in range(num_clusters):
            centroid = centroids[i]
            distance = spatial.distance.mahalanobis(point, centroid, inverse_cov_matrices[i])
            point_distances.append(distance)
        distances.append(point_distances)
    return (np.array(distances)).reshape(embeddings.shape[0], -1)

def set_up_exp_folder(path):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    print('timestamp: ',timestamp)
    save_folder = path
    if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
    checkpoint_dir = '{}/exp{}/'.format(save_folder, timestamp)
    if os.path.exists(checkpoint_dir ) == False:
            os.mkdir(checkpoint_dir )
    return checkpoint_dir

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_and_split_data(data, args):
    md_columns = [f"{i}" for i in range(args.num_clusters)]

    X = data[md_columns].values
    if (args.include_y_pred_flag == 'True'):
        X_nn = data['y_pred'].values.reshape(-1, 1)
        X = np.concatenate((X, X_nn), axis=1)

    y = abs(data['y_true'] - data['y_pred'])
    return X, y

def pad_or_truncate_tensor(tensor, threshold = 500):
    """
    function to pad or truncate the tensors
    """
    x = tensor.size(0)
    if x < threshold:
        padding = torch.zeros((threshold - x, 1280))
        padded_tensor = torch.cat((tensor, padding), dim=0)
        return padded_tensor
    elif x > threshold:
        truncated_tensor = tensor[:threshold]
        return truncated_tensor
    else:
        return tensor

class SimpleMLP(nn.Module):
    def __init__(self, input_size=50):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x2 = self.fc3(x2)
        return x2

def evaluate_model(model, valid_loader, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item() * data.size(0)
    avg_loss = total_loss / len(valid_loader.dataset)
    return avg_loss

def create_df_results(pred, df, file_name, checkpoint_dir):
    df_results = pd.DataFrame({
        'predicted_residue': pred,
        'true_residue': abs(df['y_true'] - df['y_pred']),
        'y_true': df['y_true'],
        'y_pred': df['y_pred'],
        'SMILES': df['SMILES'],
        'uniprot|pfam': df['uniprot|pfam']
    })
    tolerance = 0.5
    df_results[f'Tolerance {tolerance}'] = np.where(df_results['predicted_residue'] < tolerance, 'Normal', 'Outlier')
    file_path = os.path.join(checkpoint_dir, f'residues_values_analysis_{file_name}.csv')
    df_results.to_csv(file_path)

def evaluate_metrics(pred, df, file_name, seed, checkpoint_dir):

    df_results = pd.DataFrame({
        'Predicted Residue': pred,
        'True Residue': abs(df['y_true'] - df['y_pred']),
        'y_true': df['y_true'],
        'y_pred': df['y_pred'],
        'SMILES': df['SMILES'],
        'uniprot|pfam': df['uniprot|pfam']
    })

    tolerance = 0.5
    df_results[f'Tolerance {tolerance}'] = np.where(df_results['Predicted Residue'] < tolerance, 'Normal', 'Outlier')

    pearson_corr_all, _ = pearsonr(df_results['y_pred'], df_results['y_true'])
    spearman_corr_all, _ = spearmanr(df_results['y_pred'], df_results['y_true'])
    rmse_all = mean_squared_error(df_results['y_true'], df_results['y_pred'], squared=False)
    mae_all = mean_absolute_error(df_results['y_true'], df_results['y_pred'])

    filtered_df = df_results[df_results['Predicted Residue'] <= tolerance]
    if not filtered_df.empty:
        pearson_corr_filtered, _ = pearsonr(filtered_df['y_pred'], filtered_df['y_true'])
        spearman_corr_filtered, _ = spearmanr(filtered_df['y_pred'], filtered_df['y_true'])
        rmse_filtered = mean_squared_error(filtered_df['y_true'], filtered_df['y_pred'], squared=False)
        mae_filtered = mean_absolute_error(filtered_df['y_true'], filtered_df['y_pred'])
    else:
        pearson_corr_filtered = np.nan
        spearman_corr_filtered = np.nan
        rmse_filtered = np.nan
        mae_filtered = np.nan

    metrics = {
        'Seed': seed,
        'Dataset': file_name,
        'PearsonR_TrustAffinity': pearson_corr_all,
        'SpearmanR_TrustAffinity': spearman_corr_all,
        'RMSE_TrustAffinity': rmse_all,
        'MAE_TrustAffinity': mae_all,
        'PearsonR_eMOSAIC': pearson_corr_filtered,
        'SpearmanR_eMOSAIC': spearman_corr_filtered,
        'RMSE_eMOSAIC': rmse_filtered,
        'MAE_eMOSAIC': mae_filtered
    }

    result_file = os.path.join(checkpoint_dir, f"metrics_summary_{file_name}_seed_{seed}.csv")
    pd.DataFrame([metrics]).to_csv(result_file, index=False)

    return metrics