from torch_geometric.data import Data
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, X, y, edge_index): # X: [N, 30days, 30nodes, F], y: [N, 30nodes, 7days]
        self.X, self.y = X, y
        self.edge_index = edge_index

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        graphs = []
        for t in range(self.X.shape[1]): # 30 days
            x_t = self.X[idx, t] # [30 nodes, F]
            graphs.append(Data(x=x_t, edge_index=self.edge_index))
        target = self.y[idx] # [30 nodes, 7]
        return graphs, target # X [30 days, 30 nodes, F], y [30 nodes, 7 days]


class DynamicWindowDataset(Dataset):
    def __init__(self, X, y, edge_indices): # edge_indices: list of 30 edge_index tensors
        self.X, self.y = X, y
        self.edge_indices = edge_indices  # List of refined edge_index for each time step

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        graphs = []
        for t in range(self.X.shape[1]): # 30 days
            x_t = self.X[idx, t] # [30 nodes, F]
            # Use time-specific edge_index
            edge_index_t = self.edge_indices[t] if self.edge_indices and t < len(self.edge_indices) else self.edge_indices[0]
            graphs.append(Data(x=x_t, edge_index=edge_index_t))
        target = self.y[idx] # [30 nodes, 7]
        return graphs, target # X [30 days, 30 nodes, F], y [30 nodes, 7 days]


def inverse_transform_predictions(predictions, targets, scaler_data, encode_map=None):
    """
    Unified inverse transform function for both global and local scalers.
    
    Args:
        predictions (np.ndarray): Predictions array with shape (samples, nodes, 7)
        targets (np.ndarray): Targets array with shape (samples, nodes, 7)
        scaler_data (dict): Dictionary containing scaler information
        encode_map (dict): Optional mapping from reservoir name to node index
    
    Returns:
        tuple: (predictions_original, targets_original) - inverse transformed arrays
    """
    import numpy as np
    
    n_samples, n_nodes, n_days = predictions.shape
    scaler_type = scaler_data.get('params', {}).get('scaler_type', 'global')
    
    if scaler_type == 'global':
        # Use global scaler
        scaler_y = scaler_data['scaler_y']
        
        pred_reshaped = predictions.reshape(-1, 1)
        target_reshaped = targets.reshape(-1, 1)
        
        pred_inversed = scaler_y.inverse_transform(pred_reshaped)
        target_inversed = scaler_y.inverse_transform(target_reshaped)
        
        predictions_original = pred_inversed.reshape(n_samples, n_nodes, n_days)
        targets_original = target_inversed.reshape(n_samples, n_nodes, n_days)
        
    elif scaler_type == 'local':
        # Use local scalers
        local_scalers_y = scaler_data['local_scalers_y']
        
        # Create reverse mapping from node index to reservoir name
        idx_to_reservoir = {}
        if encode_map:
            for reservoir_name, node_idx in encode_map.items():
                idx_to_reservoir[node_idx] = reservoir_name
        
        predictions_original = np.zeros_like(predictions)
        targets_original = np.zeros_like(targets)
        
        for node_idx in range(n_nodes):
            # Get reservoir name using the reverse mapping
            reservoir_name = f"Node_{node_idx}"
            if node_idx in idx_to_reservoir:
                reservoir_name = idx_to_reservoir[node_idx]
            
            # Get the corresponding scaler for this reservoir
            if reservoir_name in local_scalers_y:
                scaler_y = local_scalers_y[reservoir_name]
                
                # Extract predictions and targets for this reservoir
                node_predictions = predictions[:, node_idx, :]  # Shape: (samples, 7)
                node_targets = targets[:, node_idx, :]         # Shape: (samples, 7)
                
                # Reshape for scaler
                pred_reshaped = node_predictions.reshape(-1, 1)
                target_reshaped = node_targets.reshape(-1, 1)
                
                # Inverse transform
                pred_inversed = scaler_y.inverse_transform(pred_reshaped)
                target_inversed = scaler_y.inverse_transform(target_reshaped)
                
                # Reshape back and store
                predictions_original[:, node_idx, :] = pred_inversed.reshape(n_samples, n_days)
                targets_original[:, node_idx, :] = target_inversed.reshape(n_samples, n_days)
            else:
                print(f"Warning: No scaler found for reservoir {reservoir_name}")
                print(f"Available scalers: {list(local_scalers_y.keys())}")
                # Keep original scaled values as fallback
                predictions_original[:, node_idx, :] = predictions[:, node_idx, :]
                targets_original[:, node_idx, :] = targets[:, node_idx, :]
    
    else:
        raise ValueError(f"Invalid scaler_type: {scaler_type}. Must be 'global' or 'local'.")
    
    return predictions_original, targets_original


def load_preprocessed_data(data_path, scaler_type="global"):
    """
    Helper function to load preprocessed data for a specific scaler type.
    
    Args:
        data_path (str): Path to the data directory
        scaler_type (str): Type of scaler to load ("global" or "local")
    
    Returns:
        tuple: (scaler_data, supervised_data) containing the loaded data
    """
    import os
    import pickle
    import torch
    
    parsed_path = os.path.join(data_path, 'parsed')
    
    # Load scaler data
    all_rsr_data_file = os.path.join(parsed_path, f"all_rsr_data_{scaler_type}.pkl")
    if not os.path.exists(all_rsr_data_file):
        raise FileNotFoundError(f"Preprocessed data not found: {all_rsr_data_file}. Please run _preprocess.py first.")
    
    with open(all_rsr_data_file, 'rb') as f:
        all_rsr_data = pickle.load(f)
    
    scaler_data = {
        'scaler_X': all_rsr_data.get('scaler_X'),
        'scaler_y': all_rsr_data.get('scaler_y'),
        'local_scalers_X': all_rsr_data.get('local_scalers_X'),
        'local_scalers_y': all_rsr_data.get('local_scalers_y'),
        'params': all_rsr_data.get('params', {})
    }
    
    # Load supervised data
    supervised_file = os.path.join(parsed_path, f"_GNN_supervise_{scaler_type}.pt")
    if not os.path.exists(supervised_file):
        raise FileNotFoundError(f"Supervised data not found: {supervised_file}. Please run _preprocess.py first.")
    
    supervised_data = torch.load(supervised_file, weights_only=True)
    
    print(f"Successfully loaded {scaler_type} scaler data:")
    print(f"  - Scaler data from: {all_rsr_data_file}")
    print(f"  - Supervised data from: {supervised_file}")
    print(f"  - Data shapes: X_train {supervised_data['X_train'].shape}, y_train {supervised_data['y_train'].shape}")
    
    return scaler_data, supervised_data


def check_available_data_files(data_path):
    """
    Check which scaler type data files are available.
    
    Args:
        data_path (str): Path to the data directory
        
    Returns:
        dict: Dictionary indicating which files are available
    """
    import os
    
    parsed_path = os.path.join(data_path, 'parsed')
    available = {}
    
    for scaler_type in ["global", "local"]:
        scaler_file = os.path.join(parsed_path, f"all_rsr_data_{scaler_type}.pkl")
        supervised_file = os.path.join(parsed_path, f"_GNN_supervise_{scaler_type}.pt")
        
        available[scaler_type] = {
            'scaler_data': os.path.exists(scaler_file),
            'supervised_data': os.path.exists(supervised_file),
            'complete': os.path.exists(scaler_file) and os.path.exists(supervised_file)
        }
    
    return available



