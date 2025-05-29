import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# GRU Cell Definition
class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.update_gate = self._make_gate()
        self.reset_gate = self._make_gate()
        self.candidate_gate = self._make_gate()

    def _make_gate(self):
        return nn.ModuleDict({
            'Wx': nn.Linear(self.input_dim, self.hidden_dim),
            'Wh': nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        })

    def forward(self, x, h):
        outputs = []
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            z = torch.sigmoid(self.update_gate['Wx'](x_t) + self.update_gate['Wh'](h))
            r = torch.sigmoid(self.reset_gate['Wx'](x_t) + self.reset_gate['Wh'](h))
            h_tilde = torch.tanh(self.candidate_gate['Wx'](x_t) + self.candidate_gate['Wh'](r * h))
            h = (1 - z) * h + z * h_tilde
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# Multi-Layer GRU Model
class MultiLayerGRU(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(GRUCell(input_dim if i == 0 else hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, h):
        for i, layer in enumerate(self.layers):
            x = layer(x, h[i])
            x = self.dropout(x)
        out = self.fc(x)
        return out, x[:, -1, :].unsqueeze(0)
    
def preprocess(filepath, sequence_length=20, target_column='C_LAST'):
    # Read CSV with headers
    df = pd.read_csv(filepath, low_memory=False)

    # Strip whitespace from headers
    df.columns = df.columns.str.strip()

    # Check for column existence
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Columns are: {df.columns.tolist()}")

    # Force target column to numeric and drop rows with missing target
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    df = df.dropna(subset=[target_column])

    # Convert all numeric columns
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Drop any rows with all NaNs
    df_numeric.dropna(how='all', inplace=True)

    # Fill remaining NaNs with 0 (or optionally use other methods)
    df_numeric.fillna(0, inplace=True)

    # Separate features and target
    feature_columns = [col for col in df_numeric.columns if col != target_column]
    features = df_numeric[feature_columns].values
    target = df_numeric[target_column].values

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Pad or truncate to 32 features
    num_features = features_scaled.shape[1]
    if num_features < 32:
        pad_width = 32 - num_features
        features_scaled = np.pad(features_scaled, ((0, 0), (0, pad_width)), mode='constant')
    elif num_features > 32:
        features_scaled = features_scaled[:, :32]

    # Build sequences
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i+sequence_length])
        y.append(target[i+sequence_length])

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    return X_tensor, y_tensor, feature_columns


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, _ = preprocess('spy_2020_2022.csv', sequence_length=20)
    dataset = TensorDataset(X, y)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold + 1} ---")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)


        model = MultiLayerGRU(input_dim=32, hidden_dim=32, num_layers=2, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        loss_fn = nn.MSELoss()

        for epoch in range(10):
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Fold {fold+1}]", leave=False)
            for xb, yb in progress_bar:
                xb, yb = xb.to(device), yb.to(device)
                h0 = torch.zeros(2, xb.size(0), 32).to(device)
                pred, _ = model(xb, h0)
                final_output = pred[:, -1, 0]
                loss = loss_fn(final_output, yb.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            scheduler.step()
            print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader):.4f}")
            torch.save(model.state_dict(), f"model_fold{fold+1}_epoch{epoch+1}.pt")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                h0 = torch.zeros(2, xb.size(0), 32).to(device)
                pred, _ = model(xb, h0)
                final_output = pred[:, -1, 0]
                loss = loss_fn(final_output, yb.squeeze())
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")


main()
