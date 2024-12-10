import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import time
from collections import deque
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import torch.nn.init as init

# configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# plots directory
os.makedirs('plots', exist_ok=True)

# focal Loss
class FocalLoss(nn.Module):
   def __init__(self, alpha=0.25, gamma=2.0):
       super().__init__()
       self.alpha = alpha
       self.gamma = gamma
   
   def forward(self, predictions, targets):
       bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
       pt = torch.exp(-bce_loss)
       focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
       return focal_loss.mean()

class ResidualBlock(nn.Module):
   def __init__(self, channels):
       super().__init__()
       self.conv = nn.Sequential(
           nn.Linear(channels, channels),
           nn.BatchNorm1d(channels),
           nn.ReLU(),
           nn.Linear(channels, channels),
           nn.BatchNorm1d(channels)
       )
       
   def forward(self, x):
       return F.relu(x + self.conv(x))

class CANDataset(Dataset):
   def __init__(self, X, y):
       self.X = torch.FloatTensor(X)
       self.y = torch.FloatTensor(y)
       
   def __len__(self):
       return len(self.X)
   
   def __getitem__(self, idx):
       return self.X[idx], self.y[idx]

class AttentionLayer(nn.Module):
   def __init__(self, hidden_dim):
       super().__init__()
       self.attention = nn.Sequential(
           nn.Linear(hidden_dim, hidden_dim),
           nn.Tanh(),
           nn.Linear(hidden_dim, 1),
           nn.Softmax(dim=1)
       )
       self._init_weights()
   
   def _init_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Linear):
               init.xavier_uniform_(m.weight)
               if m.bias is not None:
                   init.zeros_(m.bias)
   
   def forward(self, x):
       weights = self.attention(x)
       return torch.sum(weights * x, dim=1)

class CANIDS(nn.Module):
   def __init__(self, input_dim, hidden_dim=128):
       super().__init__()
       self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, 
                          batch_first=True, bidirectional=True, dropout=0.4)
       self.attention = AttentionLayer(hidden_dim * 2)
       
       self.fc = nn.Sequential(
           nn.Linear(hidden_dim * 2, hidden_dim),
           nn.BatchNorm1d(hidden_dim),
           ResidualBlock(hidden_dim),
           nn.Dropout(0.4),
           nn.Linear(hidden_dim, 64),
           nn.BatchNorm1d(64),
           ResidualBlock(64),
           nn.Dropout(0.4),
           nn.Linear(64, 32),
           nn.BatchNorm1d(32),
           ResidualBlock(32),
           nn.Linear(32, 1),
           nn.Sigmoid()
       )
       self._init_weights()
   
   def _init_weights(self):
       for name, param in self.lstm.named_parameters():
           if 'weight' in name:
               init.xavier_uniform_(param)
           elif 'bias' in name:
               init.zeros_(param)
       
       for m in self.fc.modules():
           if isinstance(m, nn.Linear):
               init.xavier_uniform_(m.weight)
               if m.bias is not None:
                   init.zeros_(m.bias)
   
   def forward(self, x):
       lstm_out, _ = self.lstm(x)
       attended = self.attention(lstm_out)
       return self.fc(attended)

def save_sample_data(df, filename='sample_data.json'):
   sample_data = {
       'normal': df[df['Attack_Type'] == 'Normal'].sample(n=100).to_dict('records'),
       'dos': df[df['Attack_Type'] == 'DoS'].sample(n=100).to_dict('records'),
       'fuzzy': df[df['Attack_Type'] == 'Fuzzy'].sample(n=100).to_dict('records'),
       'gear': df[df['Attack_Type'] == 'Gear'].sample(n=100).to_dict('records'),
       'rpm': df[df['Attack_Type'] == 'RPM'].sample(n=100).to_dict('records')
   }
   with open(filename, 'w') as f:
       json.dump(sample_data, f)

def plot_training_history(train_losses, val_losses, metrics_history, save_dir='plots'):
   os.makedirs(save_dir, exist_ok=True)
   
   # loss plot
   plt.figure(figsize=(10, 6))
   plt.plot(train_losses, label='Training Loss', marker='o')
   plt.plot(val_losses, label='Validation Loss', marker='o')
   plt.title('Training and Validation Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   plt.grid(True)
   plt.savefig(f'{save_dir}/loss_curve.png')
   plt.close()
   
   # Accuracy plot
   plt.figure(figsize=(10, 6))
   plt.plot(metrics_history['accuracy'], label='Accuracy', marker='o')
   plt.title('Model Accuracy over Epochs')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.grid(True)
   plt.savefig(f'{save_dir}/accuracy.png')
   plt.close()
   
   # metrics plot
   plt.figure(figsize=(10, 6))
   for metric in ['precision', 'recall', 'f1']:
       plt.plot(metrics_history[metric], label=metric.capitalize(), marker='o')
   plt.title('Training Metrics')
   plt.xlabel('Epoch')
   plt.ylabel('Score')
   plt.legend()
   plt.grid(True)
   plt.savefig(f'{save_dir}/metrics.png')
   plt.close()

def plot_confusion_matrix(cm, save_dir='plots'):
   plt.figure(figsize=(8, 6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
   plt.title('Confusion Matrix')
   plt.ylabel('True Label')
   plt.xlabel('Predicted Label')
   plt.savefig(f'{save_dir}/confusion_matrix.png')
   plt.close()

def format_metrics(metrics):
   total_support = sum(metrics[label]['support'] for label in ['0.0', '1.0'])
   return (
       f"\n{'='*60}\n"
       f"{'Class':>10} {'Precision':>12} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n"
       f"{'-'*60}\n"
       f"{'Normal':>10} {metrics['0.0']['precision']:>12.4f} {metrics['0.0']['recall']:>10.4f} "
       f"{metrics['0.0']['f1-score']:>10.4f} {metrics['0.0']['support']:>10.0f}\n"
       f"{'Attack':>10} {metrics['1.0']['precision']:>12.4f} {metrics['1.0']['recall']:>10.4f} "
       f"{metrics['1.0']['f1-score']:>10.4f} {metrics['1.0']['support']:>10.0f}\n"
       f"{'-'*60}\n"
       f"{'Total':>10} {'':<12} {'':>10} {'':>10} {total_support:>10.0f}\n"
       f"{'Accuracy':>10} {'':<12} {'':>10} {metrics['accuracy']:>10.4f}\n"
       f"{'Macro Avg':>10} {metrics['macro avg']['precision']:>12.4f} "
       f"{metrics['macro avg']['recall']:>10.4f} {metrics['macro avg']['f1-score']:>10.4f}\n"
       f"{'Wtd Avg':>10} {metrics['weighted avg']['precision']:>12.4f} "
       f"{metrics['weighted avg']['recall']:>10.4f} {metrics['weighted avg']['f1-score']:>10.4f}\n"
       f"{'='*60}\n"
   )

def hex_to_int(x):
   try:
       if isinstance(x, str):
           if x in ['R', 'T']:
               return 0
           return int(x, 16) if 'x' not in x.lower() else int(x, 16)
       return int(x)
   except:
       return 0

def parse_timestamp(ts):
   try:
       if isinstance(ts, str) and any(c.isalpha() for c in ts):
           return float(int(ts, 16))
       return float(ts)
   except:
       return 0.0

def process_data(df, sequence_length=5):
   features = []
   for _, row in df.iterrows():
       data = [hex_to_int(row[f'DATA{i}']) for i in range(8)]
       features.append(data + [parse_timestamp(row['Timestamp']), hex_to_int(row['DLC'])])
   
   scaler = StandardScaler()
   X = scaler.fit_transform(features)
   X = np.clip(X, -3, 3)
   y = (df['Flag'] == 'T').astype(int).values
   
   sequences = []
   labels = []
   for i in range(len(X) - sequence_length + 1):
       sequences.append(X[i:i + sequence_length])
       labels.append(1 if any(y[i:i + sequence_length]) else 0)
   
   return np.array(sequences), np.array(labels)

def train_model(model, train_loader, val_loader, class_weights, epochs=20, patience=5):
   criterion = FocalLoss()
   optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
   scheduler = OneCycleLR(optimizer, max_lr=0.002, epochs=epochs, 
                         steps_per_epoch=len(train_loader))
   device = next(model.parameters()).device
   
   train_losses = []
   val_losses = []
   metrics_history = {
       'accuracy': [], 'precision': [], 'recall': [], 'f1': []
   }
   
   best_val_loss = float('inf')
   patience_counter = 0
   
   for epoch in range(epochs):
       model.train()
       train_loss = 0
       for X_batch, y_batch in train_loader:
           X_batch, y_batch = X_batch.to(device), y_batch.to(device)
           optimizer.zero_grad()
           output = model(X_batch).squeeze()
           loss = criterion(output, y_batch)
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           optimizer.step()
           scheduler.step()
           train_loss += loss.item()
       
       model.eval()
       val_loss = 0
       val_preds = []
       val_true = []
       
       with torch.no_grad():
           for X_batch, y_batch in val_loader:
               X_batch, y_batch = X_batch.to(device), y_batch.to(device)
               output = model(X_batch).squeeze()
               loss = criterion(output, y_batch)
               val_loss += loss.item()
               val_preds.extend((output > 0.5).cpu().numpy())
               val_true.extend(y_batch.cpu().numpy())
       
       train_loss /= len(train_loader)
       val_loss /= len(val_loader)
       
       train_losses.append(train_loss)
       val_losses.append(val_loss)
       
       metrics = classification_report(val_true, val_preds, zero_division=1, output_dict=True)
       metrics_history['accuracy'].append(metrics['accuracy'])
       metrics_history['precision'].append(metrics['macro avg']['precision'])
       metrics_history['recall'].append(metrics['macro avg']['recall'])
       metrics_history['f1'].append(metrics['macro avg']['f1-score'])
       
       logger.info(f"\nEpoch {epoch+1}/{epochs}")
       logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
       logger.info(format_metrics(metrics))
       
       plot_training_history(train_losses, val_losses, metrics_history)
       
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience_counter = 0
           torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'train_loss': train_loss,
               'val_loss': val_loss,
           }, 'best_model.pth')
       else:
           patience_counter += 1
           if patience_counter >= patience:
               logger.info("Early stopping triggered")
               break
   
   return train_losses, val_losses, metrics_history

def main():
   logger.info("Loading and preprocessing data...")
   columns = ['Timestamp', 'CAN_ID', 'DLC'] + [f'DATA{i}' for i in range(8)] + ['Flag']
   
   dfs = []
   for data_type in ['normal_run_data.txt'] + [f'{attack}_dataset.csv' for attack in ['DoS', 'Fuzzy', 'Gear', 'RPM']]:
       if data_type.endswith('.txt'):
           df = pd.read_csv(data_type, sep=r'\s+', names=columns, nrows=100000)
           df['Attack_Type'] = 'Normal'
       else:
           df = pd.read_csv(data_type, names=columns, nrows=100000)
           df['Attack_Type'] = data_type.split('_')[0]
       dfs.append(df)
       logger.info(f"Loaded {len(df)} records from {data_type}")
   
   df = pd.concat(dfs, ignore_index=True)
   df = df.sample(frac=1, random_state=42).reset_index(drop=True)
   save_sample_data(df)
   
   logger.info(f"Total records: {len(df)}")
   
   X, y = process_data(df)
   
   num_normal = np.sum(y == 0)
   num_attack = np.sum(y == 1)
   class_weights = torch.tensor([1.0, num_normal/num_attack], dtype=torch.float32)
    
   X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
   X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, shuffle=True)
    
   logger.info(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
   train_dataset = CANDataset(X_train, y_train)
   val_dataset = CANDataset(X_val, y_val)
   test_dataset = CANDataset(X_test, y_test)
    
   train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
   val_loader = DataLoader(val_dataset, batch_size=128, num_workers=4)
   test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4)
    
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   logger.info(f"Using device: {device}")
   logger.info(f"MPS available: {torch.backends.mps.is_available()}")
   logger.info(f"MPS built: {torch.backends.mps.is_built()}")
    
   class_weights = class_weights.to(device)
   model = CANIDS(input_dim=10).to(device)
    
   train_losses, val_losses, metrics_history = train_model(model, train_loader, val_loader, class_weights, epochs=20, patience=5)
    
   logger.info("\nFinal Evaluation on Test Set:")
   checkpoint = torch.load('best_model.pth', weights_only=True)
   model.load_state_dict(checkpoint['model_state_dict']) 
   model.eval()
    
   test_preds = []
   test_true = []
   with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            test_preds.extend((output > 0.5).cpu().numpy().flatten())
            test_true.extend(y_batch.numpy())
    
   test_metrics = classification_report(test_true, test_preds, zero_division=1, output_dict=True)
   logger.info("\nTest Set Metrics:")
   logger.info(format_metrics(test_metrics))
    
   cm = confusion_matrix(test_true, test_preds)
   logger.info("\nConfusion Matrix:")
   logger.info(cm)
   plot_confusion_matrix(cm)

  # Save final plots
   plot_training_history(train_losses, val_losses, metrics_history)

if __name__ == "__main__":
    main()