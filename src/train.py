import argparse
import os
import yaml # Cần để tải config nếu bạn không dùng utils.py
import logging

import torch # Cho PyTorch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_loader import CWRU_dataloader
import models
from utils import load_config, log_experiment_results

import random
import numpy as np
import copy

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import time

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, 
                                   weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def set_seed(seed: int):
    """
    Thiết lập seed cho random, numpy, và torch để đảm bảo kết quả có thể tái tạo.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, criterion, optimizer, output_name,
                scheduler=None, num_epochs=100, device='cpu', 
                early_stopping_patience=40,
                augmentation=None):
    
    model.to(device)
    best_val_acc = 0.0
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0
            correct_train = 0
            total_train = 0  
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                try:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if augmentation:
                        inputs = augmentation(inputs)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"GPU OOM at epoch {epoch+1}, batch {batch_idx}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Tính metrics training
            if total_train > 0:
                epoch_loss = running_loss / total_train
                epoch_acc_train = 100.0 * correct_train / total_train
            else:
                epoch_loss = 0
                epoch_acc_train = 0
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc_train)
            
            # Validation phase
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0) 
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            # Tính metrics validation
            if total_val > 0:
                epoch_val_loss = val_loss / total_val
                epoch_acc_val = 100.0 * correct_val / total_val
            else:
                epoch_val_loss = 0
                epoch_acc_val = 0
            
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_acc_val)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc_train:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_acc_val:.2f}%")
            
            # Save best model
            if epoch_acc_val > best_val_acc:
                best_val_acc = epoch_acc_val
                patience_counter = 0

                best_model_state = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history
                }, os.path.join('.\experiments\models', output_name))
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Scheduler step
            if scheduler is not None:
                if hasattr(scheduler, 'step'):
                    # Xử lý các loại scheduler khác nhau
                    if 'ReduceLROnPlateau' in str(type(scheduler)):
                        scheduler.step(epoch_val_loss)
                    else:
                        scheduler.step()
            
            # Memory cleanup
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e
    
    model.load_state_dict(best_model_state)
    return model, history

def evaluate_model(model, test_loader, device, class_names):
    y_pred = []
    y_true = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}')
    plt.show()
    print(f" - Model accuracy: {accuracy:.2%}")
    return accuracy

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main(config_path):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    set_seed(config['randomseed'])
    data_loader = CWRU_dataloader(config['dataloader'])
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(processing_type=config['dataloader']['preprocessing_type'])



    input_channel = 1
    num_classes = 3
    if config['model'] == 'Basic1DCNN':
        model = models.Basic1DCNN(input_channel, num_classes)
    if config['model'] == 'Simple1DCNN':
        model = models.SimpleCNN1D(input_channel, num_classes)
    elif config['model'] == 'WDCNN':
        model = models.WDCNN(input_channel, num_classes)
    elif config['model'] == 'Baseline':
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.numpy())

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"Class distribution: {np.bincount(train_labels)}")
        print(f"Class weights: {class_weights}")
        criterion = FocalLoss(alpha=class_weights, gamma=2)
        model = models.Baseline(num_classes)
    if config['model'] != 'Baseline':
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate']) #weight_decay=1e-4
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= config['training']['epochs'], eta_min=config['training']['learning_rate'] *1e-2)

    trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, 'best_model.pth', scheduler, config['training']['epochs'], device=device)

    plot_history(history)
    CLASS_NAMES = ['Normal','IR', 'OR'] 
    acc = evaluate_model(trained_model, test_loader, device,CLASS_NAMES)
    current_timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_data = {
        'timestamp': current_timestamp,
        'sample_length': config['dataloader']['sample_length'],
        'overlapping_ratio': config['dataloader']['overlapping_ratio'],
        'n_rev_resampling': config['dataloader']['n_rev_resampling'],
        'preprocessing_type': config['dataloader']['preprocessing_type'],
        'batch_size': config['dataloader']['batch_size'],
        'nperseg': config['dataloader']['spectrogram']['nperseg'],
        'ratio': config['dataloader']['spectrogram']['ratio'],
        'data_type': config['dataloader']['data_type'],
        'model': config['model'],
        'learning_rate': config['training']['learning_rate'],
        'epochs': config['training']['epochs'],
        'accuracy': acc
    }
    acc = evaluate_model(trained_model, val_loader, device,CLASS_NAMES)
    acc = evaluate_model(trained_model, train_loader, device,CLASS_NAMES)
    exp_base_dir = 'experiments'
    global_log_filepath = os.path.join(exp_base_dir, 'experiment_log.csv')
    log_experiment_results(global_log_filepath, experiment_data)

if __name__ == "__main__":
    main('config/default_config.yaml') 

    
    