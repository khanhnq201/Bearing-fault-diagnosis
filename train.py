import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import re
import random
import os

# --- Import các module từ dự án của bạn ---
import cfg
# import trainer # Sẽ không dùng trainer.py gốc, mà dùng hàm tùy chỉnh bên dưới
from src import baseline_models, trainer, evaluate
from torch.utils.data import DataLoader, Dataset

# =====================================================================================
# PHẦN 1: LOGIC TẢI DỮ LIỆU THÔ (RAW DATA)
# Ghi chú: Tái sử dụng các hàm đã được xác nhận
# =====================================================================================

def get_label_from_path(file_path: Path) -> int:
    path_parts = file_path.parts
    if 'Normal' in path_parts: return 0
    elif 'IR' in path_parts: return 1
    elif 'OR' in path_parts: return 2
    return -1

def import_cwru_data_raw(file_keys: list, sample_length: int, base_path: str, overlapping_ratio: float = 0.0):
    all_samples, all_labels = [], []
    base_path_obj = Path(base_path)
    step = int(sample_length * (1 - overlapping_ratio))
    if step < 1: step = 1

    for key in file_keys:
        match = re.match(r'(\d+)(\w+)', key)
        if not match: continue
        file_num_str, data_key_suffix = match.groups()
        glob_pattern = f'{file_num_str}_*.mat'
        found_files = list(base_path_obj.rglob(glob_pattern))
        if not found_files: continue
        file_path = found_files[0]
        mat_data = loadmat(file_path)
        label = get_label_from_path(file_path)
        mat_key_zfill = f'X{file_num_str.zfill(3)}_{data_key_suffix}_time'
        mat_key_normal = f'X{file_num_str}_{data_key_suffix}_time'
        mat_key = mat_key_zfill if mat_key_zfill in mat_data else mat_key_normal
        if mat_key not in mat_data: continue
        time_series = mat_data[mat_key].flatten()
        for i in range(0, len(time_series) - sample_length + 1, step):
            all_samples.append(time_series[i : i + sample_length])
            all_labels.append(label)
    return np.array(all_samples), np.array(all_labels)

def data_import_raw(sample_length: int, overlapping_ratio: float, train_ratio=0.7, val_ratio=0.2):
    normal_files = ['97DE', '97FE', '98DE', '98FE', '99DE', '99FE', '100DE', '100FE']
    ir_files = ['211DE', '279DE', '274FE', '272DE', '272FE', '272BA', '209DE', '209FE', '210DE', '278DE', '278FE', '280DE', '280BA', '271DE', '271FE', '271BA', '276FE', '276BA', '277FE', '277BA', '212DE', '281DE', '275FE', '273DE', '273FE', '273BA']
    or_files = ['132DE', '146DE', '146FE', '146BA', '159DE', '312DE', '312FE', '315DE', '133DE', '147DE', '147FE', '147BA', '160DE', '317DE', '317FE', '317BA', '130DE', '131DE', '144DE', '144BA', '145DE', '145FE', '145BA', '156DE', '156FE', '310DE', '310FE', '309DE', '311DE', '311FE', '313DE', '313FE']
    data = [normal_files, ir_files, or_files]
    train_files, val_files, test_files = [], [], []
    for fault in data:
        random.shuffle(fault)
        n_samples = len(fault)
        train_end = int(train_ratio * n_samples)
        val_end = train_end + round(val_ratio * n_samples)
        train_files.extend(fault[:train_end]); val_files.extend(fault[train_end:val_end]); test_files.extend(fault[val_end:])
    X_train, Y_train = import_cwru_data_raw(train_files, sample_length, cfg.BASE_PATH, overlapping_ratio)
    X_val, Y_val = import_cwru_data_raw(val_files, sample_length, cfg.BASE_PATH, 0.0)
    X_test, Y_test = import_cwru_data_raw(test_files, sample_length, cfg.BASE_PATH, 0.0)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

class BearingDatasetRaw(Dataset):
    def __init__(self, X_data, Y_data, normalization=None):
        self.data = torch.from_numpy(X_data).float()
        self.labels = torch.from_numpy(Y_data).long()
        self.normalization = normalization
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.normalization == 'z-score':
            std = sample.std()
            if std > 1e-8: sample = (sample - sample.mean()) / std
        elif self.normalization == 'min-max':
            min_val, max_val = sample.min(), sample.max()
            range_val = max_val - min_val
            if range_val > 1e-8: sample = (sample - min_val) / range_val
        return sample.unsqueeze(0), self.labels[idx]

# =====================================================================================
# PHẦN 2: HÀM HUẤN LUYỆN TÙY CHỈNH CHO ACDIN
# =====================================================================================

def train_acdin_model(model, train_loader, val_loader, criterion, optimizer, output_name,
                      scheduler=None, num_epochs=100, device='cpu', 
                      early_stopping_patience=5, aux_loss_weight=0.3):
    model.to(device)
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running_loss, correct_train, total_train = 0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # ACDIN trả về 3 outputs khi training
            main_out, aux1_out, aux2_out = model(inputs)
            
            # Tính loss cho từng output
            loss_main = criterion(main_out, labels)
            loss_aux1 = criterion(aux1_out, labels)
            loss_aux2 = criterion(aux2_out, labels)
            
            # Tính loss tổng hợp
            total_loss = loss_main + aux_loss_weight * (loss_aux1 + loss_aux2)
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * inputs.size(0)
            _, predicted = torch.max(main_out.data, 1) # Chỉ tính accuracy trên output chính
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total_train
        epoch_acc_train = 100.0 * correct_train / total_train
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc_train)
        
        # --- Validation phase ---
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Khi eval, ACDIN chỉ trả về 1 output
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / total_val
        epoch_acc_val = 100.0 * correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_acc_val)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc_train:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_acc_val:.2f}%")
        
        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict()}, os.path.join('./outputs/models', output_name))
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if scheduler: scheduler.step()

    return model, history

# =====================================================================================
# PHẦN 3: QUY TRÌNH HUẤN LUYỆN ACDIN
# =====================================================================================

if __name__ == '__main__':
    # 1. Thiết lập các tham số
    ACDIN_INPUT_LENGTH = 5120
    TRAIN_OVERLAPPING_RATIO = 0.95
    NORMALIZATION_METHOD = None # Tắt chuẩn hóa

    cfg.BATCH_SIZE = 32
    cfg.LEARNING_RATE = 1e-4

    accuracy_list= []
    for i in range(30):
        trainer.set_seed(i)
    
        # 2. Tải và chuẩn bị dữ liệu
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_import_raw(
            sample_length=ACDIN_INPUT_LENGTH,
            overlapping_ratio=TRAIN_OVERLAPPING_RATIO
        )
        
        train_dataset = BearingDatasetRaw(X_train, Y_train)
        val_dataset = BearingDatasetRaw(X_val, Y_val)
        test_dataset = BearingDatasetRaw(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)


        # 3. Khởi tạo mô hình, optimizer, scheduler, loss
        model = baseline_models.ACDIN(input_channels=1, num_classes=cfg.NUM_CLASSES)
        model.to(cfg.DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.NUM_EPOCHS, eta_min=5e-5 * 1e-2
        )


        trained_model, history = train_acdin_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            output_name='acdin_best.pth', # Tên file lưu model
            scheduler=scheduler, 
            num_epochs= 10, 
            device=cfg.DEVICE,
            aux_loss_weight=0.3 # Trọng số cho loss phụ
        )

        accuracy = evaluate.plot_confusion_matrix(trained_model, test_loader, 'cpu', cfg.CLASS_NAMES)
        accuracy_list.append(accuracy)
        evaluate.plot_history(history)

    print('='*20)
    print('Mean accuracy:', np.mean(accuracy_list))
    print('Std accuracy:', np.std(accuracy_list))