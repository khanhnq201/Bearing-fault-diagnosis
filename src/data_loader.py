import numpy as np
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.signal import hilbert
from scipy.fft import fft, ifft
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from scipy.signal import decimate # <--- IMPORT MỚI
from pathlib import Path
import re
import scipy
import random

def preprocessing(raw_signal, sampling_rate, fmax=None):
    """
    Tính toán phổ bao (envelope spectrum) của tín hiệu đã được làm trắng (whitened).

    Hàm này thực hiện các bước:
    1. Làm trắng tín hiệu trong miền tần số.
    2. Tính toán tín hiệu bao (envelope) bằng biến đổi Hilbert.
    3. Tính toán phổ của tín hiệu bao (FFT).
    4. Trích xuất và trả về các giá trị biên độ của phổ đến tần số fmax.

    Args:
        raw_signal (np.ndarray): Mảng 1D chứa tín hiệu thô.
        sampling_rate (int): Tần số lấy mẫu của tín hiệu (ví dụ: 12000).
        fmax (int): Tần số tối đa của phổ cần trích xuất (ví dụ: 500).

    Returns:
        tuple[np.ndarray, np.ndarray]: Một tuple chứa hai mảng:
            - freq_axis_under_fmax: Các giá trị tần số đến fmax.
            - magnitude_values_under_fmax: Các giá trị biên độ tương ứng.
    """
    # 1. Làm trắng tín hiệu (Whitening)
    fft_raw = np.fft.fft(raw_signal)
    phase_raw = np.angle(fft_raw)
    fft_whitened = np.exp(1j * phase_raw)
    signal_whitened = np.fft.ifft(fft_whitened).real

    # 2. Tính tín hiệu bao (Envelope)
    analytic_signal = scipy.signal.hilbert(signal_whitened)
    envelope = np.abs(analytic_signal)
    
    # Bỏ thành phần DC để kết quả FFT tốt hơn ở tần số gần 0
    envelope_centered = envelope - np.mean(envelope)

    # 3. Tính phổ của tín hiệu bao (Envelope Spectrum)
    N = len(envelope_centered)
    fft_result = np.fft.fft(envelope_centered)
    
    # Tính toán biên độ và trục tần số cho nửa phổ dương
    magnitude_full = 2.0/N * np.abs(fft_result[:N//2])
    freq_axis_full = np.fft.fftfreq(N, 1/sampling_rate)[:N//2]

    if fmax== None:
        return freq_axis_full, magnitude_full
    else:
    # 4. Trích xuất các giá trị đến fmax
        end_index = np.searchsorted(freq_axis_full, fmax, side='right')
        
        magnitude_values_under_fmax = magnitude_full[:end_index]
        freq_axis_under_fmax = freq_axis_full[:end_index]
        
        return freq_axis_under_fmax, magnitude_values_under_fmax

def get_label_from_path(file_path: Path) -> int:
    """Analyze the number and percent of each class in a dataset. 

    Args:
        file_path (path): The path of given file.

    Returns:
        label (int): 4 classes originally, but 3 classes after selecting.
            - 0 for 'Normal'
            - 1 for 'Inner'
            - 2 for 'Outter'
    """
    path_parts = file_path.parts
    if 'Normal' in path_parts:

        return 0
    
    #--For full label only--
    # elif 'B' in path_parts:
    #     return 1
    # elif 'IR' in path_parts:
    #     return 2
    # elif 'OR' in path_parts:
    #     return 3
    #------------------------

    elif 'IR' in path_parts:

        return 1
    elif 'OR' in path_parts:

        return 2
    return -1

def import_cwru_data(
    file_keys: list,
    sample_length: int,
    overlapping_ratio: float,
    f_cutoff,
    base_path: str = 'CWRU-dataset-main'
):
    """
    Import data from file name list 
    
    Args:
        file_keys (np.arr[str]): file name list in format 'number of file'+'sensor location'.
        sample_length (int): length of each sample importing (2048, 4096 recommended).
        overlapping_ratio (float 0 -> 1): (0.25 for training set recommended).
        base_path (path): the path to data folder in the workspace.

    Returns
    ------
        X (np.arr): data 
        Y (np.arr): labels (wo onehot coding)
    """

    all_samples = []
    all_labels = []
    base_path_obj = Path(base_path)

    if overlapping_ratio > 1 or overlapping_ratio < 0: overlapping_ratio = 0
    step = int(sample_length * (1 - overlapping_ratio))
    if step < 1:
        step = 1

    for key in file_keys:        
        match = re.match(r'(\d+)(\w+)', key)      
        file_num_str, data_key_suffix = match.groups()


        glob_pattern = f'{file_num_str}_*.mat'
        found_files = list(base_path_obj.rglob(glob_pattern))

        if not found_files:
            print(f"    [WARNING] No file for '{file_num_str}'")
            continue
        
        file_path = found_files[0]
        label = get_label_from_path(file_path)

        mat_data = loadmat(file_path)
        
        # Each key in mat file stand for each sensor 
        mat_key_zfill = f'X{file_num_str.zfill(3)}_{data_key_suffix}_time'
        mat_key_normal = f'X{file_num_str}_{data_key_suffix}_time'
        
        if mat_key_zfill in mat_data:
            mat_key = mat_key_zfill
        elif mat_key_normal in mat_data:
            mat_key = mat_key_normal

        time_series = mat_data[mat_key].flatten()

        #Overlapping
        num_samples_in_file = 0
        file_samples = []
        for i in range(0, len(time_series) - sample_length + 1, step):
            sample = time_series[i : i + sample_length]
            _, sample2 = preprocessing(sample, 12000, f_cutoff)
            file_samples.append(sample2)
            num_samples_in_file += 1
        
        if num_samples_in_file > 0:
            all_samples.extend(file_samples)
            all_labels.extend([label] * num_samples_in_file)

    X = np.array(all_samples)
    Y = np.array(all_labels)
    
    return X, Y

def data_import(cfg, f_cutoff = None, train_ratio = 0.7, val_ratio=0.2): 
    """
    Import train-val-test set (fixed file)
    
    Args:
        sample_length (int): length of each sample importing (2048, 4096 recommended).
        overlapping_ratio (float): for training set only (0.25 recommended).
        base_path (path): the path to data folder in the workspace.
        preprocessing (bool): turn on if you wanna do envelope analysis

    Returns
    ------
        X (np.arr): data 
        Y (np.arr): labels (wo onehot coding)
    """

    # train_files = [
    #     # --- Normal Data (Label 0) ---
    #     '97DE', '97FE',   # Normal @ 0HP
    #     '98DE', '98FE',  # Normal @ 1HP

    #     # --- Inner Race (IR) Faults (Label 2) ---
    #     '209DE',
    #     '209FE',  # DE, IR, 0.021"
    #     '210DE',          # DE, IR, 0.021"
    #     '278DE', '278FE',  # FE, IR, 0.007"
    #     '280DE', '280BA',  # FE, IR, 0.007"
    #     '271DE', '271FE', '271BA', # FE, IR, 0.021"
    #     '276FE', 
    #     '276BA',  # FE, IR, 0.014"
    #     '277FE', '277BA',  # FE, IR, 0.014"
        

    #     # --- Outer Race (OR) Faults (Label 3) ---
    #     '130DE', '131DE',  # DE, OR (Centred), 0.007"
    #     '144DE', '144BA',  # DE, OR (Orthogonal), 0.007"
    #     '145DE', '145FE', '145BA', # DE, OR (Orthogonal), 0.007"
    #     '156DE', '156FE',  # DE, OR (Opposite), 0.007"
    #     '310DE', '310FE',  # FE, OR (Orthogonal), 0.007"
    #     '309DE',          # FE, OR (Orthogonal), 0.014"
    #     '311DE', '311FE',  # FE, OR (Orthogonal), 0.014"
    #     '313DE', '313FE',  # FE, OR (Centred), 0.007"  
    # ]

    # val_files = [
    #     # --- Normal Data (Label 0) ---
    #     '99DE', '99FE',   # Normal @ 2HP

    #     # --- Inner Race (IR) Faults (Label 2) ---
    #     '211DE',          # DE, IR, 0.021"
    #     '279DE',          # FE, IR, 0.007"
    #     '274FE',          # FE, IR, 0.014"
    #     '272DE', '272FE', '272BA', # FE, IR, 0.021"
        
    #     # --- Outer Race (OR) Faults (Label 3) ---
    #     '132DE',          # DE, OR (Centred), 0.014"
    #     '146DE', '146FE', '146BA', # DE, OR (Orthogonal), 0.014"
    #     '159DE',          # DE, OR (Opposite), 0.014"
    #     '312DE', '312FE',  # FE, OR (Orthogonal), 0.021"
    #     '315DE',          # FE, OR (Centred), 0.021"
    # ]

    # test_files = [
    #     # --- Normal Data (Label 0) ---
    #     '100DE', '100FE', # Normal @ 3HP

    #     # --- Inner Race (IR) Faults (Label 2) ---
    #     '212DE',          # DE, IR, 0.021"
    #     '281DE',          # FE, IR, 0.007"
    #     '275FE',          # FE, IR, 0.014"
    #     '273DE', '273FE', '273BA', # FE, IR, 0.021"

    #     # --- Outer Race (OR) Faults (Label 3) ---
    #     '133DE',          # DE, OR (Centred), 0.021"
    #     '147DE', '147FE', '147BA', # DE, OR (Orthogonal), 0.021"
    #     '160DE',          # DE, OR (Opposite), 0.021"
    #     '317DE', '317FE', '317BA', # FE, OR (Orthogonal), 0.021"
    # ]

    normal_files = [
        # --- Normal Data (Label 0) ---
        '97DE', '97FE',   # Normal @ 0HP
        '98DE', '98FE',  # Normal @ 1HP
        # --- Normal Data (Label 0) ---
        '99DE', '99FE',   # Normal @ 2HP
        # --- Normal Data (Label 0) ---
        '100DE', '100FE', # Normal @ 3HP
    ]

    ir_files = [
        # --- Inner Race (IR) Faults (Label 2) ---
        '211DE',          # DE, IR, 0.021"
        '279DE',          # FE, IR, 0.007"
        '274FE',          # FE, IR, 0.014"
        '272DE', '272FE', '272BA', # FE, IR, 0.021"
        # --- Inner Race (IR) Faults (Label 2) ---
        '209DE',
        '209FE',  # DE, IR, 0.021"
        '210DE',          # DE, IR, 0.021"
        '278DE', '278FE',  # FE, IR, 0.007"
        '280DE', '280BA',  # FE, IR, 0.007"
        '271DE', '271FE', '271BA', # FE, IR, 0.021"
        '276FE', 
        '276BA',  # FE, IR, 0.014"
        '277FE', '277BA',  # FE, IR, 0.014"
        # --- Inner Race (IR) Faults (Label 2) ---
        '212DE',          # DE, IR, 0.021"
        '281DE',          # FE, IR, 0.007"
        '275FE',          # FE, IR, 0.014"
        '273DE', '273FE', '273BA', # FE, IR, 0.021"
    ]

    or_files = [
        # --- Outer Race (OR) Faults (Label 3) ---
        '132DE',          # DE, OR (Centred), 0.014"
        '146DE', '146FE', '146BA', # DE, OR (Orthogonal), 0.014"
        '159DE',          # DE, OR (Opposite), 0.014"
        '312DE', '312FE',  # FE, OR (Orthogonal), 0.021"
        '315DE',          # FE, OR (Centred), 0.021"
        # --- Outer Race (OR) Faults (Label 3) ---
        '133DE',          # DE, OR (Centred), 0.021"
        '147DE', '147FE', '147BA', # DE, OR (Orthogonal), 0.021"
        '160DE',          # DE, OR (Opposite), 0.021"
        '317DE', '317FE', '317BA', # FE, OR (Orthogonal), 0.021"
        # --- Outer Race (OR) Faults (Label 3) ---
        '130DE', '131DE',  # DE, OR (Centred), 0.007"
        '144DE', '144BA',  # DE, OR (Orthogonal), 0.007"
        '145DE', '145FE', '145BA', # DE, OR (Orthogonal), 0.007"
        '156DE', '156FE',  # DE, OR (Opposite), 0.007"
        '310DE', '310FE',  # FE, OR (Orthogonal), 0.007"
        '309DE',          # FE, OR (Orthogonal), 0.014"
        '311DE', '311FE',  # FE, OR (Orthogonal), 0.014"
        '313DE', '313FE',  # FE, OR (Centred), 0.007" 
    ]
    data = [normal_files, ir_files, or_files]

    train_files, val_files, test_files = [], [], []
    for fault in data:
        random.shuffle(fault)
        n_samples = len(fault)
        train_end = int(train_ratio*n_samples)
        val_end = train_end + round(val_ratio*n_samples)
        # print('='*50)
        # print('>>> File: ',fault[:train_end])
        # print('>>> Len: ',len(fault[:train_end]))
        # print('>>> File: ',fault[train_end:val_end])
        # print('>>> Len: ',len(fault[train_end:val_end]))
        # print('>>> File: ',fault[val_end:])
        # print('>>> Len: ',len(fault[val_end:]))
        
        train_files.extend(fault[:train_end])
        val_files.extend(fault[train_end:val_end])
        test_files.extend(fault[val_end:])
    print(cfg.OVERLAPPING_RATIO)
    X_train, Y_train = import_cwru_data(train_files, cfg.SAMPLE_LENGTH, cfg.OVERLAPPING_RATIO,f_cutoff, cfg.BASE_PATH)

    X_val, Y_val = import_cwru_data(val_files, cfg.SAMPLE_LENGTH, 0.75,f_cutoff, cfg.BASE_PATH)
    X_test, Y_test = import_cwru_data(test_files, cfg.SAMPLE_LENGTH, 0.75,f_cutoff, cfg.BASE_PATH)

    len_processed = int(X_train.shape[1])

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, test_files, len_processed

class BearingDataset(Dataset):
    def __init__(self, X_data, Y_data, is_train=True): 
        self.data = torch.from_numpy(X_data).float()
        self.labels = torch.from_numpy(Y_data).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def create_dataloaders(X_train, Y_train, 
                       X_val, Y_val, 
                       X_test, Y_test, cfg, sample_length):
    """ Load data
    """

    X_train_reshaped = np.reshape(X_train, (-1, 1, sample_length))
    X_val_reshaped = np.reshape(X_val, (-1, 1, sample_length))
    X_test_reshaped = np.reshape(X_test, (-1, 1, sample_length))

    train_dataset = BearingDataset(X_train_reshaped, Y_train)
    train_loader = DataLoader(train_dataset, batch_size= cfg.BATCH_SIZE,shuffle=True, num_workers=0)

    val_dataset = BearingDataset(X_val_reshaped, Y_val)
    val_loader = DataLoader(val_dataset, batch_size= cfg.BATCH_SIZE,shuffle=False, num_workers=0)

    test_dataset = BearingDataset(X_test_reshaped, Y_test)
    test_loader = DataLoader(test_dataset, batch_size= cfg.BATCH_SIZE,shuffle=False, num_workers=0)

    # print("\n" + "="*50)
    # print("DATA DISTRIBUTION SUMMARY")
    # print("="*50)
    # print(f"Training samples:   {len(train_loader.dataset):>6}")
    # print(f"Validation samples: {len(val_loader.dataset):>6}")
    # print(f"Test samples:       {len(test_loader.dataset):>6}")
    # print(f"Total samples:      {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset):>6}")

    return train_loader, val_loader, test_loader

def evaluate_model_on_files(model, file_keys, config):
    """
    Đánh giá một model đã được huấn luyện trên một danh sách các file cụ thể,
    cung cấp báo cáo chi tiết cho từng file và một báo cáo tổng hợp.

    Args:
        model (torch.nn.Module): Model PyTorch đã được huấn luyện.
        file_keys (list): Danh sách các key của file cần đánh giá (ví dụ: ['99DE', '211DE']).
        config: Đối tượng config chứa các tham số.
    """
    print("\n" + "="*70)
    print(f" BẮT ĐẦU ĐÁNH GIÁ CHI TIẾT TRÊN CÁC FILE: {file_keys}")
    print("="*70)

    if config.SAMPLE_LENGTH == 2048: 
        sample_length = 86
    elif config.SAMPLE_LENGTH == 4096:
        sample_length = 171

    # 1. Đưa model về chế độ đánh giá
    device = config.DEVICE
    model.to(device)
    model.eval()

    # Chuẩn bị các list để lưu kết quả tổng hợp
    all_preds_total = []
    all_labels_total = []


    # 2. Lặp qua từng file để xử lý và báo cáo riêng lẻ
    for file_key in file_keys:
        print(f"\n--- Đang xử lý File: {file_key} ---")

        X_file, Y_file = import_cwru_data([file_key], config.SAMPLE_LENGTH, 0, config.BASE_PATH)
        X_file_norm = X_file *1e2
        num_samples = X_file_norm.shape[0]
        true_label_index = Y_file[0] 
        true_label_name = config.CLASS_NAMES[true_label_index]

        X_file_preprocessed = np.reshape(X_file_norm, (-1, 1, sample_length))

        # Tạo DataLoader cho file
        file_dataset = BearingDataset(X_file_preprocessed, Y_file)
        file_loader = DataLoader(file_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Thực hiện dự đoán trên dữ liệu của file
        file_preds = []
        with torch.no_grad():
            for data, labels in file_loader:
                data = data.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                file_preds.extend(predicted.cpu().numpy())
        
        # Phân tích và in kết quả dự đoán cho file này
        unique_predictions, counts = np.unique(file_preds, return_counts=True)
        
        # Sắp xếp kết quả dự đoán theo số lượng giảm dần để dễ đọc
        prediction_summary = sorted(zip(unique_predictions, counts), key=lambda item: item[1], reverse=True)

        for pred_label, count in prediction_summary:
            pred_name = config.CLASS_NAMES[pred_label]
            percentage = (count / num_samples) * 100
            print(f"    [INFO]'{pred_name}': {count:<4} mẫu ({percentage:>6.2f}%)")

        # Thêm kết quả của file này vào danh sách tổng hợp
        all_preds_total.extend(file_preds)
        all_labels_total.extend(Y_file)

    # 3. Báo cáo kết quả tổng hợp cho tất cả các file đã xử lý
    if not all_labels_total:
        print("\n[THÔNG BÁO] Không có dữ liệu nào được xử lý. Dừng báo cáo tổng kết.")
        return

    print("\n" + "="*70)
    print(" TỔNG KẾT TOÀN BỘ CÁC FILE ĐÃ KIỂM TRA")
    print("="*70)
    
    accuracy = accuracy_score(all_labels_total, all_preds_total)
    print(f"-> Độ chính xác tổng hợp (Overall Accuracy): {accuracy:.4f} ({int(accuracy*len(all_labels_total))}/{len(all_labels_total)})")

    all_possible_labels = list(range(len(config.CLASS_NAMES)))

    print("\n-> Báo cáo phân loại tổng hợp (Overall Classification Report):")
    print(classification_report(all_labels_total, 
                                all_preds_total, 
                                labels=all_possible_labels, 
                                target_names=config.CLASS_NAMES, 
                                zero_division=0))

    # print("\n-> Ma trận nhầm lẫn tổng hợp (Overall Confusion Matrix):")
    # cm = confusion_matrix(all_labels_total, all_preds_total, labels=all_possible_labels)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Overall Confusion Matrix on Custom Data')
    # plt.show()