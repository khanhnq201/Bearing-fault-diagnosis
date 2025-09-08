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
        
        train_files.extend(fault[:train_end])
        val_files.extend(fault[train_end:val_end])
        test_files.extend(fault[val_end:])

    X_train, Y_train = import_cwru_data(train_files, cfg.SAMPLE_LENGTH, 0,f_cutoff, cfg.BASE_PATH)
    X_val, Y_val = import_cwru_data(val_files, cfg.SAMPLE_LENGTH, 0,f_cutoff, cfg.BASE_PATH)
    X_test, Y_test = import_cwru_data(test_files, cfg.SAMPLE_LENGTH, 0,f_cutoff, cfg.BASE_PATH)

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


    return train_loader, val_loader, test_loader
