import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import re
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import hilbert, stft
from scipy.interpolate import interp1d

class BearingDataset(Dataset):
    def __init__(self, X_data, Y_data): 
        self.data = torch.from_numpy(X_data).float()
        self.labels = torch.from_numpy(Y_data).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CWRU_dataloader: 
    def __init__(self, batch_size=32):
        self.base_path = "CWRU-dataset-main"
        self.fs = 12000
        self.batch_size = batch_size

        self.sample_length = 4096
        self.overlapping_ratio = 0

        self.train_list = [
                #Normal
                "97DE", "98DE",

                #IR
                "105DE", "106DE",
                 "169DE", "170DE",
                 "209DE", "210DE", 

                 #OR
                 "130DE", "131DE", 
                 "234DE", "235DE",
                 "144DE", "145DE",
                 "156DE", "158DE",
                 "246DE", "247DE",
                 "258DE", "259DE"]

        self.val_list = [
                "99DE",

                "107DE",
                "171DE",
                "211DE",

                "132DE",
                "236DE",
                "146DE",
                "159DE",
                "248DE",
                "260DE"
        ]

        self.test_list = [
                "100DE",

                "108DE",
                "172DE",
                "212DE",

                "133DE",
                "237DE",
                "147DE",
                "160DE",
                "249DE",
                "261DE"
        ]


        self.model_input_size = 512
        self.n_rev = 64
        self.default_rpm = 1750
        
        # Store RPM for each file
        self.rpm_dict = {}

    def get_label_from_path(self, file_path: Path) -> int:
        path_parts = file_path.parts
        if 'Normal' in path_parts: return 0
        elif 'IR' in path_parts: return 1
        elif 'OR' in path_parts: return 2
        return -1

    def import_files(self, file_keys):
        all_samples = []
        all_labels = []
        all_file_ids = []  # Track which file each sample came from
        base_path_obj = Path(self.base_path)

        if self.overlapping_ratio > 1 or self.overlapping_ratio < 0: 
            self.overlapping_ratio = 0
        step = int(self.sample_length * (1 - self.overlapping_ratio))
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
            label = self.get_label_from_path(file_path)

            mat_data = loadmat(file_path)

            # Get time series data
            mat_key_zfill = f'X{file_num_str.zfill(3)}_{data_key_suffix}_time'
            mat_key_normal = f'X{file_num_str}_{data_key_suffix}_time'
            
            if mat_key_zfill in mat_data:
                mat_key = mat_key_zfill
            elif mat_key_normal in mat_data:
                mat_key = mat_key_normal
            else:
                print(f"    [WARNING] No time series key found for '{key}'")
                continue
                
            time_series = mat_data[mat_key].flatten()

            # Try to get RPM from file
            rpm = self.default_rpm
            rpm_key_zfill = f'X{file_num_str.zfill(3)}_RPM'
            rpm_key_normal = f'X{file_num_str}_RPM'
            
            if rpm_key_zfill in mat_data:
                rpm = float(mat_data[rpm_key_zfill].flatten()[0])
            elif rpm_key_normal in mat_data:
                rpm = float(mat_data[rpm_key_normal].flatten()[0])
            else:
                # Try to find any key with RPM
                for k in mat_data.keys():
                    if 'RPM' in k and not k.startswith('__'):
                        rpm = float(mat_data[k].flatten()[0])
                        break
            
            self.rpm_dict[key] = rpm

            # Overlapping windowing
            num_samples_in_file = 0
            file_samples = []
            for i in range(0, len(time_series) - self.sample_length + 1, step):
                sample = time_series[i : i + self.sample_length]
                file_samples.append(sample)
                all_file_ids.append(key)
                num_samples_in_file += 1
            
            if num_samples_in_file > 0:
                all_samples.extend(file_samples)
                all_labels.extend([label] * num_samples_in_file)

        X = np.array(all_samples)
        Y = np.array(all_labels)
        file_ids = np.array(all_file_ids)
        
        return X, Y, file_ids
        
    def import_data(self):
        self.train_samples, self.train_labels, self.train_file_ids = self.import_files(self.train_list)
        self.val_samples, self.val_labels, self.val_file_ids = self.import_files(self.val_list)
        self.test_samples, self.test_labels, self.test_file_ids = self.import_files(self.test_list)

        return self.train_samples, self.train_labels, self.val_samples, self.val_labels, self.test_samples, self.test_labels
    
    def _angular_resampling(self, signal, rpm):
        fr = rpm / 60.0
        t = np.arange(len(signal)) / self.fs
        theta_t = 2 * np.pi * fr * t
        theta_t = np.squeeze(theta_t)

        theta_target_step = 2 * np.pi / self.n_rev
        theta_target = np.arange(0, self.model_input_size * theta_target_step, theta_target_step)

        interp_func = interp1d(theta_t, signal, kind='cubic', bounds_error=False, fill_value=0)
        signal_angular = interp_func(theta_target)
        return signal_angular
    
    def _envelope_extraction(self, signal):
        envelope = np.abs(hilbert(signal))
        envelope_centered = envelope - np.mean(envelope)
        return envelope_centered
    
    def _spectrum(self, signal):
        N = len(signal)
        fft_result = np.fft.fft(signal)
        magnitude = 2.0 / N * np.abs(fft_result[:N//2])
        return magnitude
    
    def freq_spectrum(self):
        """Convert time domain signals to frequency spectrum using FFT"""
        print("Processing frequency spectrum...")
        
        # Process training data
        train_spectrums = []
        for signal in self.train_samples:
            envelope = self._envelope_extraction(signal)
            spectrum = self._spectrum(envelope)
            train_spectrums.append(spectrum)
        self.preprocessed_train_samples = np.array(train_spectrums)[:, np.newaxis, :]  # (N, 1, freq_bins)
        
        # Process validation data
        val_spectrums = []
        for signal in self.val_samples:
            envelope = self._envelope_extraction(signal)
            spectrum = self._spectrum(envelope)
            val_spectrums.append(spectrum)
        self.preprocessed_val_samples = np.array(val_spectrums)[:, np.newaxis, :]
        
        # Process test data
        test_spectrums = []
        for signal in self.test_samples:
            envelope = self._envelope_extraction(signal)
            spectrum = self._spectrum(envelope)
            test_spectrums.append(spectrum)
        self.preprocessed_test_samples = np.array(test_spectrums)[:, np.newaxis, :]
        
        print(f"Frequency spectrum shape - Train: {self.preprocessed_train_samples.shape}, Val: {self.preprocessed_val_samples.shape}, Test: {self.preprocessed_test_samples.shape}")
    
    def order_spectrum(self):
        """Convert to order domain then compute spectrum"""
        print("Processing order spectrum...")
        
        # Process training data
        train_spectrums = []
        for i, signal in enumerate(self.train_samples):
            file_id = self.train_file_ids[i]
            rpm = self.rpm_dict.get(file_id, self.default_rpm)
            signal_angular = self._angular_resampling(signal, rpm)
            envelope = self._envelope_extraction(signal_angular)
            spectrum = self._spectrum(envelope)
            train_spectrums.append(spectrum)
        self.preprocessed_train_samples = np.array(train_spectrums)[:, np.newaxis, :]
        
        # Process validation data
        val_spectrums = []
        for i, signal in enumerate(self.val_samples):
            file_id = self.val_file_ids[i]
            rpm = self.rpm_dict.get(file_id, self.default_rpm)
            signal_angular = self._angular_resampling(signal, rpm)
            envelope = self._envelope_extraction(signal_angular)
            spectrum = self._spectrum(envelope)
            val_spectrums.append(spectrum)
        self.preprocessed_val_samples = np.array(val_spectrums)[:, np.newaxis, :]
        
        # Process test data
        test_spectrums = []
        for i, signal in enumerate(self.test_samples):
            file_id = self.test_file_ids[i]
            rpm = self.rpm_dict.get(file_id, self.default_rpm)
            signal_angular = self._angular_resampling(signal, rpm)
            spectrum = self._spectrum(signal_angular)
            test_spectrums.append(spectrum)
        self.preprocessed_test_samples = np.array(test_spectrums)[:, np.newaxis, :]
        
        print(f"Order spectrum shape - Train: {self.preprocessed_train_samples.shape}, Val: {self.preprocessed_val_samples.shape}, Test: {self.preprocessed_test_samples.shape}")
    
    def order_spectrogram(self, nperseg=256, noverlap=128):
        """Convert to order domain then compute spectrogram using STFT"""
        print("Processing order spectrogram...")
        
        # Process training data
        train_spectrograms = []
        for i, signal in enumerate(self.train_samples):
            file_id = self.train_file_ids[i]
            rpm = self.rpm_dict.get(file_id, self.default_rpm)
            signal_angular = self._angular_resampling(signal, rpm)
            envelope = self._envelope_extraction(signal_angular)
            
            # Compute STFT
            f, t, Zxx = stft(envelope, fs=self.n_rev, nperseg=nperseg, noverlap=noverlap)
            spectrogram = np.abs(Zxx)  # Magnitude
            train_spectrograms.append(spectrogram)
        
        self.preprocessed_train_samples = np.array(train_spectrograms)[:, np.newaxis, :, :]  # (N, 1, freq, time)
        
        # Process validation data
        val_spectrograms = []
        for i, signal in enumerate(self.val_samples):
            file_id = self.val_file_ids[i]
            rpm = self.rpm_dict.get(file_id, self.default_rpm)
            signal_angular = self._angular_resampling(signal, rpm)
            envelope = self._envelope_extraction(signal_angular)
            
            # Compute STFT
            f, t, Zxx = stft(envelope, fs=self.n_rev, nperseg=nperseg, noverlap=noverlap)
            spectrogram = np.abs(Zxx)
            val_spectrograms.append(spectrogram)
        
        self.preprocessed_val_samples = np.array(val_spectrograms)[:, np.newaxis, :, :]
        
        # Process test data
        test_spectrograms = []
        for i, signal in enumerate(self.test_samples):
            file_id = self.test_file_ids[i]
            rpm = self.rpm_dict.get(file_id, self.default_rpm)
            signal_angular = self._angular_resampling(signal, rpm)
            envelope = self._envelope_extraction(signal_angular)
            
            # Compute STFT
            f, t, Zxx = stft(envelope, fs=self.n_rev, nperseg=nperseg, noverlap=noverlap)
            spectrogram = np.abs(Zxx)
            test_spectrograms.append(spectrogram)
        
        self.preprocessed_test_samples = np.array(test_spectrograms)[:, np.newaxis, :, :]
        
        print(f"Order spectrogram shape - Train: {self.preprocessed_train_samples.shape}, Val: {self.preprocessed_val_samples.shape}, Test: {self.preprocessed_test_samples.shape}")
  
    def create_dataloaders(self):
        """Create PyTorch DataLoaders"""
        train_dataset = BearingDataset(self.preprocessed_train_samples, self.train_labels)
        val_dataset = BearingDataset(self.preprocessed_val_samples, self.val_labels)
        test_dataset = BearingDataset(self.preprocessed_test_samples, self.test_labels)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_dataloaders(self, processing_type='freq_spectrum'):
        """
        Main method to get dataloaders with specified preprocessing
        
        Args:
            processing_type: 'freq_spectrum', 'order_spectrum', or 'order_spectrogram'
        
        Returns:
            train_loader, val_loader, test_loader
        """
        print(f"Loading data with {processing_type} preprocessing...")
        self.import_data()

        if processing_type == 'freq_spectrum':
            self.freq_spectrum()
        elif processing_type == 'order_spectrum':
            self.order_spectrum()
        elif processing_type == 'order_spectrogram':
            self.order_spectrogram()
        else:
            raise ValueError(f"Unknown processing_type: {processing_type}")
        
        return self.create_dataloaders()

if __name__ == "__main__":
    # Example 1: Import raw data
    data_loader = CWRU_dataloader(batch_size=32)
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = data_loader.import_data()
    print(f"Raw data shapes - Train: {Xtrain.shape}, Val: {Xval.shape}, Test: {Xtest.shape}")
    
    # Example 2: Get dataloaders with frequency spectrum
    data_loader = CWRU_dataloader(batch_size=64)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(processing_type='freq_spectrum')
    
    # Example 3: Get dataloaders with order spectrum
    data_loader = CWRU_dataloader(batch_size=32)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(processing_type='order_spectrum')
    
    # Example 4: Get dataloaders with order spectrogram
    data_loader = CWRU_dataloader(batch_size=16)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(processing_type='order_spectrogram')