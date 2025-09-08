# src/augmentation.py (Phiên bản cập nhật)

import torch
import random

class SingleAugmentation:
    """
    Áp dụng ngẫu nhiên CHỈ MỘT phép tăng cường dữ liệu.
    
    Args:
        methods (list): ['noise', 'shift', 'scale', 'mask', 'snr_noise']
        device: Thiết bị ('cpu' hoặc 'cuda').
        noise_std (float): Dùng cho method 'noise'.
        target_snr_db (float): Mức SNR (dB) mong muốn cho method 'snr_noise'.
        ... (các tham số khác)
    """
    def __init__(self, 
                 methods: list,
                 device,
                 noise_std=0.05,
                 target_snr_db=15.0, # <<< THAM SỐ MỚI
                 shift_max_pct=0.03,
                 scale_range=(0.9, 1.1),
                 mask_max_pct=0.1):
        
        self.device = device
        self.noise_std = noise_std
        self.target_snr_db = target_snr_db # <<< LƯU THAM SỐ MỚI
        self.shift_max_pct = shift_max_pct
        self.scale_range = scale_range
        self.mask_max_pct = mask_max_pct

        self.method_funcs = {
            'noise': self._add_gaussian_noise_relative,
            'shift': self._time_shift,
            'scale': self._amplitude_scaling,
            'mask': self._frequency_masking,
            'snr_noise': self._add_noise_snr # <<< PHƯƠNG PHÁP MỚI
        }
        
        self.transforms_to_apply = []
        for method_name in methods:
            if method_name in self.method_funcs:
                self.transforms_to_apply.append(self.method_funcs[method_name])
            else:
                print(f"[Warning] Phương pháp augmentation '{method_name}' không tồn tại.")
        
        if not self.transforms_to_apply:
            print("[Warning] Không có phương pháp augmentation nào được chọn.")

    # --- Các hàm augmentation ---

    def _add_noise_snr(self, signal_batch):
        """Thêm nhiễu Gaussian để đạt được một mức SNR (dB) mục tiêu."""
        # Tính công suất của tín hiệu cho mỗi mẫu trong batch
        # Power = mean(signal^2)
        signal_power = torch.mean(signal_batch**2, dim=2, keepdim=True)
        
        # Chuyển đổi SNR từ dB sang tỷ lệ tuyến tính
        snr_linear = 10**(self.target_snr_db / 10.0)
        
        # Tính công suất nhiễu cần thiết
        noise_power = signal_power / snr_linear
        
        # Tạo nhiễu Gaussian chuẩn (mean=0, std=1)
        # Công suất của nhiễu này xấp xỉ 1
        noise = torch.randn_like(signal_batch)
        
        # Scale nhiễu để có công suất mong muốn
        # Power_new = k^2 * Power_old => k = sqrt(Power_new / Power_old)
        # Vì Power_old ~ 1, k ~ sqrt(Power_new)
        noise_scaled = noise * torch.sqrt(noise_power)
        
        return signal_batch + noise_scaled

    def _add_gaussian_noise_relative(self, signal_batch):
        """Thêm nhiễu với std tương đối so với biên độ đỉnh."""
        max_vals = torch.max(torch.abs(signal_batch), dim=2, keepdim=True)[0] + 1e-6
        noise_magnitude = self.noise_std * max_vals
        noise = torch.randn(signal_batch.size(), device=self.device) * noise_magnitude
        return signal_batch + noise
    
    # ... (các hàm _time_shift, _amplitude_scaling, _frequency_masking không đổi) ...
    def _time_shift(self, signal_batch):
        seq_len = signal_batch.size(2)
        shift_max = int(seq_len * self.shift_max_pct)
        shift = torch.randint(-shift_max, shift_max, (1,)).item()
        return torch.roll(signal_batch, shifts=shift, dims=2)
    
    def _amplitude_scaling(self, signal_batch):
        scales = torch.empty(signal_batch.size(0), 1, 1, device=self.device).uniform_(*self.scale_range)
        return signal_batch * scales

    def _frequency_masking(self, signal_batch):
        augmented_batch = signal_batch.clone()
        seq_len = augmented_batch.size(2)
        for i in range(augmented_batch.size(0)):
            mask_size = torch.randint(1, int(seq_len * self.mask_max_pct), (1,)).item()
            mask_start = torch.randint(0, seq_len - mask_size, (1,)).item()
            augmented_batch[i, :, mask_start:mask_start + mask_size] = 0
        return augmented_batch

    def __call__(self, signal_batch):
        if not self.transforms_to_apply:
            return signal_batch
        transform_func = random.choice(self.transforms_to_apply)
        return transform_func(signal_batch)