# src/preprocessing.py (Phiên bản cập nhật)

import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
    PowerTransformer
)

class DataScaler:
    """
    Một class bao đóng nhiều phương pháp chuẩn hóa dữ liệu cho dữ liệu chuỗi.
    Tuân thủ API của scikit-learn với các phương thức fit, transform, và fit_transform.
    
    Methods:
    - 'z_score': Standardization (khuyên dùng nhất).
    - 'min_max': Min-Max scaling.
    - 'robust': Scaling sử dụng median và IQR, chống lại outliers.
    - 'power': Power transformation (Yeo-Johnson) để dữ liệu gần phân phối chuẩn hơn.
    - 'rms': Normalization bằng giá trị Root Mean Square toàn cục.
    - 'peak': Normalization bằng giá trị đỉnh (max absolute value) toàn cục.
    - 'l2_norm': Normalization mỗi mẫu (sample) về vector đơn vị (L2 norm).
    - 'log': Logarithmic normalization, phù hợp cho dữ liệu phổ tần số.
    - 'simple_scale': Nhân với một hệ số không đổi (ví dụ: 1e2).
    """
    def __init__(self, method='z_score', **kwargs):
        """
        Khởi tạo Scaler.
        
        Args:
            method (str): Tên của phương pháp chuẩn hóa.
            **kwargs: Các tham số bổ sung.
                      - Cho 'min_max': feature_range=(0, 1)
                      - Cho 'log': epsilon=1e-12
                      - Cho 'simple_scale': scale_factor=100.0
        """
        self.method = method
        self.kwargs = kwargs
        self.scaler = None
        self._validate_method()

    def _validate_method(self):
        # <<< ADDED 'simple_scale'
        valid_methods = ['z_score', 'min_max', 'robust', 'power', 'rms', 
                         'peak', 'l2_norm', 'log', 'simple_scale']
        if self.method not in valid_methods:
            raise ValueError(f"Phương pháp '{self.method}' không hợp lệ. "
                             f"Các phương pháp được hỗ trợ: {valid_methods}")

    def fit(self, data: np.ndarray):
        """
        Học các tham số chuẩn hóa (ví dụ: mean, std) từ dữ liệu training.
        Đối với 'simple_scale', nó chỉ lưu lại hệ số nhân.
        
        Args:
            data (np.ndarray): Dữ liệu training, shape (num_samples, sequence_length).
        """
        if self.method == 'z_score':
            self.scaler = StandardScaler()
            self.scaler.fit(data)
        elif self.method == 'min_max':
            self.scaler = MinMaxScaler(**self.kwargs)
            self.scaler.fit(data)
        elif self.method == 'robust':
            self.scaler = RobustScaler()
            self.scaler.fit(data)
        elif self.method == 'power':
            # Xác định các cột có phương sai gần bằng 0
            variances = np.var(data, axis=0)
            threshold = 1e-7  # Ngưỡng phương sai
            
            good_cols_idx = np.where(variances > threshold)[0]
            bad_cols_idx = np.where(variances <= threshold)[0]
            
            # Tạo scaler riêng cho từng nhóm cột
            scaler_good = PowerTransformer(method='yeo-johnson', standardize=True)
            scaler_bad = StandardScaler() # Dùng scaler an toàn cho các cột xấu
            
            # Fit scaler trên dữ liệu tương ứng
            if len(good_cols_idx) > 0:
                scaler_good.fit(data[:, good_cols_idx])
            if len(bad_cols_idx) > 0:
                scaler_bad.fit(data[:, bad_cols_idx])
            
            # Lưu tất cả thông tin cần thiết
            self.scaler = {
                'scaler_good': scaler_good,
                'scaler_bad': scaler_bad,
                'good_cols_idx': good_cols_idx,
                'bad_cols_idx': bad_cols_idx
            }
        elif self.method == 'l2_norm':
            self.scaler = Normalizer(norm='l2')
        elif self.method == 'rms':
            rms_val = np.sqrt(np.mean(data**2))
            self.scaler = rms_val if rms_val != 0 else 1.0
        elif self.method == 'peak':
            peak_val = np.max(np.abs(data))
            self.scaler = peak_val if peak_val != 0 else 1.0
        elif self.method == 'log':
            epsilon = self.kwargs.get('epsilon', 1e-12)
            log_data = np.log10(data + epsilon)
            self.scaler = {'mean': np.mean(log_data), 'std': np.std(log_data), 'epsilon': epsilon}
        # <<< ADDED BLOCK for simple_scale
        elif self.method == 'simple_scale':
            # Lấy scale_factor từ kwargs, mặc định là 100.0 (1e2)
            self.scaler = self.kwargs.get('scale_factor', 100.0)
        
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Áp dụng phép biến đổi đã học lên dữ liệu.
        
        Args:
            data (np.ndarray): Dữ liệu cần biến đổi.
        
        Returns:
            np.ndarray: Dữ liệu đã được chuẩn hóa.
        """
        if self.scaler is None:
            raise RuntimeError("Scaler chưa được fit. Vui lòng gọi fit() trên dữ liệu training trước.")

        if self.method in ['z_score', 'min_max', 'robust', 'l2_norm']:
            return self.scaler.transform(data)
        if self.method == 'power':
            # Tạo một mảng trống để chứa kết quả
            transformed_data = np.zeros_like(data, dtype=float)
            
            # Lấy thông tin đã lưu từ bước fit
            good_cols_idx = self.scaler['good_cols_idx']
            bad_cols_idx = self.scaler['bad_cols_idx']
            
            # Transform từng nhóm cột và đặt vào đúng vị trí
            if len(good_cols_idx) > 0:
                transformed_data[:, good_cols_idx] = self.scaler['scaler_good'].transform(data[:, good_cols_idx])
            if len(bad_cols_idx) > 0:
                transformed_data[:, bad_cols_idx] = self.scaler['scaler_bad'].transform(data[:, bad_cols_idx])

            return transformed_data
        elif self.method in ['rms', 'peak']:
            return data / self.scaler
        elif self.method == 'log':
            log_data = np.log10(data + self.scaler['epsilon'])
            return (log_data - self.scaler['mean']) / self.scaler['std']
        # <<< ADDED BLOCK for simple_scale
        elif self.method == 'simple_scale':
            return data * self.scaler

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Thực hiện fit và transform trên cùng một dữ liệu.
        """
        self.fit(data)
        return self.transform(data)