import torch
import torch.nn as nn
import torch.nn.functional as F

class WDCNN(nn.Module):
    """Kiến trúc WDCNN với lớp conv đầu tiên rộng và các lớp sâu theo sau."""
    def __init__(self, input_channels=1, num_classes=3):
        super(WDCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, padding=0), # Padding 'No'
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Với input 2048, output của features là (N, 64, 3)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 100),
            nn.BatchNorm1d(100), nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class Basic1DCNN(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(output_size=1)   # Global-Max-Pooling
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 128 × 1  → 128
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, input_channels, timesteps)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

class Narrow_1DCNN(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 dropout_rate: float = 0.2): # Có thể giảm dropout
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=32,  # 64 -> 32
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32,
                      out_channels=64,  # 128 -> 64
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),      # 64 x 1 -> 64
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, input_channels, timesteps)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

class Wider1DCNN(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 dropout_rate: float = 0.5):
        super().__init__()

        # Tăng số channels từ (64, 128) -> (128, 256)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        # Classifier cũng cần được cập nhật
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), # Tăng kích thước lớp ẩn
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Deeper1DCNN(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 dropout_rate: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2),

            # Block 3 (Lớp Conv cuối cùng)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
        )
        
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, input_channels, timesteps)
        
        # Lấy feature maps từ các lớp conv
        feature_maps = self.features(x) # -> shape: (batch, 256, sequence_length_after_conv)
        
        # Áp dụng pooling
        pooled_features = self.pool(feature_maps) # -> shape: (batch, 256, 1)
        
        # Làm phẳng (Flatten)
        flattened_features = torch.flatten(pooled_features, 1) # -> shape: (batch, 256)
        
        # Đưa qua classifier
        output = self.classifier(flattened_features)
        
        return output

class SimpleCNN1D(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 dropout_rate: float = 0.5):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Use AvgPool instead of MaxPool
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        
        return x

class Baseline(nn.Module):
    def __init__(self, num_classes=3):
        super(Baseline, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        # Dùng LazyLinear - tự động tính input size
        self.fc1 = nn.LazyLinear(625)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(625, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x