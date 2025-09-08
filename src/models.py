import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 32, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 128, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class StudentModel_Improved(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel_Improved, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class StudentModel_With_FCDropout(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2): # Thêm dropout_rate
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Tách phần classifier ra để thêm Dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate), # <-- THÊM DROPOUT Ở ĐÂY
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

#--- 1D model---

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

        # Giữ nguyên 2 khối Conv nhưng giảm số kênh (ví dụ: giảm một nửa)
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

        # Có thể giữ lại classifier phức tạp hơn một chút hoặc đơn giản hóa nó
        # Ở đây ta đơn giản hóa bằng cách bỏ lớp Linear ẩn
        self.classifier = nn.Sequential(
            nn.Flatten(),      # 64 x 1 -> 64
            nn.Linear(64, num_classes)
            # Nếu muốn, bạn có thể thêm lại một lớp ẩn nhỏ hơn:
            # nn.Linear(64, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),
            # nn.Linear(64, num_classes)
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

# class Deeper1DCNN(nn.Module):
#     def __init__(self,
#                  input_channels: int,
#                  num_classes: int,
#                  dropout_rate: float = 0.5):
#         super().__init__()

#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2),

#             # Block 2
#             nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=2),

#             # Block 3 (MỚI)
#             nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
            
#             # Global Pooling
#             nn.AdaptiveMaxPool1d(output_size=1)
#         )

#         # Classifier phải được cập nhật để nhận đầu vào 256 channels
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             # Kích thước đầu vào của Linear layer bây giờ là 256
#             nn.Linear(256, 128), 
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

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
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # Block 3 (Lớp Conv cuối cùng)
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(inplace=True),
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


