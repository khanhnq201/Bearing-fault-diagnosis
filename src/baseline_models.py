import torch
import torch.nn as nn

# =====================================================================================
# Model 1: T-Model
# Paper: [3] Y.Wang, 2024, IEEE ITJ (Adaptive Knowledge Distillation-Based...)
# Input Shape: (batch_size, 1, 1024)
# =====================================================================================

class T_BasicBlock(nn.Module):
    """Khối cơ bản cho T-Model với kết nối tắt (residual connection)."""
    def __init__(self, in_channels, out_channels):
        super(T_BasicBlock, self).__init__()
        self.use_residual = in_channels == out_channels
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        if self.use_residual:
            out += identity
        return self.relu(out)

class T_Model(nn.Module):
    """Kiến trúc T-Model gồm 8 khối T_BasicBlock nối tiếp."""
    def __init__(self, input_channels=1, num_classes=3):
        super(T_Model, self).__init__()
        channels = [input_channels, 64, 64, 128, 128, 256, 256, 512, 512]
        
        layers = [T_BasicBlock(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
        self.features = nn.Sequential(*layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# =====================================================================================
# Model 2: WDCNN
# Paper: [3] Zhang et al., 2017, Sensors (A New Deep Learning Model for Fault Diagnosis...)
# Input Shape: (batch_size, 1, 2048)
# =====================================================================================

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

# =====================================================================================
# Model 3: ACDIN
# Paper: [4] Chen et al., 2018, Neurocomputing (ACDIN: Bridging the gap...)
# Input Shape: (batch_size, 1, 5120)
# =====================================================================================

class InceptionModuleACDIN(nn.Module):
    """Module Inception tùy chỉnh cho ACDIN với Atrous Convolution."""
    def __init__(self, in_channels, c_1x1, c_5x5_r, c_5x5, c_7x7_r, c_7x7, c_pool, dilation_rate=2):
        super(InceptionModuleACDIN, self).__init__()
        padding_5x5 = (dilation_rate * (5 - 1)) // 2
        padding_7x7 = (dilation_rate * (7 - 1)) // 2
        
        self.branch1 = nn.Conv1d(in_channels, c_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, c_5x5_r, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv1d(c_5x5_r, c_5x5, kernel_size=5, padding=padding_5x5, dilation=dilation_rate)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, c_7x7_r, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv1d(c_7x7_r, c_7x7, kernel_size=7, padding=padding_7x7, dilation=dilation_rate)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, c_pool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class ACDIN(nn.Module):
    """Kiến trúc ACDIN với các module Inception và bộ phân loại phụ."""
    def __init__(self, input_channels=1, num_classes=3):
        super(ACDIN, self).__init__()
        # Các lớp ban đầu
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(3, stride=3)

        # Các khối Inception
        self.inception1a = InceptionModuleACDIN(128, 16, 16, 64, 16, 64, 32) # out: 176
        self.inception1b = InceptionModuleACDIN(176, 16, 16, 64, 16, 64, 32) # out: 176
        self.pool3 = nn.MaxPool1d(3, stride=3)
        
        self.inception2a = InceptionModuleACDIN(176, 16, 16, 64, 16, 64, 32) # out: 176
        self.inception2b = InceptionModuleACDIN(176, 16, 16, 48, 16, 48, 32) # out: 144
        self.inception2c = InceptionModuleACDIN(144, 16, 16, 48, 16, 48, 32) # out: 144
        self.pool4 = nn.MaxPool1d(3, stride=3)
        
        self.inception3a = InceptionModuleACDIN(144, 16, 16, 64, 16, 96, 64) # out: 240
        self.inception3b = InceptionModuleACDIN(240, 16, 16, 128, 16, 128, 256) # out: 528
        self.pool5 = nn.MaxPool1d(2, stride=2)
        
        self.inception4a = InceptionModuleACDIN(528, 16, 16, 96, 16, 96, 128) # out: 336
        self.inception4b = InceptionModuleACDIN(336, 16, 16, 96, 16, 96, 128) # out: 336
        self.bn_final = nn.BatchNorm1d(336)
        
        # Các bộ phân loại
        self.flatten = nn.Flatten()
        self.pool_final = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(336, num_classes)
        
        # Bộ phân loại phụ (chỉ dùng khi training)
        self.aux1_pool = nn.AdaptiveAvgPool1d(1)
        self.aux1_fc = nn.Linear(176, num_classes)
        self.aux2_pool = nn.AdaptiveAvgPool1d(1)
        self.aux2_fc = nn.Linear(528, num_classes)

    def forward(self, x):
        # Base
        x = self.pool2(self.conv2(self.pool1(self.conv1(x))))
        
        # Inception 1 & Aux 1
        x = self.inception1b(self.inception1a(x))
        if self.training:
            aux1_out = self.aux1_fc(self.flatten(self.aux1_pool(x)))
        x = self.pool3(x)
        
        # Inception 2
        x = self.inception2c(self.inception2b(self.inception2a(x)))
        x = self.pool4(x)
        
        # Inception 3 & Aux 2
        x = self.inception3b(self.inception3a(x))
        if self.training:
            aux2_out = self.aux2_fc(self.flatten(self.aux2_pool(x)))
        x = self.pool5(x)
        
        # Inception 4 & Final Classifier
        x = self.bn_final(self.inception4b(self.inception4a(x)))
        x = self.flatten(self.pool_final(x))
        main_out = self.classifier(x)
        
        if self.training:
            return main_out, aux1_out, aux2_out
        return main_out
    
# =====================================================================================
# Model 4: MobileNetV3-Small
# =====================================================================================

class conv_block(nn.Module):
    """Khối Conv-BN-Activation tiêu chuẩn."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=None):
        super().__init__()
        padding = kernel_size // 2
        self.activation_fn = None
        if activation == 'ReLU':
            self.activation_fn = nn.ReLU6(inplace=True)
        elif activation == 'H_swish':
            self.activation_fn = nn.Hardswish(inplace=True)
        
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_channels)
        ]
        if self.activation_fn:
            layers.append(self.activation_fn)
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class SE_block(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, in_channel, ratio=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // ratio, in_channel, bias=False),
            nn.Hardsigmoid(inplace=True)
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class bottleneck(nn.Module):
    """Khối Bottleneck của MobileNetV3."""
    def __init__(self, in_channels, kernel_size, expansion_size, out_channels, use_attention, activation, stride):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv_block = nn.Sequential(
            conv_block(in_channels, expansion_size, 1, activation=activation),
            conv_block(expansion_size, expansion_size, kernel_size, stride=stride, groups=expansion_size, activation=activation),
            SE_block(expansion_size) if use_attention else nn.Identity(),
            conv_block(expansion_size, out_channels, 1) # Lớp cuối không có activation
        )

    def forward(self, x):
        res = self.conv_block(x)
        if self.use_residual:
            res += x
        return res

class MobileNetV3_small_1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super().__init__()
        
        # QUYẾT ĐỊNH 1: Lớp đầu tiên chấp nhận input_channels=1
        self.features = nn.Sequential(
            conv_block(input_channels, 16, kernel_size=3, stride=2, activation='H_swish'),
            bottleneck(16, 3, 16, 16, True, 'ReLU', 2),
            bottleneck(16, 3, 72, 24, False, 'ReLU', 2),
            bottleneck(24, 3, 88, 24, False, 'ReLU', 1),
            bottleneck(24, 5, 96, 40, True, 'H_swish', 2),
            bottleneck(40, 5, 240, 40, True, 'H_swish', 1),
            bottleneck(40, 5, 240, 40, True, 'H_swish', 1),
            bottleneck(40, 5, 120, 48, True, 'H_swish', 1),
            bottleneck(48, 5, 144, 48, True, 'H_swish', 1),
            bottleneck(48, 5, 288, 96, True, 'H_swish', 2),
            bottleneck(96, 5, 576, 96, True, 'H_swish', 1),
            bottleneck(96, 5, 576, 96, True, 'H_swish', 1),
            conv_block(96, 576, 1, activation='H_swish')
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # QUYẾT ĐỊNH 3: Sử dụng classifier tiêu chuẩn
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
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

            # Block 3 
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
        feature_maps = self.features(x)
        pooled_features = self.pool(feature_maps)
        flattened_features = torch.flatten(pooled_features, 1)
        output = self.classifier(flattened_features)
        
        return output