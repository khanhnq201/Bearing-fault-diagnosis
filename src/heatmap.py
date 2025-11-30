import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple, Union
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")

# from data_loader import CWRU_dataloader
# import models
# from utils import load_config
from src.utils import load_config
from src.data_loader import CWRU_dataloader
from src import models

import os

class GradCAM1D:
    """
    Grad-CAM implementation for 1D CNN models.
    Visualizes which parts of the input signal are most important for classification.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize Grad-CAM for 1D signals.
        
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer. If None, uses the last Conv1d layer
        """
        self.model = model
        self.model.eval()
        
        # Find target layer
        if target_layer is None:
            self.target_layer = self._find_last_conv_layer()
        else:
            self.target_layer = target_layer
            
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _find_last_conv_layer(self) -> nn.Module:
        """Find the last Conv1d layer in the model."""
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                conv_layers.append((name, module))
        
        if not conv_layers:
            raise ValueError("No Conv1d layer found in the model")
        
        last_conv_name, last_conv = conv_layers[-1]
        print(f"Using layer: {last_conv_name}")
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, 
                     input_signal: torch.Tensor, 
                     target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input signal.
        
        Args:
            input_signal: Input tensor of shape [batch_size, channels, length]
            target_class: Target class index. If None, uses predicted class
            
        Returns:
            cam: Grad-CAM heatmap of shape [batch_size, length]
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_signal)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.tensor([target_class] * input_signal.size(0)).to(input_signal.device)
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate Grad-CAM
        gradients = self.gradients  # [batch, channels, length]
        activations = self.activations  # [batch, channels, length]
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=2, keepdim=True)  # [batch, channels, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1)  # [batch, length]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        for i in range(cam.shape[0]):
            if cam[i].max() > 0:
                cam[i] = cam[i] / cam[i].max()
        
        # Resize to input length
        input_length = input_signal.shape[2]
        cam_resized = np.zeros((cam.shape[0], input_length))
        for i in range(cam.shape[0]):
            cam_resized[i] = np.interp(
                np.linspace(0, cam.shape[1] - 1, input_length),
                np.arange(cam.shape[1]),
                cam[i]
            )
        
        return cam_resized


class HeatmapVisualizer:
    """
    Visualize Grad-CAM heatmaps overlayed on 1D signals.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize visualizer.
        
        Args:
            class_names: List of class names for display
        """
        self.class_names = class_names
        
    def plot_signal_with_heatmap(self,
                                  signals: Union[torch.Tensor, np.ndarray],
                                  heatmaps: np.ndarray,
                                  predictions: Optional[torch.Tensor] = None,
                                  confidences: Optional[torch.Tensor] = None,
                                  true_labels: Optional[List[int]] = None,
                                  x_axis: Optional[np.ndarray] = None,
                                  titles: Optional[List[str]] = None,
                                  figsize: Tuple[int, int] = (12, 4),
                                  save_path: Optional[str] = None,
                                  xlabel: str = "Sample Index",
                                  ylabel: str = "Amplitude"):
        """
        Plot multiple signals with Grad-CAM heatmap overlay.
        
        Args:
            signals: Input signals [batch_size, channels, length] or [batch_size, length]
            heatmaps: Grad-CAM heatmaps [batch_size, length]
            predictions: Predicted class indices [batch_size]
            confidences: Prediction confidences [batch_size]
            true_labels: True class labels (optional)
            x_axis: Custom x-axis values (e.g., frequency bins, time)
            titles: Custom titles for each subplot
            figsize: Figure size
            save_path: Path to save figure
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        # Convert to numpy
        if isinstance(signals, torch.Tensor):
            signals = signals.detach().cpu().numpy()
        
        # Handle shape
        if signals.ndim == 3:
            signals = signals[:, 0, :]  # Take first channel
        
        batch_size = signals.shape[0]
        # n_cols = min(3, batch_size)
        # n_rows = (batch_size + n_cols - 1) // n_cols
        
        # fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # if batch_size == 1:
        #     axes = np.array([axes])
        # axes = axes.flatten()
        
        # Create custom colormap (transparent to red)
        colors = ['#ffffff00', '#ff000033', '#ff0000aa', '#ff0000ff']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # X-axis
        if x_axis is None:
            x_axis = np.arange(signals.shape[1])
        
        for idx in range(batch_size):
            # ax = axes[idx]
            plt.figure(figsize=figsize)

            signal = signals[idx]
            heatmap = heatmaps[idx]
            
            # Plot signal
            plt.plot(x_axis, signal, color='#2c3e50', linewidth=0.8, zorder=1)
            ax = plt.gca() #added
            y_min_plot, y_max_plot = ax.get_ylim() 
            # Overlay heatmap
            # Normalize signal for color mapping
            signal_min, signal_max = signal.min(), signal.max()
            if signal_max > signal_min:
                signal_norm = (signal - signal_min) / (signal_max - signal_min)
            else:
                signal_norm = np.zeros_like(signal)
            
            # Create filled area with heatmap colors
            for i in range(len(x_axis) - 1):
                alpha = heatmap[i]
                if alpha > 0.1:  # Only show significant activations
                    ax.fill_between(
                        x_axis[i:i+2],
                        y_min_plot,
                        y_max_plot,
                        color=cmap(heatmap[i]),
                        zorder=0
                    )
            
            # Title
            if titles is not None:
                title = titles[idx]
            else:
                title = f"Sample {idx + 1}"
                
                if predictions is not None:
                    pred_class = predictions[idx].item()
                    pred_name = self.class_names[pred_class] if self.class_names else f"Class {pred_class}"
                    title += f" | Pred: {pred_name}"
                    
                    if confidences is not None:
                        conf = confidences[idx].item() * 100
                        title += f" ({conf:.1f}%)"
                
                if true_labels is not None:
                    true_class = true_labels[idx]
                    true_name = self.class_names[true_class] if self.class_names else f"Class {true_class}"
                    title += f"\nTrue: {true_name}"
                    
                    # Add checkmark or cross
                    if predictions is not None:
                        if pred_class == true_class:
                            title += " ✓"
                        else:
                            title += " ✗"
            plt.title(title, fontsize=10, fontweight='bold')
            plt.xlabel(xlabel) # Font size mặc định
            plt.ylabel(ylabel) # Font size mặc định
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        #     ax.set_title(title, fontsize=10, fontweight='bold')
        #     ax.set_xlabel(xlabel, fontsize=9)
        #     ax.set_ylabel(ylabel, fontsize=9)
        #     ax.grid(True, alpha=0.3, linestyle='--')
        #     ax.spines['top'].set_visible(False)
        #     ax.spines['right'].set_visible(False)
        
        # # Hide empty subplots
        # for idx in range(batch_size, len(axes)):
        #     axes[idx].axis('off')
        
        # plt.tight_layout()
        
        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     print(f"Figure saved to {save_path}")
        
        # plt.show()


# Example usage
if __name__ == "__main__":
    config = load_config('config/default_config.yaml')
    model_path = "experiments/models/best_model.pth"   # đường dẫn file model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    model = models.SimpleCNN1D(1, num_classes).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    print(f"✅ Loaded model from: {model_path}")

    data_loader = CWRU_dataloader(config['dataloader'])
    X, Y, x_axis = data_loader.get_samples_for_heatmap(processing_type=config['dataloader']['preprocessing_type'], data_list = config['heatmap']['data_list'], num_samples = 3)

    gradcam = GradCAM1D(model)
    
    # Get predictions
    tensor = torch.as_tensor(X, dtype=torch.float32)
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)
        confidences = probabilities.max(dim=1)[0]
    
    # Generate heatmaps
    heatmaps = gradcam.generate_cam(tensor)
    
    # Visualize
    class_names = ['Normal', 'IR', 'OR']
    visualizer = HeatmapVisualizer(class_names=class_names)
    
    fname = config['heatmap']['name']
    save_dir = os.path.join(".", "experiments", "heatmaps")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, fname)
    visualizer.plot_signal_with_heatmap(
        signals=tensor,
        heatmaps=heatmaps,
        predictions=predictions,
        confidences=confidences,
        true_labels=Y,
        x_axis=x_axis,
        xlabel="Frequency (Hz)",
        ylabel="Magnitude",
        save_path=save_path
    )
    
    print("\nVisualization complete!")
    print(f"Target layer: {gradcam.target_layer.__class__.__name__}")

