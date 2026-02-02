"""
Convolutional Neural Network (CNN) for spectrogram-based music classification
"""

import torch
import torch.nn as nn


class SpectrogramCNN(nn.Module):
    """
    CNN for classifying music from mel spectrograms.
    
    Architecture inspired by image classification CNNs adapted for audio.
    
    Args:
        num_classes (int): Number of countries to classify (default: 57)
        input_channels (int): Number of input channels (default: 1 for mono)
        conv_channels (list): Number of filters in each conv layer (default: [32, 64, 128, 256])
        kernel_size (int): Size of convolutional kernels (default: 3)
        pool_size (int): Size of max pooling windows (default: 2)
        fc_hidden_dim (int): Hidden dimension for fully connected layer (default: 128)
        dropout (float): Dropout probability (default: 0.5)
    """
    
    def __init__(
        self,
        num_classes=57,
        input_channels=1,
        conv_channels=[32, 64, 128, 256],
        kernel_size=3,
        pool_size=2,
        fc_hidden_dim=128,
        dropout=0.5
    ):
        super(SpectrogramCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(conv_channels):
            # Convolutional block
            conv_layers.extend([
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    padding=kernel_size//2
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size, pool_size)
            ])
            in_channels = out_channels
        
        # Global average pooling to handle variable input sizes
        conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[-1], fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input spectrogram of shape (batch_size, 1, height, width)
                              e.g., (32, 1, 128, 1300) for mel spectrogram
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # Convolutional feature extraction
        features = self.conv_layers(x)
        
        # Classification
        logits = self.fc_layers(features)
        
        return logits
    
    def get_conv_features(self, x):
        """
        Extract convolutional features before fully connected layers.
        Useful for visualization and transfer learning.
        
        Args:
            x (torch.Tensor): Input spectrogram
            
        Returns:
            torch.Tensor: Convolutional features
        """
        return self.conv_layers(x)


def create_cnn_model(config):
    """
    Factory function to create CNN model from config.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        SpectrogramCNN: Instantiated model
    """
    model = SpectrogramCNN(
        num_classes=config['training']['num_classes'],
        input_channels=config['model']['cnn']['input_channels'],
        conv_channels=config['model']['cnn']['conv_channels'],
        kernel_size=config['model']['cnn']['kernel_size'],
        pool_size=config['model']['cnn']['pool_size'],
        fc_hidden_dim=config['model']['cnn']['fc_hidden_dim'],
        dropout=config['model']['cnn']['dropout']
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing SpectrogramCNN...")
    
    # Create dummy spectrogram data
    # Typical mel spectrogram: 128 mel bins × ~1300 time frames for 30 seconds
    batch_size = 8
    channels = 1
    height = 128  # mel bins
    width = 1300  # time frames
    num_classes = 57
    
    x = torch.randn(batch_size, channels, height, width)
    
    # Create model
    model = SpectrogramCNN(
        num_classes=num_classes,
        input_channels=channels,
        conv_channels=[32, 64, 128, 256],
        dropout=0.5
    )
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test conv features extraction
    features = model.get_conv_features(x)
    print(f"Conv features shape: {features.shape}")
    
    # Test with different input size (adaptive pooling handles this)
    x_small = torch.randn(batch_size, channels, 128, 650)
    logits_small = model(x_small)
    print(f"Small input shape: {x_small.shape}")
    print(f"Small output shape: {logits_small.shape}")
    
    print("\n✓ Model test passed!")
