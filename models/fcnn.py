"""
Fully Connected Neural Network (FCNN) for music geographic classification
"""

import torch
import torch.nn as nn


class MusicClassifierFC(nn.Module):
    """
    Fully connected neural network for classifying music by geographic origin.
    
    Args:
        input_dim (int): Number of input features (e.g., 13 for Spotify features)
        num_classes (int): Number of countries to classify (default: 57)
        hidden_dims (list): List of hidden layer dimensions (default: [256, 128, 64])
        dropout (float): Dropout probability (default: 0.3)
        use_batch_norm (bool): Whether to use batch normalization (default: True)
        activation (str): Activation function ('relu', 'leaky_relu', 'elu') (default: 'relu')
    """
    
    def __init__(
        self,
        input_dim,
        num_classes=57,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        use_batch_norm=True,
        activation='relu'
    ):
        super(MusicClassifierFC, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation - will use CrossEntropyLoss which includes softmax)
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Combine all layers into sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def get_embeddings(self, x):
        """
        Extract learned representations from the penultimate layer.
        Useful for visualization (t-SNE, UMAP) and interpretability.
        
        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Embeddings from second-to-last layer
        """
        # Forward through all layers except the last one
        for layer in self.network[:-1]:
            x = layer(x)
        return x


def create_fcnn_model(config):
    """
    Factory function to create FCNN model from config.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        MusicClassifierFC: Instantiated model
    """
    model = MusicClassifierFC(
        input_dim=config['model']['fcnn']['input_dim'],
        num_classes=config['training']['num_classes'],
        hidden_dims=config['model']['fcnn']['hidden_dims'],
        dropout=config['model']['fcnn']['dropout'],
        use_batch_norm=config['model']['fcnn']['use_batch_norm'],
        activation=config['model']['fcnn']['activation']
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing MusicClassifierFC...")
    
    # Create dummy data
    batch_size = 32
    input_dim = 13
    num_classes = 57
    
    x = torch.randn(batch_size, input_dim)
    
    # Create model
    model = MusicClassifierFC(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[256, 128, 64],
        dropout=0.3
    )
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test embeddings extraction
    embeddings = model.get_embeddings(x)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test with probabilities
    probs = torch.softmax(logits, dim=1)
    print(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(batch_size))}")
    
    print("\n✓ Model test passed!")
