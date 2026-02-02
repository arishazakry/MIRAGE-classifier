"""
Recurrent Neural Network (RNN) with LSTM for sequential music classification
"""

import torch
import torch.nn as nn


class MusicRNN(nn.Module):
    """
    LSTM-based model for classifying music from sequential features.
    
    Args:
        input_dim (int): Dimension of input features at each time step
        num_classes (int): Number of countries to classify (default: 57)
        hidden_dim (int): Hidden dimension of LSTM (default: 128)
        num_layers (int): Number of LSTM layers (default: 2)
        dropout (float): Dropout probability (default: 0.3)
        bidirectional (bool): Whether to use bidirectional LSTM (default: True)
    """
    
    def __init__(
        self,
        input_dim,
        num_classes=57,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        super(MusicRNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Input: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize LSTM and FC weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        # c_n: (num_layers * num_directions, batch, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from both directions (if bidirectional)
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            forward_hidden = h_n[-2, :, :]  # Forward direction
            backward_hidden = h_n[-1, :, :]  # Backward direction
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # Use final hidden state from last layer
            final_hidden = h_n[-1, :, :]
        
        # Classification
        logits = self.fc(final_hidden)
        
        return logits
    
    def get_lstm_outputs(self, x):
        """
        Get LSTM outputs at all time steps.
        Useful for attention mechanisms or visualization.
        
        Args:
            x (torch.Tensor): Input sequence
            
        Returns:
            torch.Tensor: LSTM outputs of shape (batch, seq_len, hidden_dim * num_directions)
        """
        lstm_out, _ = self.lstm(x)
        return lstm_out


class MusicAttentionRNN(nn.Module):
    """
    LSTM with self-attention mechanism for music classification.
    
    Attention allows the model to focus on important time steps in the sequence.
    
    Args:
        input_dim (int): Dimension of input features at each time step
        num_classes (int): Number of countries to classify (default: 57)
        hidden_dim (int): Hidden dimension of LSTM (default: 128)
        num_layers (int): Number of LSTM layers (default: 2)
        dropout (float): Dropout probability (default: 0.3)
        bidirectional (bool): Whether to use bidirectional LSTM (default: True)
        attention_type (str): Type of attention ('self', 'additive', 'dot') (default: 'self')
    """
    
    def __init__(
        self,
        input_dim,
        num_classes=57,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        attention_type='self'
    ):
        super(MusicAttentionRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.attention_type = attention_type
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * self.num_directions
        
        if attention_type == 'self':
            self.attention = SelfAttention(lstm_output_dim)
        elif attention_type == 'additive':
            self.attention = AdditiveAttention(lstm_output_dim)
        elif attention_type == 'dot':
            self.attention = DotProductAttention(lstm_output_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
            return_attention (bool): Whether to return attention weights
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
            torch.Tensor (optional): Attention weights if return_attention=True
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Classification
        logits = self.fc(context)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits


class SelfAttention(nn.Module):
    """Self-attention mechanism"""
    
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output (torch.Tensor): (batch, seq_len, hidden_dim)
            
        Returns:
            context (torch.Tensor): (batch, hidden_dim)
            attention_weights (torch.Tensor): (batch, seq_len)
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_dim)
        
        return context, attention_weights.squeeze(-1)


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism"""
    
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        # Query is mean of LSTM outputs
        query = torch.mean(lstm_output, dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        
        # Compute attention scores
        scores = self.v(torch.tanh(self.W1(query) + self.W2(lstm_output)))
        attention_weights = torch.softmax(scores, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context, attention_weights.squeeze(-1)


class DotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism"""
    
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.scale = hidden_dim ** 0.5
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, lstm_output):
        # Query is projection of mean
        query = self.query_proj(torch.mean(lstm_output, dim=1, keepdim=True))
        
        # Compute scaled dot product
        scores = torch.bmm(query, lstm_output.transpose(1, 2)) / self.scale
        attention_weights = torch.softmax(scores, dim=2)
        
        # Weighted sum
        context = torch.bmm(attention_weights, lstm_output).squeeze(1)
        
        return context, attention_weights.squeeze(1)


def create_rnn_model(config):
    """Factory function to create RNN model from config"""
    model = MusicRNN(
        input_dim=config['model']['rnn']['input_dim'],
        num_classes=config['training']['num_classes'],
        hidden_dim=config['model']['rnn']['hidden_dim'],
        num_layers=config['model']['rnn']['num_layers'],
        dropout=config['model']['rnn']['dropout'],
        bidirectional=config['model']['rnn']['bidirectional']
    )
    return model


def create_attention_model(config):
    """Factory function to create Attention-based RNN model from config"""
    model = MusicAttentionRNN(
        input_dim=config['model']['attention']['input_dim'],
        num_classes=config['training']['num_classes'],
        hidden_dim=config['model']['attention']['hidden_dim'],
        num_layers=config['model']['attention']['num_layers'],
        dropout=config['model']['attention']['dropout'],
        bidirectional=config['model']['attention']['bidirectional'],
        attention_type=config['model']['attention']['attention_type']
    )
    return model


if __name__ == "__main__":
    print("Testing MusicRNN...")
    
    # Test data
    batch_size = 16
    seq_len = 600  # 600 frames for 30 seconds at 50ms per frame
    input_dim = 13
    num_classes = 57
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test RNN
    rnn_model = MusicRNN(input_dim=input_dim, num_classes=num_classes)
    rnn_logits = rnn_model(x)
    print(f"RNN Input: {x.shape}, Output: {rnn_logits.shape}")
    print(f"RNN Parameters: {sum(p.numel() for p in rnn_model.parameters()):,}")
    
    # Test Attention RNN
    attention_model = MusicAttentionRNN(input_dim=input_dim, num_classes=num_classes)
    attn_logits, attn_weights = attention_model(x, return_attention=True)
    print(f"\nAttention RNN Input: {x.shape}, Output: {attn_logits.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention RNN Parameters: {sum(p.numel() for p in attention_model.parameters()):,}")
    
    print("\n✓ Models test passed!")
