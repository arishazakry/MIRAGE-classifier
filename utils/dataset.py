"""
PyTorch Dataset and DataLoader utilities for music geographic classification
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os


class MusicFeatureDataset(Dataset):
    """
    Dataset for music features (e.g., Spotify API features).
    
    Args:
        features (np.ndarray or torch.Tensor): Feature matrix of shape (n_samples, n_features)
        labels (np.ndarray or torch.Tensor): Labels of shape (n_samples,)
        label_encoder (LabelEncoder, optional): Label encoder for country names
    """
    
    def __init__(self, features, labels, label_encoder=None):
        if isinstance(features, np.ndarray):
            self.features = torch.FloatTensor(features)
        else:
            self.features = features.float()
        
        if isinstance(labels, np.ndarray):
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = labels.long()
        
        self.label_encoder = label_encoder
        
        assert len(self.features) == len(self.labels), \
            f"Features and labels must have same length: {len(self.features)} vs {len(self.labels)}"
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def get_label_name(self, label_idx):
        """Convert label index to country name"""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform([label_idx])[0]
        return str(label_idx)


class MusicSpectrogramDataset(Dataset):
    """
    Dataset for mel spectrograms (for CNN models).
    
    Args:
        spectrograms (np.ndarray or torch.Tensor): Spectrograms of shape (n_samples, height, width)
        labels (np.ndarray or torch.Tensor): Labels of shape (n_samples,)
        label_encoder (LabelEncoder, optional): Label encoder for country names
        transform (callable, optional): Optional transform to be applied
    """
    
    def __init__(self, spectrograms, labels, label_encoder=None, transform=None):
        if isinstance(spectrograms, np.ndarray):
            self.spectrograms = torch.FloatTensor(spectrograms)
        else:
            self.spectrograms = spectrograms.float()
        
        if isinstance(labels, np.ndarray):
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = labels.long()
        
        self.label_encoder = label_encoder
        self.transform = transform
        
        # Ensure spectrograms have channel dimension
        if len(self.spectrograms.shape) == 3:
            # Add channel dimension: (N, H, W) -> (N, 1, H, W)
            self.spectrograms = self.spectrograms.unsqueeze(1)
    
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        label = self.labels[idx]
        
        if self.transform:
            spec = self.transform(spec)
        
        return spec, label


class MusicSequenceDataset(Dataset):
    """
    Dataset for sequential features (for RNN models).
    
    Args:
        sequences (np.ndarray or torch.Tensor): Sequences of shape (n_samples, seq_len, n_features)
        labels (np.ndarray or torch.Tensor): Labels of shape (n_samples,)
        label_encoder (LabelEncoder, optional): Label encoder for country names
    """
    
    def __init__(self, sequences, labels, label_encoder=None):
        if isinstance(sequences, np.ndarray):
            self.sequences = torch.FloatTensor(sequences)
        else:
            self.sequences = sequences.float()
        
        if isinstance(labels, np.ndarray):
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = labels.long()
        
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_mirage_data(csv_path, remove_duplicates=True):
    """
    Load MIRAGE corpus data.
    
    Args:
        csv_path (str): Path to MIRAGE CSV file
        remove_duplicates (bool): Whether to remove duplicate songs (same artist + track)
        
    Returns:
        pd.DataFrame: Loaded data
    """
    df = pd.read_csv(csv_path)
    
    # Basic cleaning
    df = df.dropna(subset=['country', 'artist', 'track'])
    
    if remove_duplicates:
        # Remove duplicates based on artist and track
        df = df.drop_duplicates(subset=['artist', 'track'], keep='first')
        print(f"After removing duplicates: {len(df)} unique songs")
    
    return df


def create_label_encoders(df):
    """
    Create label encoders for geographic labels.
    
    Args:
        df (pd.DataFrame): DataFrame with 'country', 'region', 'continent' columns
        
    Returns:
        dict: Dictionary of label encoders
    """
    encoders = {}
    
    # Country encoder
    if 'country' in df.columns:
        country_encoder = LabelEncoder()
        df['country_label'] = country_encoder.fit_transform(df['country'])
        encoders['country'] = country_encoder
    
    # Region encoder
    if 'region' in df.columns:
        region_encoder = LabelEncoder()
        df['region_label'] = region_encoder.fit_transform(df['region'])
        encoders['region'] = region_encoder
    
    # Continent encoder
    if 'continent' in df.columns:
        continent_encoder = LabelEncoder()
        df['continent_label'] = continent_encoder.fit_transform(df['continent'])
        encoders['continent'] = continent_encoder
    
    return df, encoders


def create_train_val_test_splits(
    df, 
    features, 
    label_col='country_label',
    train_size=0.7, 
    val_size=0.1, 
    test_size=0.2,
    random_state=42,
    stratify=True
):
    """
    Create train/validation/test splits.
    
    Args:
        df (pd.DataFrame): DataFrame with labels
        features (np.ndarray): Feature matrix
        label_col (str): Name of label column
        train_size (float): Proportion for training
        val_size (float): Proportion for validation
        test_size (float): Proportion for testing
        random_state (int): Random seed
        stratify (bool): Whether to stratify by labels
        
    Returns:
        tuple: (train_features, val_features, test_features,
                train_labels, val_labels, test_labels)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split proportions must sum to 1.0"
    
    labels = df[label_col].values
    
    stratify_array = labels if stratify else None
    
    # First split: train+val vs test
    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_array
    )
    
    # Second split: train vs val
    val_proportion = val_size / (train_size + val_size)
    stratify_train_val = train_val_labels if stratify else None
    
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_val_features,
        train_val_labels,
        test_size=val_proportion,
        random_state=random_state,
        stratify=stratify_train_val
    )
    
    print(f"Train: {len(train_features)} samples")
    print(f"Val: {len(val_features)} samples")
    print(f"Test: {len(test_features)} samples")
    
    return train_features, val_features, test_features, train_labels, val_labels, test_labels


def compute_class_weights(labels, num_classes):
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        labels (np.ndarray): Array of labels
        num_classes (int): Total number of classes
        
    Returns:
        torch.Tensor: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    
    # Create full weight tensor (including classes not in training set)
    weights = torch.ones(num_classes)
    weights[unique_labels] = torch.FloatTensor(class_weights)
    
    return weights


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
):
    """
    Create PyTorch DataLoaders.
    
    Args:
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        test_dataset (Dataset): Test dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for batch norm stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def save_datasets(
    train_features, val_features, test_features,
    train_labels, val_labels, test_labels,
    label_encoder,
    save_dir
):
    """Save processed datasets to disk"""
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'train_features.npy'), train_features)
    np.save(os.path.join(save_dir, 'val_features.npy'), val_features)
    np.save(os.path.join(save_dir, 'test_features.npy'), test_features)
    
    np.save(os.path.join(save_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(save_dir, 'val_labels.npy'), val_labels)
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)
    
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Datasets saved to {save_dir}")


def load_datasets(load_dir):
    """Load processed datasets from disk"""
    train_features = np.load(os.path.join(load_dir, 'train_features.npy'))
    val_features = np.load(os.path.join(load_dir, 'val_features.npy'))
    test_features = np.load(os.path.join(load_dir, 'test_features.npy'))
    
    train_labels = np.load(os.path.join(load_dir, 'train_labels.npy'))
    val_labels = np.load(os.path.join(load_dir, 'val_labels.npy'))
    test_labels = np.load(os.path.join(load_dir, 'test_labels.npy'))
    
    with open(os.path.join(load_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Datasets loaded from {load_dir}")
    
    return (train_features, val_features, test_features,
            train_labels, val_labels, test_labels,
            label_encoder)


if __name__ == "__main__":
    # Test dataset classes
    print("Testing dataset classes...")
    
    # Create dummy data
    n_samples = 1000
    n_features = 13
    num_classes = 57
    
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, num_classes, n_samples)
    
    # Test MusicFeatureDataset
    dataset = MusicFeatureDataset(features, labels)
    print(f"Feature Dataset length: {len(dataset)}")
    
    sample_features, sample_label = dataset[0]
    print(f"Sample features shape: {sample_features.shape}")
    print(f"Sample label: {sample_label}")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_features, batch_labels = next(iter(loader))
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    print("\n✓ Dataset test passed!")
