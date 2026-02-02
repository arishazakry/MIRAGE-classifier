"""
Main training script for music geographic classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import yaml
import os
import sys
from pathlib import Path
import argparse
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.fcnn import create_fcnn_model
from models.cnn import create_cnn_model
from models.rnn import create_rnn_model, create_attention_model
from utils.dataset import (
    MusicFeatureDataset,
    create_dataloaders,
    compute_class_weights
)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create model based on config"""
    model_type = config['model']['type']
    
    if model_type == 'fcnn':
        model = create_fcnn_model(config)
    elif model_type == 'cnn':
        model = create_cnn_model(config)
    elif model_type == 'rnn':
        model = create_rnn_model(config)
    elif model_type == 'attention':
        model = create_attention_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_optimizer(model, config):
    """Create optimizer based on config"""
    optimizer_name = config['training']['optimizer']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=config['training']['adam_betas'],
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=config['training']['adam_betas'],
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config['training']['sgd_momentum'],
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler"""
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=scheduler_config['patience'],
            factor=scheduler_config['factor'],
            min_lr=scheduler_config['min_lr'],
            verbose=True
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, grad_clip=None):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100.0 * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc, np.array(all_preds), np.array(all_labels)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved: {path}')


def train(config, train_loader, val_loader, model, device):
    """Main training function"""
    
    # Loss function
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['training'].get('label_smoothing', 0.0)
    )
    
    # If using class weights for imbalanced data
    if config['training'].get('use_class_weights', False):
        # Assume we have precomputed class weights
        class_weights_path = os.path.join(
            config['data']['splits_dir'], 
            'class_weights.npy'
        )
        if os.path.exists(class_weights_path):
            class_weights = torch.FloatTensor(np.load(class_weights_path)).to(device)
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config['training'].get('label_smoothing', 0.0)
            )
            print("Using class weights for imbalanced data")
    
    # Optimizer
    optimizer = create_optimizer(model, config)
    
    # Scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Mixed precision scaler
    scaler = GradScaler() if config['training'].get('mixed_precision', False) else None
    
    # Early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_val_acc = 0.0
    checkpoint_dir = config['experiment']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device,
            scaler=scaler,
            grad_clip=config['training'].get('gradient_clip')
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model,
            val_loader,
            criterion,
            device
        )
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        if config['training']['scheduler']['type'] == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
            print(f'✓ New best model! Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 and not config['experiment'].get('save_best_only', True):
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("\nEarly stopping triggered!")
                break
    
    print(f'\nTraining complete!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train music geographic classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/splits/',
                        help='Directory with processed data splits')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed for reproducibility
    set_seed(config['reproducibility']['seed'])
    
    # Device
    device = torch.device(
        config['hardware']['device'] 
        if torch.cuda.is_available() 
        else 'cpu'
    )
    print(f'Using device: {device}')
    
    # Load data (assuming preprocessed)
    print('\nLoading data...')
    train_features = np.load(os.path.join(args.data_dir, 'train_features.npy'))
    val_features = np.load(os.path.join(args.data_dir, 'val_features.npy'))
    train_labels = np.load(os.path.join(args.data_dir, 'train_labels.npy'))
    val_labels = np.load(os.path.join(args.data_dir, 'val_labels.npy'))
    
    print(f'Train set: {len(train_features)} samples')
    print(f'Val set: {len(val_features)} samples')
    
    # Create datasets
    train_dataset = MusicFeatureDataset(train_features, train_labels)
    val_dataset = MusicFeatureDataset(val_features, val_labels)
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        train_dataset,
        val_dataset,
        val_dataset,  # Placeholder for test
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Create model
    print('\nCreating model...')
    model = create_model(config)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {config["model"]["type"]}')
    print(f'Parameters: {num_params:,}')
    
    # Train
    print('\nStarting training...')
    trained_model = train(config, train_loader, val_loader, model, device)
    
    print('\n✓ Training complete!')


if __name__ == '__main__':
    main()
