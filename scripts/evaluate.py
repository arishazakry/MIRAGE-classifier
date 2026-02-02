"""
Evaluation script for music geographic classification
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score
)
import yaml
import os
import sys
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import create_model, load_config
from utils.dataset import MusicFeatureDataset, create_dataloaders


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Returns:
        dict: Dictionary containing predictions, labels, and probabilities
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def calculate_metrics(results, label_encoder):
    """Calculate comprehensive evaluation metrics"""
    preds = results['predictions']
    labels = results['labels']
    probs = results['probabilities']
    
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(labels, preds) * 100
    
    # Top-k accuracy
    metrics['top3_accuracy'] = top_k_accuracy_score(labels, probs, k=3) * 100
    metrics['top5_accuracy'] = top_k_accuracy_score(labels, probs, k=5) * 100
    
    # F1 scores
    metrics['f1_macro'] = f1_score(labels, preds, average='macro') * 100
    metrics['f1_micro'] = f1_score(labels, preds, average='micro') * 100
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted') * 100
    
    # Per-class metrics
    country_names = label_encoder.classes_
    class_report = classification_report(
        labels, 
        preds, 
        target_names=country_names,
        output_dict=True,
        zero_division=0
    )
    
    metrics['per_class'] = class_report
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    metrics['confusion_matrix'] = cm
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    metrics['per_class_accuracy'] = dict(zip(country_names, per_class_accuracy * 100))
    
    return metrics


def plot_confusion_matrix(cm, label_encoder, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(20, 16))
    
    country_names = label_encoder.classes_
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=False,
        cmap='Blues',
        xticklabels=country_names,
        yticklabels=country_names,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.xlabel('Predicted Country', fontsize=12)
    plt.ylabel('True Country', fontsize=12)
    plt.title('Confusion Matrix - Geographic Music Classification', fontsize=14)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Confusion matrix saved to {save_path}')


def plot_per_class_accuracy(per_class_acc, save_path):
    """Plot per-class accuracy"""
    countries = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    countries = [countries[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 10))
    plt.barh(countries, accuracies, color='steelblue')
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.title('Per-Country Classification Accuracy', fontsize=14)
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Per-class accuracy plot saved to {save_path}')


def plot_top_confusions(cm, label_encoder, top_n=20, save_path=None):
    """Plot top confusion pairs"""
    country_names = label_encoder.classes_
    
    # Get off-diagonal confusion pairs
    confusion_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true': country_names[i],
                    'pred': country_names[j],
                    'count': cm[i, j]
                })
    
    # Sort by count
    confusion_pairs = sorted(confusion_pairs, key=lambda x: x['count'], reverse=True)
    
    # Take top N
    top_confusions = confusion_pairs[:top_n]
    
    labels = [f"{c['true']} → {c['pred']}" for c in top_confusions]
    counts = [c['count'] for c in top_confusions]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(labels)), counts, color='coral')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Number of Confusions', fontsize=12)
    plt.title(f'Top {top_n} Country Confusions', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Top confusions plot saved to {save_path}')
    else:
        plt.show()


def save_results(metrics, save_dir):
    """Save evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as text
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MUSIC GEOGRAPHIC CLASSIFICATION - EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Top-3 Accuracy: {metrics['top3_accuracy']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%\n")
        f.write(f"F1 Score (Macro): {metrics['f1_macro']:.2f}%\n")
        f.write(f"F1 Score (Micro): {metrics['f1_micro']:.2f}%\n")
        f.write(f"F1 Score (Weighted): {metrics['f1_weighted']:.2f}%\n\n")
        
        f.write("Top 10 Best Performing Countries:\n")
        f.write("-" * 60 + "\n")
        sorted_countries = sorted(
            metrics['per_class_accuracy'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for country, acc in sorted_countries[:10]:
            f.write(f"{country}: {acc:.2f}%\n")
        
        f.write("\nTop 10 Worst Performing Countries:\n")
        f.write("-" * 60 + "\n")
        for country, acc in sorted_countries[-10:]:
            f.write(f"{country}: {acc:.2f}%\n")
    
    # Save confusion matrix
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), metrics['confusion_matrix'])
    
    # Save all metrics as pickle
    with open(os.path.join(save_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f'Results saved to {save_dir}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/splits/',
                        help='Directory with processed data splits')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device(
        config['hardware']['device']
        if torch.cuda.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')
    
    # Load test data
    print('\nLoading test data...')
    test_features = np.load(os.path.join(args.data_dir, 'test_features.npy'))
    test_labels = np.load(os.path.join(args.data_dir, 'test_labels.npy'))
    
    with open(os.path.join(args.data_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f'Test set: {len(test_features)} samples')
    print(f'Number of classes: {len(label_encoder.classes_)}')
    
    # Create dataset and dataloader
    test_dataset = MusicFeatureDataset(test_features, test_labels, label_encoder)
    _, _, test_loader = create_dataloaders(
        test_dataset,
        test_dataset,
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        pin_memory=False
    )
    
    # Load model
    print('\nLoading model...')
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    print(f'Validation accuracy: {checkpoint["val_acc"]:.2f}%')
    
    # Evaluate
    print('\nEvaluating model...')
    results = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    print('\nCalculating metrics...')
    metrics = calculate_metrics(results, label_encoder)
    
    # Print results
    print('\n' + '=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    print(f'Test Accuracy: {metrics["accuracy"]:.2f}%')
    print(f'Top-3 Accuracy: {metrics["top3_accuracy"]:.2f}%')
    print(f'Top-5 Accuracy: {metrics["top5_accuracy"]:.2f}%')
    print(f'F1 Score (Macro): {metrics["f1_macro"]:.2f}%')
    print(f'F1 Score (Micro): {metrics["f1_micro"]:.2f}%')
    print('=' * 60)
    
    # Save results
    print('\nSaving results...')
    save_results(metrics, args.output_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        label_encoder,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Plot per-class accuracy
    plot_per_class_accuracy(
        metrics['per_class_accuracy'],
        os.path.join(args.output_dir, 'per_class_accuracy.png')
    )
    
    # Plot top confusions
    plot_top_confusions(
        metrics['confusion_matrix'],
        label_encoder,
        top_n=20,
        save_path=os.path.join(args.output_dir, 'top_confusions.png')
    )
    
    print('\n✓ Evaluation complete!')


if __name__ == '__main__':
    main()
