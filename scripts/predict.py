"""
Inference script for making predictions on new songs
"""

import torch
import numpy as np
import yaml
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import create_model, load_config


class MusicGeographyPredictor:
    """
    Inference wrapper for geographic music classification.
    
    Usage:
        predictor = MusicGeographyPredictor(
            config_path='configs/config.yaml',
            checkpoint_path='checkpoints/best_model.pth',
            label_encoder_path='data/splits/label_encoder.pkl'
        )
        
        predictions = predictor.predict(features, top_k=5)
    """
    
    def __init__(self, config_path, checkpoint_path, label_encoder_path, device='cuda'):
        """Initialize predictor"""
        self.config = load_config(config_path)
        
        # Set device
        self.device = torch.device(
            device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        )
        print(f'Using device: {self.device}')
        
        # Load model
        self.model = create_model(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f'Model loaded from {checkpoint_path}')
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f'Label encoder loaded with {len(self.label_encoder.classes_)} countries')
    
    def predict(self, features, top_k=5):
        """
        Predict geographic origin from features.
        
        Args:
            features (np.ndarray): Feature array of shape (n_features,) or (batch_size, n_features)
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of (country, probability) tuples
        """
        # Handle single sample
        if len(features.shape) == 1:
            features = features[np.newaxis, :]
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions for each sample
        results = []
        for i in range(len(features)):
            top_probs, top_indices = torch.topk(probs[i], top_k)
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                country = self.label_encoder.inverse_transform([idx.item()])[0]
                predictions.append((country, prob.item()))
            
            results.append(predictions)
        
        # If single sample, return just the list
        if len(results) == 1:
            return results[0]
        
        return results
    
    def predict_from_spotify(self, track_name, artist_name, spotify_api=None):
        """
        Predict from Spotify track (requires Spotify API setup).
        
        Args:
            track_name (str): Track name
            artist_name (str): Artist name
            spotify_api: Spotipy client (optional)
            
        Returns:
            list: Top predictions
        """
        # TODO: Implement Spotify feature extraction
        # This would require:
        # 1. Search for track using spotipy
        # 2. Get audio features
        # 3. Extract the 13 features we need
        # 4. Call predict()
        
        raise NotImplementedError(
            "Spotify integration not yet implemented. "
            "Use predict() with pre-extracted features instead."
        )


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions on music')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--label_encoder', type=str, default='data/splits/label_encoder.pkl')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to .npy file with features')
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    
    # Load predictor
    predictor = MusicGeographyPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        label_encoder_path=args.label_encoder
    )
    
    # Load features
    features = np.load(args.features)
    print(f'\nLoaded features with shape: {features.shape}')
    
    # Make predictions
    print(f'\nMaking predictions (top {args.top_k})...\n')
    predictions = predictor.predict(features, top_k=args.top_k)
    
    # Handle single vs batch predictions
    if isinstance(predictions[0], tuple):
        # Single prediction
        print("Predictions:")
        for i, (country, prob) in enumerate(predictions, 1):
            print(f"{i}. {country}: {prob*100:.2f}%")
    else:
        # Batch predictions
        for idx, song_preds in enumerate(predictions):
            print(f"\nSong {idx+1}:")
            for i, (country, prob) in enumerate(song_preds, 1):
                print(f"  {i}. {country}: {prob*100:.2f}%")


if __name__ == '__main__':
    main()
