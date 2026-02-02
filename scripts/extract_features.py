"""
Feature extraction from MIRAGE-MetaCorpus
Downloads data from Zenodo and extracts features using Spotify API
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Spotify API
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sys.path.append(str(Path(__file__).parent.parent))
from utils.dataset import (
    create_label_encoders,
    create_train_val_test_splits,
    save_datasets,
    compute_class_weights
)


def download_mirage_data(output_dir='data/raw/'):
    """
    Download MIRAGE-MetaCorpus from Zenodo.
    
    Zenodo record: https://zenodo.org/records/18112107
    
    Manual download recommended due to size (several GB).
    """
    print("\n" + "="*60)
    print("MIRAGE-MetaCorpus Download Instructions")
    print("="*60)
    print("\n1. Visit: https://zenodo.org/records/18112107")
    print("\n2. Download these files to", output_dir)
    print("   - MIRAGE.csv (complete metacorpus)")
    print("   - events.csv (all event-level metadata)")
    print("   - tracks.csv (all track-level metadata)")
    print("   - artists.csv (all artist-level metadata)")
    print("   - stations.csv (all station-level metadata)")
    print("   - locations.csv (all location-level metadata)")
    print("\n3. Or download via command line:")
    print("\n   wget -P", output_dir, "https://zenodo.org/records/18112107/files/MIRAGE.csv")
    print("   wget -P", output_dir, "https://zenodo.org/records/18112107/files/events.csv")
    print("   # ... (repeat for other files)")
    print("\n" + "="*60 + "\n")


def load_mirage_corpus(data_dir='data/raw/', use_complete=True):
    """
    Load MIRAGE-MetaCorpus datasets.
    
    Args:
        data_dir (str): Directory containing MIRAGE CSV files
        use_complete (bool): If True, use MIRAGE.csv; otherwise merge individual CSVs
        
    Returns:
        dict: Dictionary of DataFrames
    """
    print("Loading MIRAGE-MetaCorpus...")
    
    data = {}
    
    if use_complete and os.path.exists(os.path.join(data_dir, 'MIRAGE.csv')):
        # Load complete corpus (fastest)
        print("Loading MIRAGE.csv (complete corpus)...")
        data['complete'] = pd.read_csv(os.path.join(data_dir, 'MIRAGE.csv'))
        print(f"  Loaded {len(data['complete']):,} events")
    else:
        # Load individual datasets
        files = {
            'events': 'events.csv',
            'tracks': 'tracks.csv',
            'artists': 'artists.csv',
            'stations': 'stations.csv',
            'locations': 'locations.csv'
        }
        
        for key, filename in files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                print(f"Loading {filename}...")
                data[key] = pd.read_csv(filepath)
                print(f"  Loaded {len(data[key]):,} rows")
            else:
                print(f"  Warning: {filename} not found")
    
    return data


def setup_spotify_client():
    """
    Set up Spotify API client.
    
    Requires environment variables:
    - SPOTIPY_CLIENT_ID
    - SPOTIPY_CLIENT_SECRET
    
    Or create .env file:
    SPOTIPY_CLIENT_ID=your_client_id
    SPOTIPY_CLIENT_SECRET=your_client_secret
    """
    load_dotenv()
    
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise ValueError(
            "Spotify API credentials not found!\n"
            "Please set environment variables:\n"
            "  export SPOTIPY_CLIENT_ID='your_client_id'\n"
            "  export SPOTIPY_CLIENT_SECRET='your_client_secret'\n"
            "Or create a .env file with these variables.\n\n"
            "Get credentials at: https://developer.spotify.com/dashboard"
        )
    
    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    # Test connection
    try:
        sp.search(q='test', limit=1)
        print("✓ Spotify API connected successfully")
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Spotify API: {e}")
    
    return sp


def extract_spotify_features(sp, track_name, artist_name, max_retries=3):
    """
    Extract Spotify audio features for a track.
    
    Args:
        sp: Spotipy client
        track_name (str): Track name
        artist_name (str): Artist name
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        np.ndarray: 13 Spotify features, or None if not found
    """
    for attempt in range(max_retries):
        try:
            # Search for track
            query = f"track:{track_name} artist:{artist_name}"
            results = sp.search(q=query, type='track', limit=1)
            
            if not results['tracks']['items']:
                return None
            
            track_id = results['tracks']['items'][0]['id']
            
            # Get audio features
            features = sp.audio_features(track_id)
            
            if not features or not features[0]:
                return None
            
            features = features[0]
            
            # Extract 13 features in consistent order
            feature_vector = np.array([
                features['danceability'],
                features['energy'],
                features['key'],
                features['loudness'],
                features['mode'],
                features['speechiness'],
                features['acousticness'],
                features['instrumentalness'],
                features['liveness'],
                features['valence'],
                features['tempo'],
                features['duration_ms'],
                features['time_signature']
            ], dtype=np.float32)
            
            return feature_vector
            
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:  # Rate limit
                wait_time = int(e.headers.get('Retry-After', 5))
                print(f"\n  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(1)  # Brief pause before retry
            else:
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
    
    return None


def prepare_mirage_for_extraction(df, sample_size=None, random_state=42):
    """
    Prepare MIRAGE corpus for feature extraction.
    
    Args:
        df (pd.DataFrame): MIRAGE dataframe
        sample_size (int, optional): Subsample to this many tracks
        random_state (int): Random seed
        
    Returns:
        pd.DataFrame: Prepared dataframe
    """
    print("\nPreparing data for extraction...")
    
    # Required columns
    required_cols = ['track_name', 'artist_name', 'country']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        print("Available columns:", df.columns.tolist())
        # Try alternative column names
        column_mapping = {
            'track_title': 'track_name',
            'track': 'track_name',
            'artist': 'artist_name',
            'country_name': 'country',
            'location_country': 'country'
        }
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
                print(f"  Mapped {old_col} → {new_col}")
    
    # Drop rows with missing critical info
    initial_len = len(df)
    df = df.dropna(subset=['track_name', 'artist_name', 'country'])
    print(f"  Dropped {initial_len - len(df):,} rows with missing data")
    
    # Remove duplicates (same track + artist)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['track_name', 'artist_name'], keep='first')
    print(f"  Removed {initial_len - len(df):,} duplicate tracks")
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"  Sampled {sample_size:,} tracks")
    
    print(f"Final dataset: {len(df):,} unique tracks")
    print(f"Countries: {df['country'].nunique()}")
    
    return df


def batch_extract_features(df, sp, batch_size=100, save_every=1000, 
                          output_path='data/features/spotify_features.pkl'):
    """
    Extract Spotify features for all tracks with progress saving.
    
    Args:
        df (pd.DataFrame): DataFrame with track_name, artist_name, country
        sp: Spotipy client
        batch_size (int): Process this many tracks between rate limit checks
        save_every (int): Save progress every N tracks
        output_path (str): Where to save features
        
    Returns:
        dict: Dictionary with features, labels, and metadata
    """
    print(f"\nExtracting Spotify features for {len(df):,} tracks...")
    print(f"This may take a while (~{len(df)/1000:.1f} min for 1000 tracks)")
    
    # Initialize storage
    features_list = []
    metadata_list = []
    failed_tracks = []
    
    # Check for existing progress
    progress_file = output_path.replace('.pkl', '_progress.pkl')
    start_idx = 0
    
    if os.path.exists(progress_file):
        print(f"\nFound existing progress file: {progress_file}")
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() == 'y':
            with open(progress_file, 'rb') as f:
                checkpoint = pickle.load(f)
                features_list = checkpoint['features']
                metadata_list = checkpoint['metadata']
                failed_tracks = checkpoint['failed']
                start_idx = checkpoint['index']
            print(f"Resuming from track {start_idx:,}/{len(df):,}")
    
    # Progress bar
    pbar = tqdm(total=len(df), initial=start_idx, desc='Extracting features')
    
    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        
        # Extract features
        features = extract_spotify_features(sp, row['track_name'], row['artist_name'])
        
        if features is not None:
            features_list.append(features)
            metadata_list.append({
                'index': idx,
                'track_name': row['track_name'],
                'artist_name': row['artist_name'],
                'country': row['country']
            })
        else:
            failed_tracks.append({
                'index': idx,
                'track_name': row['track_name'],
                'artist_name': row['artist_name'],
                'country': row['country']
            })
        
        pbar.update(1)
        
        # Rate limiting: pause after each batch
        if (idx + 1) % batch_size == 0:
            time.sleep(1)
        
        # Save progress
        if (idx + 1) % save_every == 0:
            checkpoint = {
                'features': features_list,
                'metadata': metadata_list,
                'failed': failed_tracks,
                'index': idx + 1
            }
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            pbar.set_postfix({'saved': f'{idx+1}/{len(df)}'})
    
    pbar.close()
    
    # Summary
    success_rate = len(features_list) / len(df) * 100
    print(f"\n✓ Extraction complete!")
    print(f"  Successful: {len(features_list):,}/{len(df):,} ({success_rate:.1f}%)")
    print(f"  Failed: {len(failed_tracks):,}")
    
    # Convert to arrays
    features_array = np.array(features_list)
    
    # Get labels
    labels = [m['country'] for m in metadata_list]
    
    # Save final results
    result = {
        'features': features_array,
        'metadata': metadata_list,
        'labels': labels,
        'failed_tracks': failed_tracks,
        'feature_names': [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
        ]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"\n✓ Features saved to: {output_path}")
    
    # Save failed tracks list
    if failed_tracks:
        failed_path = output_path.replace('.pkl', '_failed.csv')
        pd.DataFrame(failed_tracks).to_csv(failed_path, index=False)
        print(f"✓ Failed tracks list saved to: {failed_path}")
    
    # Clean up progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    return result


def create_splits_from_features(features_data, config, output_dir='data/splits/'):
    """
    Create train/val/test splits from extracted features.
    
    Args:
        features_data (dict): Dictionary with 'features' and 'labels'
        config (dict): Configuration dictionary
        output_dir (str): Where to save splits
        
    Returns:
        dict: Split data and encoders
    """
    print("\nCreating train/val/test splits...")
    
    features = features_data['features']
    labels_str = features_data['labels']
    
    # Create DataFrame for easier handling
    df = pd.DataFrame({'country': labels_str})
    
    # Encode labels
    df, encoders = create_label_encoders(df)
    labels = df['country_label'].values
    
    print(f"\nLabel encoding:")
    print(f"  Countries: {len(encoders['country'].classes_)}")
    print(f"  Classes: {encoders['country'].classes_[:10]}...")  # Show first 10
    
    # Create splits
    train_features, val_features, test_features, \
    train_labels, val_labels, test_labels = create_train_val_test_splits(
        df,
        features,
        label_col='country_label',
        train_size=config['dataset']['train_split'],
        val_size=config['dataset']['val_split'],
        test_size=config['dataset']['test_split'],
        random_state=config['dataset']['random_seed'],
        stratify=config['dataset']['stratify']
    )
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(
        train_labels,
        num_classes=len(encoders['country'].classes_)
    )
    
    print(f"\nClass weights computed (for handling imbalance)")
    print(f"  Min weight: {class_weights.min():.3f}")
    print(f"  Max weight: {class_weights.max():.3f}")
    
    # Save everything
    save_datasets(
        train_features, val_features, test_features,
        train_labels, val_labels, test_labels,
        encoders['country'],
        output_dir
    )
    
    # Save class weights
    np.save(os.path.join(output_dir, 'class_weights.npy'), class_weights.numpy())
    print(f"✓ Class weights saved")
    
    # Save feature names for reference
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        for name in features_data['feature_names']:
            f.write(f"{name}\n")
    print(f"✓ Feature names saved to: {feature_names_path}")
    
    return {
        'train_features': train_features,
        'val_features': val_features,
        'test_features': test_features,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'encoder': encoders['country'],
        'class_weights': class_weights
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from MIRAGE-MetaCorpus'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw/',
        help='Directory with MIRAGE CSV files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/splits/',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Subsample to this many tracks (for testing)'
    )
    parser.add_argument(
        '--download_instructions',
        action='store_true',
        help='Show download instructions and exit'
    )
    args = parser.parse_args()
    
    # Show download instructions
    if args.download_instructions:
        download_mirage_data(args.data_dir)
        return
    
    # Load configuration
    config_path = 'configs/config.yaml'
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use defaults
        config = {
            'dataset': {
                'train_split': 0.7,
                'val_split': 0.1,
                'test_split': 0.2,
                'random_seed': 42,
                'stratify': True
            }
        }
    
    # Step 1: Load MIRAGE corpus
    mirage_data = load_mirage_corpus(args.data_dir)
    
    if 'complete' in mirage_data:
        df = mirage_data['complete']
    elif 'events' in mirage_data and 'tracks' in mirage_data:
        # Merge datasets if needed
        print("\nMerging datasets...")
        df = mirage_data['events'].merge(
            mirage_data['tracks'],
            on='track_id',
            how='left'
        )
        if 'artists' in mirage_data:
            df = df.merge(
                mirage_data['artists'],
                on='artist_id',
                how='left'
            )
    else:
        raise ValueError("Could not find MIRAGE data. Use --download_instructions")
    
    # Step 2: Prepare data
    df_prepared = prepare_mirage_for_extraction(
        df,
        sample_size=args.sample_size
    )
    
    # Step 3: Set up Spotify API
    print("\nSetting up Spotify API...")
    sp = setup_spotify_client()
    
    # Step 4: Extract features
    features_path = 'data/features/spotify_features.pkl'
    
    if os.path.exists(features_path):
        print(f"\nFound existing features: {features_path}")
        response = input("Use existing features? (y/n): ")
        if response.lower() == 'y':
            with open(features_path, 'rb') as f:
                features_data = pickle.load(f)
            print(f"✓ Loaded {len(features_data['features']):,} extracted features")
        else:
            features_data = batch_extract_features(df_prepared, sp, output_path=features_path)
    else:
        features_data = batch_extract_features(df_prepared, sp, output_path=features_path)
    
    # Step 5: Create splits
    splits = create_splits_from_features(features_data, config, args.output_dir)
    
    print("\n" + "="*60)
    print("✓ FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nData saved to: {args.output_dir}")
    print(f"\nYou can now train models with:")
    print(f"  python scripts/train.py --data_dir {args.output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
