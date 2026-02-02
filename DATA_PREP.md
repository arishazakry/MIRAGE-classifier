# MIRAGE-MetaCorpus Data Preparation Guide

Complete guide for extracting features from the MIRAGE-MetaCorpus for music geographic classification.

## 📚 About MIRAGE-MetaCorpus

**MIRAGE** (Music Informatics for Radio Across the GlobE) is a large-scale corpus of global radio broadcast metadata.

**Dataset Details:**
- **Source**: Zenodo - https://zenodo.org/records/18112107
- **Size**: ~1 million broadcast events
- **Stations**: 10,000 radio stations
- **Countries**: 57 countries
- **Time Period**: October 2022 - January 2023

**Included Files:**
- `MIRAGE.csv` - Complete metacorpus (1M events)
- `events.csv` - Event-level metadata (1M)
- `tracks.csv` - Track-level metadata (414,886 unique tracks)
- `artists.csv` - Artist-level metadata (259,783 artists)
- `stations.csv` - Station-level metadata (10,000 stations)
- `locations.csv` - Location-level metadata (4,324 locations)

## 🚀 Quick Start

### Step 1: Download MIRAGE Data (15-30 minutes)

```bash
# Create data directory
mkdir -p data/raw

# Option A: Download via browser
# Visit: https://zenodo.org/records/18112107
# Download MIRAGE.csv (or all CSVs) to data/raw/

# Option B: Download via command line (Linux/Mac)
cd data/raw
wget https://zenodo.org/records/18112107/files/MIRAGE.csv

# Option C: Show download instructions
python scripts/extract_features.py --download_instructions
```

**File Sizes:**
- MIRAGE.csv: ~500 MB
- Complete dataset (all CSVs): ~1.5 GB

### Step 2: Get Spotify API Credentials (5 minutes)

1. Go to https://developer.spotify.com/dashboard
2. Log in (or create account)
3. Click "Create an App"
4. Note your **Client ID** and **Client Secret**

### Step 3: Set Environment Variables

**Option A: Using .env file (recommended)**
```bash
# Create .env file in project root
cat > .env << EOF
SPOTIPY_CLIENT_ID='your_client_id_here'
SPOTIPY_CLIENT_SECRET='your_client_secret_here'
EOF
```

**Option B: Export directly**
```bash
export SPOTIPY_CLIENT_ID='your_client_id_here'
export SPOTIPY_CLIENT_SECRET='your_client_secret_here'
```

**Option C: Windows**
```cmd
set SPOTIPY_CLIENT_ID=your_client_id_here
set SPOTIPY_CLIENT_SECRET=your_client_secret_here
```

### Step 4: Extract Features (varies: 1 hour - 1 day)

```bash
# Full corpus extraction (~10-20 hours)
python scripts/extract_features.py \
  --data_dir data/raw/ \
  --output_dir data/splits/

# Quick test with 1000 tracks (~15-30 minutes)
python scripts/extract_features.py \
  --data_dir data/raw/ \
  --output_dir data/splits/ \
  --sample_size 1000
```

**What happens:**
1. Loads MIRAGE.csv
2. Removes duplicates (same track + artist)
3. For each unique track:
   - Searches Spotify API
   - Extracts 13 audio features
   - Saves progress every 1000 tracks
4. Creates train/val/test splits (70/10/20)
5. Saves encoded labels and class weights

**Expected output:**
```
Loading MIRAGE-MetaCorpus...
  Loaded 1,000,000 events
Preparing data for extraction...
  Dropped X rows with missing data
  Removed Y duplicate tracks
Final dataset: ~414,000 unique tracks

Extracting Spotify features...
Extracting features: 100%|████████| 414000/414000
✓ Extraction complete!
  Successful: 350,000/414,000 (84.5%)
  Failed: 64,000

Creating train/val/test splits...
Train: 245,000 samples
Val: 35,000 samples  
Test: 70,000 samples

✓ FEATURE EXTRACTION COMPLETE!
```

## 📊 Data Structure

### Input: MIRAGE-MetaCorpus

**Key columns:**
```python
- track_name: "Sutera"
- artist_name: "Aisha Retno"  
- country: "Indonesia"
- region: "Southeast Asia"
- continent: "Asia"
- station_id: Station identifier
- event_time: Timestamp of broadcast
```

### Output: Processed Features

**File structure:**
```
data/splits/
├── train_features.npy         # (245000, 13) - Spotify features
├── val_features.npy           # (35000, 13)
├── test_features.npy          # (70000, 13)
├── train_labels.npy           # (245000,) - Country IDs (0-56)
├── val_labels.npy             # (35000,)
├── test_labels.npy            # (70000,)
├── label_encoder.pkl          # Country name ↔ ID mapping
├── class_weights.npy          # For handling imbalanced classes
└── feature_names.txt          # List of 13 Spotify features
```

**Spotify Features (13 total):**
```python
1.  danceability      # 0-1, how suitable for dancing
2.  energy            # 0-1, intensity and activity
3.  key               # 0-11, musical key (C, C#, D, ...)
4.  loudness          # dB, overall loudness
5.  mode              # 0/1, major or minor
6.  speechiness       # 0-1, presence of spoken words
7.  acousticness      # 0-1, acoustic vs electronic
8.  instrumentalness  # 0-1, vocal vs instrumental
9.  liveness          # 0-1, presence of audience
10. valence           # 0-1, musical positivity
11. tempo             # BPM, beats per minute
12. duration_ms       # milliseconds, track length
13. time_signature    # beats per bar (3, 4, 5, ...)
```

## 🔄 Resume Interrupted Extraction

Feature extraction saves progress every 1000 tracks. If interrupted:

```bash
# Resume from checkpoint
python scripts/extract_features.py \
  --data_dir data/raw/ \
  --output_dir data/splits/

# You'll see:
# Found existing progress file: data/features/spotify_features_progress.pkl
# Resume from checkpoint? (y/n): y
# Resuming from track 5,000/414,000
```

## ⚡ Performance Tips

### Speed Up Extraction

**1. Use parallel processing (advanced)**
```bash
# Split MIRAGE.csv into chunks
# Run multiple extraction processes on different chunks
# Merge results

# TODO: Add multi-processing script
```

**2. Start with a sample**
```bash
# Test pipeline with 1K tracks first
python scripts/extract_features.py --sample_size 1000

# Then scale up
python scripts/extract_features.py --sample_size 10000
python scripts/extract_features.py  # Full dataset
```

**3. Monitor Spotify API limits**
- Rate limit: ~180 requests/minute
- Daily limit: None (for Client Credentials flow)
- Our script automatically handles rate limiting

### Reduce Failed Tracks

**Common reasons for failures:**
- Track not available on Spotify
- Incorrect track/artist name in MIRAGE
- Regional availability restrictions

**Solutions:**
```bash
# Check failed tracks
cat data/features/spotify_features_failed.csv

# Most common failures:
# - Local/regional artists not on Spotify
# - Misspelled track names
# - Different language scripts

# Typically 10-20% failure rate is normal
```

## 🔍 Data Quality Checks

### Verify Extracted Features

```python
import numpy as np
import pickle

# Load features
train_features = np.load('data/splits/train_features.npy')
train_labels = np.load('data/splits/train_labels.npy')

print(f"Features shape: {train_features.shape}")  # (N, 13)
print(f"Labels shape: {train_labels.shape}")      # (N,)

# Check for NaN values
print(f"NaN in features: {np.isnan(train_features).any()}")

# Check label distribution
unique, counts = np.unique(train_labels, return_counts=True)
print(f"Countries: {len(unique)}")
print(f"Min samples per country: {counts.min()}")
print(f"Max samples per country: {counts.max()}")

# Load encoder to see country names
with open('data/splits/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
    
print("\nCountries:", encoder.classes_[:10], "...")
```

### Check Class Balance

```python
import matplotlib.pyplot as plt

# Plot distribution
plt.figure(figsize=(12, 6))
plt.bar(range(len(counts)), sorted(counts, reverse=True))
plt.xlabel('Country (sorted by frequency)')
plt.ylabel('Number of samples')
plt.title('Class Distribution in Training Set')
plt.show()

# Expected: Imbalanced (US/Europe over-represented)
# Solution: We use class_weights.npy during training
```

## 🐛 Troubleshooting

### Issue: "Spotify API credentials not found"

**Solution:**
```bash
# Check environment variables
echo $SPOTIPY_CLIENT_ID
echo $SPOTIPY_CLIENT_SECRET

# If empty, set them:
export SPOTIPY_CLIENT_ID='your_id'
export SPOTIPY_CLIENT_SECRET='your_secret'

# Or create .env file (see Step 3)
```

### Issue: "Failed to connect to Spotify API"

**Solutions:**
```bash
# 1. Check internet connection
ping spotify.com

# 2. Verify credentials are correct
# Visit: https://developer.spotify.com/dashboard

# 3. Check if app is active
# Your Spotify Developer app must be in "Development" mode
```

### Issue: "Rate limit exceeded"

**Solution:**
The script automatically handles this. You'll see:
```
Rate limited. Waiting 60s...
```

If you see this frequently, the script will slow down automatically.

### Issue: "Out of memory"

**Solution:**
```bash
# Process in batches
python scripts/extract_features.py --sample_size 50000
# Move output, repeat for next batch
# Merge results manually
```

### Issue: "Column 'track_name' not found"

**Solution:**
```python
# Check MIRAGE.csv column names
import pandas as pd
df = pd.read_csv('data/raw/MIRAGE.csv')
print(df.columns.tolist())

# The script will try to auto-map common alternatives:
# 'track' → 'track_name'
# 'track_title' → 'track_name'
# 'artist' → 'artist_name'
# etc.
```

## 📈 Expected Results

### Extraction Statistics

For full MIRAGE corpus (~414K unique tracks):

| Metric | Expected Value |
|--------|---------------|
| Total tracks | ~414,000 |
| Successful extractions | ~350,000 (85%) |
| Failed extractions | ~64,000 (15%) |
| Processing time | 10-20 hours |
| Final training set | ~245,000 samples |
| Final validation set | ~35,000 samples |
| Final test set | ~70,000 samples |
| Countries | 57 |

### Feature Statistics

Expected ranges for Spotify features:

```python
Feature              Mean    Std     Min    Max
─────────────────────────────────────────────
danceability        0.58    0.16    0.0    1.0
energy              0.64    0.21    0.0    1.0
key                 5.31    3.52    0.0    11.0
loudness           -7.23    3.12   -35.0   0.0
mode                0.61    0.49    0.0    1.0
speechiness         0.09    0.11    0.0    1.0
acousticness        0.33    0.31    0.0    1.0
instrumentalness    0.11    0.26    0.0    1.0
liveness            0.19    0.16    0.0    1.0
valence             0.51    0.24    0.0    1.0
tempo             120.45   29.32   40.0   220.0
duration_ms    216835.21 76543.12 30000  900000
time_signature      3.95    0.29    3.0    7.0
```

## 🎯 Next Steps

After feature extraction completes:

```bash
# 1. Train baseline model
python scripts/train.py --data_dir data/splits/

# 2. Evaluate
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --data_dir data/splits/

# 3. Make predictions
python scripts/predict.py \
  --checkpoint checkpoints/best_model.pth \
  --features data/splits/test_features.npy
```

See [QUICKSTART.md](QUICKSTART.md) for complete training guide.

## 📝 Citation

If you use MIRAGE-MetaCorpus in your research:

```bibtex
@dataset{mirage_metacorpus_2024,
  author = {Nguyen, et al. and Sears, et al.},
  title = {MIRAGE-MetaCorpus},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18112107},
  url = {https://zenodo.org/records/18112107}
}
```

---

**Questions?** Open an issue or check the main [README.md](README.md)
