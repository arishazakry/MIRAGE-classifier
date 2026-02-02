# Music Geographic Classification - Complete Implementation

## 🎵 Project Summary

A complete, production-ready PyTorch implementation for classifying music by geographic origin using deep learning on the MIRAGE-MetaCorpus dataset.

## 📦 What's Included

### ✅ **4 Deep Learning Models**
1. **FCNN** (Fully Connected Neural Network)
   - 3 hidden layers: [256, 128, 64]
   - BatchNorm + Dropout + ReLU
   - ~150K parameters
   - Training time: 2-3 hours
   - Expected accuracy: 25-35%

2. **CNN** (Convolutional Neural Network)
   - 4 conv blocks + adaptive pooling
   - ~2M parameters
   - Training time: 8-12 hours  
   - Expected accuracy: 30-40%

3. **RNN** (LSTM-based Recurrent Network)
   - Bidirectional LSTM (2 layers)
   - ~500K parameters
   - Training time: 6-10 hours
   - Expected accuracy: 35-45%

4. **Attention** (LSTM + Self-Attention)
   - 3 attention types (self, additive, dot-product)
   - ~550K parameters
   - Training time: 10-15 hours
   - Expected accuracy: 35-45%

### ✅ **Complete Pipeline**
- **Feature Extraction** (`scripts/extract_features.py`)
  - Integrates with Spotify API
  - Processes MIRAGE-MetaCorpus
  - Handles 414K unique tracks
  - Auto-saves progress
  - Resume from checkpoints

- **Training** (`scripts/train.py`)
  - Mixed precision training
  - Gradient clipping
  - Learning rate scheduling
  - Early stopping
  - Automatic checkpointing

- **Evaluation** (`scripts/evaluate.py`)
  - Comprehensive metrics
  - Confusion matrices
  - Per-class accuracy
  - Top-K accuracy
  - Visualization plots

- **Inference** (`scripts/predict.py`)
  - Production-ready predictor
  - Batch predictions
  - Top-K results

### ✅ **Data Processing**
- **Dataset Classes** (`utils/dataset.py`)
  - `MusicFeatureDataset` - For FCNN/RNN
  - `MusicSpectrogramDataset` - For CNN
  - `MusicSequenceDataset` - For sequential models
  - Train/val/test splitting
  - Label encoding
  - Class weight computation

### ✅ **Documentation**
- **README.md** - Complete project documentation (2800+ words)
- **QUICKSTART.md** - Step-by-step tutorial with examples
- **DATA_PREP.md** - MIRAGE-specific data preparation guide
- **Code comments** - Extensive inline documentation

### ✅ **Configuration**
- **config.yaml** - Centralized hyperparameter config
  - Model architectures
  - Training settings
  - Optimizer/scheduler
  - Hardware config
  - Reproducibility settings

## 📊 Data: MIRAGE-MetaCorpus

**Source:** https://zenodo.org/records/18112107

**Statistics:**
- 1 million broadcast events
- 414,886 unique tracks
- 259,783 unique artists
- 10,000 radio stations
- 57 countries
- 19 geographic regions

**After preprocessing:**
- Train: ~245,000 samples
- Validation: ~35,000 samples
- Test: ~70,000 samples

## 🚀 Quick Start

### 1. Setup (5 minutes)
```bash
tar -xzf music_geo_classifier_complete.tar.gz
cd music_geo_classifier
pip install -r requirements.txt
```

### 2. Download MIRAGE Data (15-30 minutes)
```bash
# Visit: https://zenodo.org/records/18112107
# Download MIRAGE.csv to data/raw/
wget -P data/raw/ https://zenodo.org/records/18112107/files/MIRAGE.csv
```

### 3. Get Spotify Credentials (5 minutes)
```bash
# https://developer.spotify.com/dashboard
# Create .env file:
echo "SPOTIPY_CLIENT_ID='your_id'" > .env
echo "SPOTIPY_CLIENT_SECRET='your_secret'" >> .env
```

### 4. Extract Features (10-20 hours)
```bash
# Quick test (1000 tracks, ~30 minutes)
python scripts/extract_features.py --sample_size 1000

# Full corpus (414K tracks, ~10-20 hours)
python scripts/extract_features.py
```

### 5. Train Model (2-12 hours)
```bash
# Baseline FCNN
python scripts/train.py --config configs/config.yaml

# Try other models by editing config.yaml:
# model: {type: "cnn"}  # or "rnn" or "attention"
```

### 6. Evaluate (5 minutes)
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --output_dir results/
```

### 7. Make Predictions (instant)
```bash
python scripts/predict.py \
  --checkpoint checkpoints/best_model.pth \
  --features data/splits/test_features.npy
```

## 🎯 Expected Performance

| Model | Accuracy | Top-3 | Top-5 | Training Time |
|-------|----------|-------|-------|---------------|
| Random | 1.75% | 5.26% | 8.77% | - |
| FCNN | 25-35% | 50-65% | 60-75% | 2-3 hrs |
| CNN | 30-40% | 55-70% | 65-80% | 8-12 hrs |
| RNN | 35-45% | 60-75% | 70-85% | 6-10 hrs |
| Attention | 35-45% | 60-75% | 70-85% | 10-15 hrs |
| Human (GeoMusic) | 35-45% | 60-70% | - | - |

**Note:** These are goals. Even 25% is valuable (14× better than random)!

## 🔬 Key Features

### Production-Ready Code
- ✅ Error handling throughout
- ✅ Progress saving & resumption
- ✅ Automatic rate limiting
- ✅ Mixed precision training
- ✅ Gradient clipping
- ✅ Early stopping
- ✅ Configurable via YAML

### Research-Ready Features
- ✅ Multiple model architectures
- ✅ Comprehensive evaluation metrics
- ✅ Feature importance (SHAP ready)
- ✅ Attention visualization
- ✅ t-SNE embeddings
- ✅ Confusion matrix analysis
- ✅ Per-class metrics

### Best Practices
- ✅ PyTorch best practices
- ✅ Reproducible (seed setting)
- ✅ Class imbalance handling
- ✅ Proper train/val/test splits
- ✅ Label encoding
- ✅ Data augmentation ready

## 📁 Project Structure

```
music_geo_classifier/
├── models/                      # Model architectures
│   ├── fcnn.py                 # Baseline FC network
│   ├── cnn.py                  # Convolutional network
│   └── rnn.py                  # RNN + Attention
├── utils/                       # Utilities
│   └── dataset.py              # Dataset classes
├── scripts/                     # Executable scripts
│   ├── extract_features.py     # MIRAGE → Spotify features
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Evaluation + plots
│   └── predict.py              # Inference
├── configs/
│   └── config.yaml             # Hyperparameters
├── data/                        # Data directory
│   ├── raw/                    # MIRAGE CSVs
│   ├── features/               # Extracted features
│   ├── splits/                 # Train/val/test
│   └── outputs/                # Results
├── checkpoints/                 # Saved models
├── results/                     # Evaluation outputs
├── notebooks/                   # Jupyter notebooks
├── README.md                    # Main documentation
├── QUICKSTART.md               # Tutorial
├── DATA_PREP.md                # MIRAGE guide
└── requirements.txt            # Dependencies
```

## 💡 Research Applications

### Music Cognition
- What acoustic features predict geographic origin?
- Do humans and AI use the same cues?
- Which countries have distinctive musical signatures?

### Computational Musicology
- Quantify musical globalization
- Track cultural diffusion patterns
- Identify regional music characteristics

### Machine Learning
- Multi-class classification (57 classes)
- Imbalanced dataset handling
- Interpretability (attention, SHAP)
- Model comparison (FCNN vs CNN vs RNN)

## 🔧 Customization

### Add New Models
```python
# In models/your_model.py
class YourModel(nn.Module):
    def __init__(self, ...):
        # Your architecture
        
# In scripts/train.py, add to create_model():
elif model_type == 'your_model':
    return YourModel(...)
```

### Use Different Features
```python
# Modify extract_spotify_features() in scripts/extract_features.py
# Or add extract_librosa_features() for:
# - MFCCs (timbre)
# - Spectral features
# - Chroma (harmony)
# - Rhythm features
```

### Change Split Ratios
```yaml
# In configs/config.yaml
dataset:
  train_split: 0.8  # Instead of 0.7
  val_split: 0.1
  test_split: 0.1
```

## 🐛 Common Issues & Solutions

### "CUDA out of memory"
→ Reduce `batch_size` in config.yaml

### "Spotify API credentials not found"
→ Create `.env` file with credentials

### "Training accuracy stuck at 2%"
→ Check label encoding (should be 0-56, not 1-57)

### "Model not learning"
→ Try different learning rate (0.0001 or 0.01)

### "Training too slow"
→ Enable `mixed_precision: true` in config

See [QUICKSTART.md](QUICKSTART.md) for more troubleshooting.

## 📚 Further Reading

- **README.md** - Full documentation
- **QUICKSTART.md** - Step-by-step tutorial  
- **DATA_PREP.md** - MIRAGE data guide
- **Code comments** - Inline documentation

## 🤝 Contributing

This is research code. Contributions welcome:
- Model improvements
- Feature engineering
- Data augmentation
- Visualization tools
- Documentation improvements

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- **MIRAGE Project** - Global radio broadcast corpus
- **GeoMusic Team** - Human perception experiments
- **MCCL Lab** - University of Michigan
- **FEAST Team 2026** - Research assistants

## 📧 Contact

- **Lab**: Music Computation and Cognition Lab (MCCL)
- **Institution**: University of Michigan
- **Project**: FEAST Team 2026 - MIRAGE & GeoMusic

---

## 🎯 Success Criteria

You'll know this is working when:

✅ Features extract successfully from MIRAGE  
✅ Training loss decreases over epochs  
✅ Validation accuracy > 20% (beats random by 11×)  
✅ Model makes sensible predictions (Indonesia for gamelan music)  
✅ Confusion matrix shows geographic clustering  
✅ High-performing countries are those with distinctive instruments  

## 🚀 Get Started Now!

```bash
# 1. Extract and setup
tar -xzf music_geo_classifier_complete.tar.gz
cd music_geo_classifier && pip install -r requirements.txt

# 2. Download data
wget -P data/raw/ https://zenodo.org/records/18112107/files/MIRAGE.csv

# 3. Set credentials
echo "SPOTIPY_CLIENT_ID='your_id'" > .env
echo "SPOTIPY_CLIENT_SECRET='your_secret'" >> .env

# 4. Extract (test with 1K first)
python scripts/extract_features.py --sample_size 1000

# 5. Train baseline
python scripts/train.py

# 6. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

# 7. Predict
python scripts/predict.py \
  --checkpoint checkpoints/best_model.pth \
  --features data/splits/test_features.npy
```

**That's it! You're classifying music by geography with deep learning! 🎉**

---

**Package Version:** 1.0.0  
**Last Updated:** February 2026  
**Built with:** PyTorch 2.0+, Python 3.8+
