# QUICKSTART GUIDE

Get started with music geographic classification in 5 steps!

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- 16GB+ RAM
- MIRAGE corpus data

## Step 1: Installation (5 minutes)

```bash
# Clone repository
cd ~/projects
git clone <your-repo>
cd music_geo_classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 2: Prepare Data (varies)

### Option A: Quick Test with Dummy Data

```bash
# Create dummy data for testing (no MIRAGE corpus needed)
mkdir -p data/splits

python << EOF
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Create dummy features (1000 songs, 13 features)
train_features = np.random.randn(700, 13)
val_features = np.random.randn(150, 13)
test_features = np.random.randn(150, 13)

# Create dummy labels (57 countries)
train_labels = np.random.randint(0, 57, 700)
val_labels = np.random.randint(0, 57, 150)
test_labels = np.random.randint(0, 57, 150)

# Save
np.save('data/splits/train_features.npy', train_features)
np.save('data/splits/val_features.npy', val_features)
np.save('data/splits/test_features.npy', test_features)
np.save('data/splits/train_labels.npy', train_labels)
np.save('data/splits/val_labels.npy', val_labels)
np.save('data/splits/test_labels.npy', test_labels)

# Create label encoder
countries = ['Indonesia', 'Mexico', 'United States', 'Brazil', 'Nigeria',
             'India', 'China', 'Japan', 'Germany', 'France'] + [f'Country{i}' for i in range(47)]
le = LabelEncoder()
le.fit(countries)
with open('data/splits/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✓ Dummy data created in data/splits/")
EOF
```

### Option B: Real MIRAGE Data

```bash
# 1. Place MIRAGE corpus CSV in data/raw/
cp /path/to/mirage_corpus.csv data/raw/

# 2. Extract Spotify features (requires Spotify API credentials)
export SPOTIPY_CLIENT_ID='your_client_id'
export SPOTIPY_CLIENT_SECRET='your_client_secret'

# 3. Run feature extraction script (to be implemented)
python scripts/extract_features.py \
  --input data/raw/mirage_corpus.csv \
  --output data/splits/
```

## Step 3: Train Your First Model (30 min - 2 hours)

### Quick Training (FCNN on CPU)

```bash
# Use small config for fast testing
python scripts/train.py \
  --config configs/config.yaml \
  --data_dir data/splits/

# Monitor output - should see:
# Epoch 1/50
# Training: 100%|████████| ...
# Train Loss: 3.xx, Train Acc: xx%
# Val Loss: 3.xx, Val Acc: xx%
# ✓ New best model! Val Acc: xx%
```

### Full Training (GPU recommended)

Edit `configs/config.yaml`:
```yaml
training:
  num_epochs: 50
  batch_size: 32
  
hardware:
  device: "cuda"  # Use GPU
  num_workers: 4
```

Then train:
```bash
python scripts/train.py --config configs/config.yaml
```

**Expected output:**
- Random baseline: ~1.75% accuracy
- After 10 epochs: ~15-20% accuracy
- After 50 epochs: ~25-35% accuracy (FCNN with Spotify features)

## Step 4: Evaluate Model (5 minutes)

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data_dir data/splits/ \
  --output_dir results/

# Check results
cat results/metrics.txt
open results/confusion_matrix.png      # macOS
xdg-open results/confusion_matrix.png  # Linux
```

**What to look for:**
- Overall accuracy vs. random baseline (1.75%)
- Top-5 accuracy (should be much higher)
- Which countries are easy to classify?
- Which countries are confused with each other?

## Step 5: Make Predictions (2 minutes)

```bash
# Predict on test data
python scripts/predict.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --features data/splits/test_features.npy \
  --top_k 5

# Output:
# Song 1:
#   1. Indonesia: 45.32%
#   2. Malaysia: 18.67%
#   3. Philippines: 12.45%
#   4. Thailand: 8.21%
#   5. Singapore: 5.13%
```

---

## Next Steps

### 🧪 Experiment with Different Models

Try CNN for spectrograms:
```bash
# Edit config.yaml
model:
  type: "cnn"  # Change from "fcnn"

# Re-train
python scripts/train.py --config configs/config.yaml
```

Try RNN for sequential features:
```bash
model:
  type: "rnn"  # or "attention"
```

### 📊 Compare Models

```bash
# Train all models
for model in fcnn cnn rnn attention; do
  sed -i "s/type: .*/type: \"$model\"/" configs/config.yaml
  python scripts/train.py --config configs/config.yaml
  mv checkpoints/best_model.pth checkpoints/best_${model}.pth
done

# Evaluate each
for model in fcnn cnn rnn attention; do
  python scripts/evaluate.py \
    --checkpoint checkpoints/best_${model}.pth \
    --output_dir results/${model}/
done

# Compare results
ls results/*/metrics.txt | xargs -I {} sh -c 'echo "====" && cat {}'
```

### 🔍 Analyze What the Model Learned

```python
# In Python or Jupyter notebook
import torch
import numpy as np
from models.fcnn import MusicClassifierFC

# Load model
model = MusicClassifierFC(input_dim=13, num_classes=57)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Get test features
test_features = np.load('data/splits/test_features.npy')
test_tensor = torch.FloatTensor(test_features)

# Extract embeddings
embeddings = model.get_embeddings(test_tensor)

# Visualize with t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
coords = tsne.fit_transform(embeddings.detach().numpy())

plt.scatter(coords[:, 0], coords[:, 1], c=test_labels, cmap='tab20', alpha=0.6)
plt.colorbar(label='Country')
plt.title('t-SNE of Learned Music Representations')
plt.show()
```

### 🎯 Hyperparameter Tuning

```bash
# Install Optuna
pip install optuna

# Run hyperparameter search (to be implemented)
python scripts/hyperparameter_search.py \
  --n_trials 50 \
  --config configs/config.yaml
```

---

## Troubleshooting Common Issues

### Issue: "CUDA out of memory"

**Solution:**
```yaml
# In configs/config.yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: "Model not learning (accuracy stuck at ~2%)"

**Fixes:**
1. Check if labels are correctly encoded (0 to 56, not 1 to 57)
2. Verify features are normalized
3. Try different learning rate:
```yaml
training:
  learning_rate: 0.0001  # or 0.01
```

### Issue: "Training too slow"

**Solutions:**
1. Enable mixed precision:
```yaml
training:
  mixed_precision: true
```

2. Reduce data loading workers:
```yaml
hardware:
  num_workers: 2
```

3. Use smaller model:
```yaml
model:
  fcnn:
    hidden_dims: [128, 64]  # Instead of [256, 128, 64]
```

### Issue: "Import errors"

**Solution:**
```bash
# Ensure you're in project root and venv is activated
cd /path/to/music_geo_classifier
source venv/bin/activate

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Performance Benchmarks

On a typical setup (NVIDIA RTX 3080, 32GB RAM):

| Task | Time |
|------|------|
| Data preparation | 10-30 min |
| FCNN training (50 epochs) | 2-3 hours |
| CNN training (50 epochs) | 8-12 hours |
| RNN training (50 epochs) | 6-10 hours |
| Evaluation | 2-5 minutes |
| Inference (1000 songs) | < 1 second |

---

## Getting Help

- **Documentation**: See README.md for detailed info
- **Examples**: Check notebooks/ directory
- **Issues**: Open a GitHub issue
- **Contact**: [Your contact info]

---

**You're all set! 🎉**

Try training your first model and see how well it can identify music from around the world!
