# Music Geographic Classification with Deep Learning

A PyTorch implementation of deep neural networks for classifying music by geographic origin using the MIRAGE corpus.

## 🎵 Project Overview

This project implements four deep learning architectures to predict the country of origin for songs based on their acoustic features:
- **FCNN** (Fully Connected Neural Network) - Baseline model using Spotify API features
- **CNN** (Convolutional Neural Network) - For mel spectrogram analysis
- **RNN** (Recurrent Neural Network with LSTM) - For sequential feature analysis  
- **Attention-based RNN** - LSTM with self-attention mechanism

## 📁 Project Structure

```
music_geo_classifier/
├── configs/
│   └── config.yaml              # Configuration file
├── models/
│   ├── __init__.py
│   ├── fcnn.py                  # Fully connected network
│   ├── cnn.py                   # Convolutional network
│   └── rnn.py                   # RNN and Attention models
├── utils/
│   ├── __init__.py
│   ├── dataset.py               # Dataset and dataloader utilities
│   └── feature_extraction.py   # Feature extraction (TBD)
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   └── extract_features.py      # Feature extraction (TBD)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── data/
│   ├── raw/                     # Raw MIRAGE corpus
│   ├── features/                # Extracted features
│   ├── splits/                  # Train/val/test splits
│   └── outputs/                 # Model outputs
├── checkpoints/                 # Saved model checkpoints
├── results/                     # Evaluation results
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd music_geo_classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

**Option A: Using preprocessed data**
```bash
# Place your preprocessed features in data/splits/
data/splits/
├── train_features.npy
├── val_features.npy
├── test_features.npy
├── train_labels.npy
├── val_labels.npy
├── test_labels.npy
└── label_encoder.pkl
```

**Option B: Extract features from MIRAGE corpus**
```bash
# Place MIRAGE corpus CSV in data/raw/
# Run feature extraction (to be implemented)
python scripts/extract_features.py --input data/raw/mirage_corpus.csv --output data/features/
```

### 3. Configure Training

Edit `configs/config.yaml` to set hyperparameters:
```yaml
model:
  type: "fcnn"  # Options: "fcnn", "cnn", "rnn", "attention"
  
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
```

### 4. Train Model

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --data_dir data/splits/
```

**Training will:**
- Load data from `data/splits/`
- Create model based on config
- Train for specified epochs with validation
- Save best model to `checkpoints/best_model.pth`
- Apply early stopping if validation doesn't improve

### 5. Evaluate Model

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data_dir data/splits/ \
  --output_dir results/
```

**Evaluation outputs:**
- `results/metrics.txt` - Overall and per-class metrics
- `results/confusion_matrix.png` - Confusion matrix heatmap
- `results/per_class_accuracy.png` - Per-country accuracy bar plot
- `results/top_confusions.png` - Most confused country pairs

## 📊 Model Architectures

### 1. Fully Connected Neural Network (FCNN)

**Input:** 13 Spotify features (danceability, energy, tempo, etc.)  
**Architecture:**
- Input (13) → FC(256) → ReLU → BatchNorm → Dropout(0.3)
- FC(128) → ReLU → BatchNorm → Dropout(0.3)
- FC(64) → ReLU → BatchNorm → Dropout(0.3)
- FC(57) → Softmax

**Parameters:** ~150,000  
**Training time:** 2-3 hours on GPU

### 2. Convolutional Neural Network (CNN)

**Input:** Mel spectrogram (1 × 128 × 1300)  
**Architecture:**
- Conv2D(32) → BatchNorm → ReLU → MaxPool
- Conv2D(64) → BatchNorm → ReLU → MaxPool
- Conv2D(128) → BatchNorm → ReLU → MaxPool
- Conv2D(256) → BatchNorm → ReLU → AdaptiveAvgPool
- FC(128) → ReLU → Dropout(0.5)
- FC(57) → Softmax

**Parameters:** ~2,000,000  
**Training time:** 8-12 hours on GPU

### 3. RNN with LSTM

**Input:** Sequential features (600 frames × 13 features)  
**Architecture:**
- Bidirectional LSTM(128) × 2 layers
- FC(128) → ReLU → Dropout(0.3)
- FC(57) → Softmax

**Parameters:** ~500,000  
**Training time:** 6-10 hours on GPU

### 4. Attention-based RNN

**Input:** Sequential features (600 frames × 13 features)  
**Architecture:**
- Bidirectional LSTM(128) × 2 layers
- Self-Attention mechanism
- FC(128) → ReLU → Dropout(0.3)
- FC(57) → Softmax

**Parameters:** ~550,000  
**Training time:** 10-15 hours on GPU

## 🎯 Expected Performance

Based on similar music information retrieval tasks:

| Model | Expected Accuracy | Top-5 Accuracy |
|-------|------------------|----------------|
| Random Baseline | 1.75% | 8.77% |
| FCNN | 25-35% | 50-65% |
| CNN | 30-40% | 55-70% |
| RNN/Attention | 35-45% | 60-75% |
| Ensemble | 40-50% | 65-80% |
| Human (GeoMusic) | 35-45% | 60-70% |

**Note:** These are goals, not guarantees. Even 25% is scientifically valuable (14× better than random).

## 🔧 Hyperparameter Tuning

Use Optuna for automated hyperparameter search:

```python
# In scripts/hyperparameter_search.py (to be implemented)
python scripts/hyperparameter_search.py \
  --config configs/config.yaml \
  --n_trials 50 \
  --study_name music_geo_optuna
```

**Tunable parameters:**
- Learning rate: [1e-5, 1e-2]
- Batch size: [16, 32, 64, 128]
- Hidden dimensions: [64, 128, 256, 512]
- Dropout rate: [0.1, 0.3, 0.5]
- Number of layers: [2, 3, 4]

## 📈 Monitoring Training

### Using TensorBoard

```bash
# During training, logs are saved to runs/
tensorboard --logdir=runs/
```

### Using Weights & Biases

1. Sign up at https://wandb.ai
2. Set in `configs/config.yaml`:
```yaml
experiment:
  use_wandb: true
  wandb_project: "music-geography-classification"
  wandb_entity: "your-username"
```

## 🔬 Interpretability & Analysis

### Feature Importance (SHAP)

```python
import shap
from models.fcnn import MusicClassifierFC

# Load model
model = MusicClassifierFC(input_dim=13, num_classes=57)
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])

# Create explainer
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data)

# Visualize
shap.summary_plot(shap_values, test_data, feature_names=spotify_features)
```

### Attention Visualization

```python
from models.rnn import MusicAttentionRNN

model = MusicAttentionRNN(input_dim=13, num_classes=57)
logits, attention_weights = model(input_sequence, return_attention=True)

# Plot attention over time
plt.plot(attention_weights[0].cpu().numpy())
plt.xlabel('Time Step')
plt.ylabel('Attention Weight')
plt.show()
```

### t-SNE Embedding Visualization

```python
from sklearn.manifold import TSNE

# Extract embeddings
embeddings = model.get_embeddings(all_test_data)
tsne = TSNE(n_components=2)
coords = tsne.fit_transform(embeddings.cpu().numpy())

# Plot by country
plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab20')
plt.colorbar()
plt.show()
```

## 🐛 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # Default is 32
```

### Slow Training
```bash
# Enable mixed precision training
training:
  mixed_precision: true
  
# Reduce number of workers
hardware:
  num_workers: 2  # Default is 4
```

### Model Not Learning
- Check learning rate (try 1e-4 or 1e-2)
- Verify data normalization
- Check for NaN values in features
- Try different initialization

### Low Validation Accuracy
- Increase model capacity (more layers/neurons)
- Reduce dropout rate
- Add more training data
- Try data augmentation

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{music_geo_classifier_2026,
  author = {FEAST Team, Music Computation and Cognition Lab},
  title = {Music Geographic Classification with Deep Learning},
  year = {2026},
  publisher = {University of Michigan},
  url = {https://github.com/yourrepo/music-geo-classifier}
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **MIRAGE Project** - For the global radio broadcast corpus
- **GeoMusic Team** - For human perception data
- **MCCL Lab** - Music Computation and Cognition Lab at University of Michigan
- **FEAST Team 2026** - Research assistants and collaborators

## 📧 Contact

For questions or collaboration:
- Lab Website: [MCCL Website]
- Email: [Your Email]
- GitHub Issues: [Open an issue]

---

Built with ❤️ by the FEAST Team at University of Michigan
