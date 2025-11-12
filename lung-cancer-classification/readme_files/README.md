# 🫁 Lung Cancer Classification using Deep Learning

## 📋 Project Overview

This project implements an **AI-based lung cancer classification system** using a novel framework of **Inverted Residual and Self-Attention Deep Neural Network architectures**, based on the research paper published in *Scientific Reports* (2025).

**DOI**: 10.1038/s41598-025-93718-7

### 🎯 Objective

Classify lung CT scan images into three categories:
- **Benign** (non-cancerous)
- **Malignant** (cancerous)
- **Normal** (healthy)

### 📊 Expected Performance (from Journal)

| Metric | Target Value |
|--------|-------------|
| **Accuracy** | 95.0% |
| **Precision** | 95.0% |
| **Sensitivity** | 95.0% |
| **F1-Score** | 95.0% |

---

## 🏗️ Architecture Overview

### Two Custom Deep Learning Models:

1. **94-Layered Inverted Residual CNN (IRCNN)**
   - 5.3 million parameters
   - Lightweight inverted residual blocks
   - Output: N × 1282 feature vector

2. **84-Layered Self-Attention CNN (SACNN)**
   - 7.5 million parameters
   - 17 convolutional layers
   - Self-attention mechanism
   - Output: N × 1406 feature vector

### Complete Pipeline:

```
Input Image (224×224×3)
    ↓
Data Augmentation (Flip, Rotate)
    ↓
┌─────────────────────┬──────────────────────┐
│   IRCNN Training    │   SACNN Training     │
│   (94 layers)       │   (84 layers)        │
└──────────┬──────────┴──────────┬───────────┘
           │                     │
    Feature Extraction    Feature Extraction
    (GAP: 1282 dims)      (Attention: 1406 dims)
           │                     │
           └──────────┬──────────┘
                      │
          Feature Fusion (Serial Correlation)
          (Pearson Correlation Coefficient)
                      │
          SSA Optimization (200 iterations)
          (Salp Swarm Algorithm)
                      │
          SWNN Classifier (Shallow Wide NN)
                      │
          Final Prediction (3 classes)
```

---

## 💻 System Requirements

### Minimum Requirements:
- **OS**: Windows 10/11, Linux, or macOS
- **CPU**: Intel i5 or equivalent
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with 4GB+ VRAM (RTX 2050 or better)
- **Storage**: 20 GB free space
- **Python**: 3.8 or higher

### Recommended (for faster training):
- **CPU**: Intel i7 13th Gen or higher
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3060 (12GB) or better
- **Storage**: 50 GB SSD

### Your System:
- ✅ **OS**: Windows 11
- ✅ **CPU**: Intel i5 13th Gen
- ✅ **RAM**: 16 GB
- ✅ **GPU**: NVIDIA RTX 2050 (4GB VRAM)
- ✅ **Storage**: 100 GB available
- ✅ **Python**: 3.13.1

**Status**: Your system is suitable! Optimizations applied for 4GB VRAM.

---

## 📁 Project Structure

```
lung-cancer-classification/
├── config/
│   └── config.yaml                 # All hyperparameters and settings
├── data/
│   ├── raw/                        # Original downloaded images
│   ├── augmented/                  # Augmented images (~4000)
│   └── processed/                  # Preprocessed data
├── models/
│   ├── inverted_residual.py        # 94-layer IRCNN
│   ├── self_attention.py           # 84-layer SACNN
│   └── swnn_classifier.py          # Final classifier
├── utils/
│   ├── data_augmentation.py        # Flip, rotate operations
│   ├── feature_fusion.py           # Pearson correlation fusion
│   ├── ssa_optimization.py         # Salp Swarm Algorithm
│   └── gradcam_viz.py              # GradCAM visualization
├── training/
│   ├── train_ircnn.py              # Train IRCNN model
│   ├── train_sacnn.py              # Train SACNN model
│   └── train_swnn.py               # Train final classifier
├── evaluation/
│   ├── evaluate_model.py           # Performance metrics
│   └── confusion_matrix.py         # Generate confusion matrices
├── notebooks/
│   └── complete_pipeline.ipynb     # Jupyter notebook tutorial
├── saved_models/                   # Trained model weights
├── results/                        # Outputs, plots, logs
├── requirements.txt                # Python dependencies
├── DATASET_DOWNLOAD_GUIDE.md       # How to get the dataset
└── README.md                       # This file
```

---

## 🚀 Quick Start Guide

### Step 1: Clone/Setup Project

The project structure is already created in:
```
c:\all\ai-mini-project\lung-cancer-classification\
```

### Step 2: Install Dependencies

Open **PowerShell** or **Command Prompt** and navigate to project directory:

```powershell
cd c:\all\ai-mini-project\lung-cancer-classification
```

Create virtual environment (recommended):
```powershell
python -m venv lung_env
.\lung_env\Scripts\Activate.ps1
```

Install required packages:
```powershell
pip install -r requirements.txt
```

### Step 3: Setup GPU (NVIDIA RTX 2050)

Install CUDA and cuDNN for GPU acceleration:
1. Download CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Download cuDNN 8.6: https://developer.nvidia.com/cudnn
3. Verify installation:
```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output: List showing your GPU

### Step 4: Download Dataset

Follow the detailed guide in `DATASET_DOWNLOAD_GUIDE.md`

**Quick Option:**
- Go to: https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
- Download and extract to `data/raw/`

### Step 5: Augment Data

```powershell
python utils/data_augmentation.py
```

This will:
- Read images from `data/raw/`
- Apply flip and rotate operations
- Generate ~4000 augmented images
- Save to `data/augmented/`

### Step 6: Train Models

**Option A: Train Locally (Your RTX 2050)**

```powershell
# Train Inverted Residual CNN
python training/train_ircnn.py

# Train Self-Attention CNN
python training/train_sacnn.py

# Train Final Classifier
python training/train_swnn.py
```

**Option B: Train on Kaggle (Recommended for faster results)**

1. Upload code to Kaggle Notebook
2. Enable GPU (P100)
3. Run training scripts
4. Download trained models

### Step 7: Evaluate Model

```powershell
python evaluation/evaluate_model.py
```

This generates:
- Confusion matrix
- Precision, Recall, F1-Score
- GradCAM visualizations
- Classification report

---

## 📊 Data Augmentation

Based on **Journal Figure 3**, we implement:

| Operation | Description | Formula (from Journal) |
|-----------|-------------|----------------------|
| **Flip Left** | Horizontal flip | T(x,y) = Px(H+1-y) |
| **Flip Right** | Horizontal flip right | T(x,y) = Px(x+1-h) |
| **Rotate 90** | 90° clockwise rotation | T(x,y) = [Cos90 -Sin90; Sin90 Cos90][T; T1] |

Each original image → 4 versions (original + 3 augmented)

---

## 🧠 Model Architectures

### 1. Inverted Residual CNN (IRCNN)

```python
Input (224×224×3)
    ↓
Stem Convolution (32 filters)
    ↓
Inverted Residual Blocks:
  - Block 1: 16 filters, 1 block
  - Block 2: 24 filters, 2 blocks (parallel)
  - Block 3: 32 filters, 3 blocks (parallel)
  - Block 4: 64 filters, 4 blocks (serial)
  - Block 5: 96 filters, 3 blocks (parallel)
  - Block 6: 160 filters, 3 blocks (parallel)
  - Block 7: 320 filters, 1 block (serial)
    ↓
Final Conv (1280 filters)
    ↓
Global Average Pooling
    ↓
Output: 1282-dimensional feature vector
```

### 2. Self-Attention CNN (SACNN)

```python
Input (224×224×3)
    ↓
Initial Convolution
    ↓
Self-Attention Residual Blocks (7 blocks):
  - Each block has 4 parallel residual blocks
  - Self-attention mechanism between blocks
  - Captures long-range dependencies
    ↓
Global Average Pooling
    ↓
Output: 1406-dimensional feature vector
```

---

## 🔗 Feature Fusion

**Serial-based Strong Correlation** using Pearson Correlation Coefficient:

```
r(U,V) = Σ(Ui-Ū)(Vi-V̄) / √[Σ(Ui-Ū)² × Σ(Vi-V̄)²]
```

Where:
- U = Features from IRCNN (1282 dimensions)
- V = Features from SACNN (1406 dimensions)
- Output = Fused features (2688 dimensions)

---

## 🐚 Salp Swarm Algorithm (SSA)

Optimization algorithm to select best features:

**Parameters:**
- Population size: 30
- Iterations: 200
- Fitness function: Standard Error Mean (SEM)

**Process:**
1. Initialize salp chain
2. Update leader position (exploration)
3. Update follower positions (exploitation)
4. Compute fitness
5. Select best features

---

## 📈 Training Configuration

Based on **Journal Page 7**:

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.00021 |
| **Momentum** | 0.701 |
| **Batch Size** | 16 (local), 64 (Kaggle) |
| **Epochs** | 100 (with early stopping) |
| **Loss Function** | Categorical Crossentropy |
| **Mixed Precision** | Enabled (for 4GB GPU) |

---

## 🎨 GradCAM Visualization

**Grad-CAM** (Gradient-weighted Class Activation Mapping) shows which regions of the image the model focuses on:

- Helps interpret model decisions
- Validates that model looks at relevant areas
- Journal shows >90% correct region prediction

---

## 📊 Results Format

After training, you'll get:

### 1. **Confusion Matrix**
```
                Predicted
              B    M    N
Actual  B  [###]  [ ]  [ ]
        M  [ ]  [###]  [ ]
        N  [ ]  [ ]  [###]
```

### 2. **Metrics Table**
| Metric | Value |
|--------|-------|
| Accuracy | 95.0% |
| Precision | 95.0% |
| Sensitivity | 95.0% |
| F1-Score | 95.0% |

### 3. **GradCAM Visualizations**
- Original image
- Heatmap overlay
- Predicted class
- Confidence score

---

## 🔧 Troubleshooting

### GPU Out of Memory Error

If you see `ResourceExhaustedError`:

**Solution 1**: Reduce batch size
```yaml
# In config/config.yaml
training:
  batch_size_local: 8  # Reduce from 16 to 8
```

**Solution 2**: Enable memory growth (already configured)
```python
# Automatic in our scripts
tf.config.experimental.set_memory_growth(gpu, True)
```

**Solution 3**: Use Kaggle GPU instead (16GB VRAM)

### Dataset Not Found

Ensure images are in correct structure:
```
data/raw/
├── benign/
├── malignant/
└── normal/
```

### Slow Training

- **Local (RTX 2050)**: ~15-20 minutes expected
- **Kaggle (P100)**: ~10 minutes expected

---

## 📚 References

1. **Main Paper**: Aftab, J., Khan, M. A., Arshad, S., et al. (2025). Artificial intelligence based classification and prediction of medical imaging using a novel framework of inverted and self-attention deep neural network architecture. *Scientific Reports*, 15(1). DOI: 10.1038/s41598-025-93718-7

2. **Dataset**: IQ-OTH/NCCD Lung Cancer Dataset (Kaggle)

3. **TensorFlow**: https://www.tensorflow.org/

4. **Grad-CAM**: Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

---

## 📝 Notes

- This implementation is optimized for **Windows 11** and **RTX 2050 4GB**
- All hyperparameters match the **journal specifications exactly**
- Code is heavily **commented** for learning purposes
- Hybrid approach: **Develop locally**, **Train on Kaggle** (optional)

---

## 🎓 Learning Resources

- **Deep Learning**: [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- **Medical Imaging**: [Kaggle Medical Imaging](https://www.kaggle.com/learn/computer-vision)
- **Model Interpretability**: [GradCAM Guide](https://keras.io/examples/vision/grad_cam/)

---

## ✅ Checklist

- [x] Project structure created
- [x] Configuration file set up
- [x] Dependencies listed
- [ ] Dataset downloaded
- [ ] Data augmented
- [ ] IRCNN trained
- [ ] SACNN trained
- [ ] Features fused
- [ ] SSA optimization done
- [ ] SWNN classifier trained
- [ ] Model evaluated
- [ ] Results analyzed

---

## 📧 Support

If you encounter any issues:
1. Check configuration in `config/config.yaml`
2. Verify GPU is detected: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
3. Review logs in `results/logs/`
4. Check dataset structure

---

## 🏆 Expected Outcomes

Upon completion, you will have:
- ✅ Trained IRCNN model (94 layers, 5.3M params)
- ✅ Trained SACNN model (84 layers, 7.5M params)
- ✅ Optimized feature set (via SSA)
- ✅ Final SWNN classifier
- ✅ 95% classification accuracy
- ✅ Confusion matrices and metrics
- ✅ GradCAM visualizations
- ✅ Complete evaluation report

---

**Ready to classify lung cancer with AI! 🚀🫁**

**Next Step**: Follow `DATASET_DOWNLOAD_GUIDE.md` to get the data!
