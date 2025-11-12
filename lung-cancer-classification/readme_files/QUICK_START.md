# 🚀 QUICK START GUIDE - Lung Cancer Classification

## ⚡ 5-Minute Setup

### Step 1: Open PowerShell
- Press `Win + X`
- Select "Windows PowerShell" or "Terminal"

### Step 2: Navigate to Project
```powershell
cd c:\all\ai-mini-project\lung-cancer-classification
```

### Step 3: Run Setup Script
```powershell
.\setup_windows.ps1
```

**This will:**
- ✅ Check Python installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Verify GPU configuration
- ✅ Setup directory structure

**Time**: ~5-10 minutes

---

## 📥 Download Dataset (Choose One Method)

### Method 1: Kaggle Website (Easiest)
1. Go to: https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
2. Click "Download" (requires Kaggle login)
3. Extract ZIP file
4. Copy folders to `data/raw/`

### Method 2: Kaggle API (Advanced)
```powershell
# Install Kaggle API
pip install kaggle

# Setup credentials (download from Kaggle → Settings → API Token)
# Place kaggle.json in: C:\Users\YourUsername\.kaggle\

# Download dataset
kaggle datasets download -d adityamahimkar/iqothnccd-lung-cancer-dataset

# Extract
Expand-Archive -Path iqothnccd-lung-cancer-dataset.zip -DestinationPath data\raw\
```

---

## 🎨 Generate Augmented Data

```powershell
python utils\data_augmentation.py
```

**This creates:**
- ~4000 augmented images (from ~200 original)
- Flip left, flip right, rotate 90° operations
- Organized in `data/augmented/` folder

**Time**: ~2-5 minutes

---

## 🧠 Train Models

### Option A: Local Training (Your RTX 2050)

```powershell
# Step 1: Train IRCNN (94-layer model)
python training\train_ircnn.py
# Time: ~8-12 minutes

# Step 2: Train SACNN (84-layer model)
python training\train_sacnn.py
# Time: ~8-12 minutes

# Step 3: Train SWNN Classifier
python training\train_swnn.py
# Time: ~2-3 minutes

# Total: ~20-30 minutes
```

### Option B: Kaggle Training (Faster, Recommended)

1. **Upload to Kaggle:**
   - Create new Kaggle Notebook
   - Upload all `.py` files
   - Upload `config/config.yaml`
   - Upload augmented data (or use Kaggle dataset directly)

2. **Enable GPU:**
   - Settings → Accelerator → GPU (P100)

3. **Run training:**
   ```python
   # In Kaggle notebook
   !python training/train_ircnn.py
   !python training/train_sacnn.py
   !python training/train_swnn.py
   ```

4. **Download models:**
   - Download from `saved_models/` folder
   - Place in your local `saved_models/` directory

**Time on Kaggle**: ~10-15 minutes total

---

## 📊 Evaluate Results

```powershell
python evaluation\evaluate_model.py
```

**Generates:**
- ✅ Confusion matrix
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Classification report
- ✅ GradCAM visualizations

**Results saved in**: `results/` folder

---

## 🎯 Expected Results

| Metric | Target |
|--------|--------|
| Accuracy | 95.0% |
| Precision | 95.0% |
| Sensitivity | 95.0% |
| F1-Score | 95.0% |

---

## 🐛 Common Issues & Solutions

### Issue 1: GPU Not Detected
**Solution:**
```powershell
# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install CUDA:
# https://developer.nvidia.com/cuda-11-8-0-download-archive
```

### Issue 2: Out of Memory Error
**Solution:**
```yaml
# Edit config/config.yaml
training:
  batch_size_local: 8  # Reduce from 16
```

### Issue 3: Dataset Not Found
**Solution:**
```
Ensure structure:
data/raw/
├── benign/
├── malignant/
└── normal/
```

### Issue 4: Slow Training
**Normal:**
- Local (RTX 2050): 20-30 min
- Kaggle (P100): 10-15 min

**Solution:** Use Kaggle for faster training

---

## 📝 Workflow Checklist

- [ ] Run `setup_windows.ps1`
- [ ] Download dataset
- [ ] Extract to `data/raw/`
- [ ] Run `data_augmentation.py`
- [ ] Train IRCNN
- [ ] Train SACNN
- [ ] Train SWNN
- [ ] Evaluate results
- [ ] View confusion matrix
- [ ] Check GradCAM visualizations

---

## 💡 Pro Tips

1. **Save Time**: Train on Kaggle (free GPU)
2. **Monitor Training**: Check `results/logs/training.log`
3. **GPU Memory**: Use batch size 16 (not 64) on RTX 2050
4. **Early Stopping**: Training auto-stops if no improvement
5. **Checkpoints**: Models saved every 5 epochs

---

## 📚 Key Files to Know

| File | Purpose |
|------|---------|
| `config/config.yaml` | All settings & hyperparameters |
| `README.md` | Complete documentation |
| `DATASET_DOWNLOAD_GUIDE.md` | How to get data |
| `requirements.txt` | Python dependencies |

---

## 🎓 Learning Path

1. **Day 1**: Setup + Download data
2. **Day 2**: Understand augmentation + Run it
3. **Day 3**: Study IRCNN architecture
4. **Day 4**: Study SACNN architecture
5. **Day 5**: Train models (local or Kaggle)
6. **Day 6**: Evaluate & visualize results

---

## ✅ Success Criteria

You're done when you have:
- ✅ Trained models in `saved_models/`
- ✅ Accuracy ~95% in results
- ✅ Confusion matrix showing good predictions
- ✅ GradCAM showing correct focus regions

---

## 🆘 Need Help?

1. Check logs: `results/logs/training.log`
2. Verify GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
3. Read README.md for detailed info
4. Check configuration in `config/config.yaml`

---

**Ready? Start with `setup_windows.ps1`! 🚀**
