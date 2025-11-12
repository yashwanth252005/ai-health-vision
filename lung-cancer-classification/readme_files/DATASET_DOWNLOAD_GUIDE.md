# ============================================================================
# LUNG CANCER DATASET DOWNLOAD GUIDE
# ============================================================================

## 🎯 DATASET REQUIREMENTS (Based on Journal)

According to the journal (Table 1):
- **Dataset Name**: Lung Cancer
- **Original Images**: 197
- **After Augmentation**: 4000 images
- **Training/Testing Split**: 50%-50% (2000 train / 2000 test)
- **Classes**: 3 (Benign, Malignant, Normal) - based on confusion matrices

---

## 📥 OPTION 1: Kaggle Lung Cancer Datasets (RECOMMENDED)

### **Dataset 1: IQ-OTH/NCCD - Lung Cancer Dataset**
🔗 **Link**: https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset

**Details:**
- Contains lung CT scan images
- Multiple classes including Normal, Benign, Malignant
- Good quality medical images
- **RECOMMENDED** for this project

**How to Download:**

1. **Login to Kaggle** with your account

2. **Method A - Direct Download (GUI):**
   - Go to the dataset link above
   - Click "Download" button (requires Kaggle login)
   - Extract the ZIP file
   - Place images in `data/raw/` folder

3. **Method B - Using Kaggle API (Command Line):**
   ```bash
   # Install Kaggle API
   pip install kaggle
   
   # Setup Kaggle credentials
   # 1. Go to https://www.kaggle.com/settings/account
   # 2. Click "Create New API Token"
   # 3. Download kaggle.json
   # 4. Place it in: C:\Users\YourUsername\.kaggle\kaggle.json
   
   # Download dataset
   kaggle datasets download -d adityamahimkar/iqothnccd-lung-cancer-dataset
   
   # Extract
   unzip iqothnccd-lung-cancer-dataset.zip -d data/raw/
   ```

---

### **Dataset 2: Chest CT-Scan Images Dataset**
🔗 **Link**: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

**Details:**
- 1000 CT scan images
- 4 categories: Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, Normal
- High quality images

---

### **Dataset 3: Lung Cancer Dataset (Multiple Sources)**
🔗 **Link**: https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer

**Details:**
- Combined dataset from multiple sources
- Various lung conditions
- Good for classification tasks

---

## 📥 OPTION 2: Public Medical Datasets

### **The Cancer Imaging Archive (TCIA)**
🔗 **Link**: https://www.cancerimagingarchive.net/

**Instructions:**
1. Search for "Lung" datasets
2. Download LIDC-IDRI or similar datasets
3. Requires registration (free)

---

## 📥 OPTION 3: Use Our Provided Script (Coming Next!)

I will create a Python script that will:
- Automatically download from Kaggle using API
- Organize images into proper folders
- Verify dataset integrity
- Create train/test splits

---

## 🗂️ EXPECTED FOLDER STRUCTURE AFTER DOWNLOAD:

```
data/raw/
├── benign/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── malignant/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── normal/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

---

## ✅ DATASET VERIFICATION

After downloading, the dataset should have:
- **Minimum**: ~200 original images (close to journal's 197)
- **3 Classes**: Benign, Malignant, Normal
- **Format**: JPG, PNG, or DICOM
- **Size**: Varies (we'll resize to 224×224 automatically)

---

## 🚀 NEXT STEPS AFTER DOWNLOAD:

1. Place raw images in `data/raw/` folder
2. Run our data augmentation script (I'll create this next)
3. Script will generate 4000 augmented images automatically
4. Images will be split 50-50 for training and testing

---

## 💡 MY RECOMMENDATION:

**START WITH DATASET 1** (IQ-OTH/NCCD):
- Most similar to journal requirements
- Good quality
- Easy to download
- Proper class labels

---

## ❓ NEED HELP?

If you face any issues:
1. Check if Kaggle account is verified
2. Make sure kaggle.json credentials are in correct location
3. I can create an automated download script for you
4. Alternative: Manual download and I'll help organize files

---

## 📝 IMPORTANT NOTES:

- **Dataset size**: After augmentation, expect ~2-3 GB
- **Original images**: We need around 200 images (journal used 197)
- **Augmentation**: Our script will create flip/rotate versions automatically
- **Classes**: Must have 3 classes (benign, malignant, normal)

---

**Once you download the dataset, let me know and I'll create the preprocessing and augmentation scripts!** 🎯
