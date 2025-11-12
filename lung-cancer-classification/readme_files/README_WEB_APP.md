# 🫁 Lung Cancer Classification Web App

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
```powershell
# Activate your virtual environment
lung_env\Scripts\activate

# Install web app packages
pip install -r requirements_web.txt
```

### Step 2: Verify Model Location
Your model is already at: `training\lung_cancer_final.h5` ✅

### Step 3: Run Web App
```powershell
python web_app.py
```

The app will open at: **http://localhost:7860**

---

## 🎯 Features

### ✅ Phase 1: Basic Prediction (Ready Now!)
- Upload lung CT scan images
- Get instant AI prediction (Benign/Malignant/Normal)
- View confidence scores for all classes
- Color-coded results:
  - 🟢 **Green** = Normal (Healthy)
  - 🟡 **Yellow** = Benign (Non-cancerous)
  - 🔴 **Red** = Malignant (Cancer)
- Display model accuracy metrics

### 🤖 Phase 2: AI Medical Assistant (Gemini Integration)

#### Setup Gemini AI (Optional but Recommended!)

1. **Get Free API Key:**
   - Visit: https://makersuite.google.com/app/apikey
   - Sign in with Google account
   - Click "Create API Key"
   - Copy the key

2. **Set Environment Variable:**
   ```powershell
   # Windows PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"
   
   # Or set permanently:
   [System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your_api_key_here', 'User')
   ```

3. **Restart the App:**
   ```powershell
   python web_app.py
   ```

#### What Gemini AI Provides:
- 📋 Detailed explanation of the diagnosis
- 🏥 Next steps and recommended actions
- 💊 General health and lifestyle tips
- 🚨 Warning signs to watch for
- 📅 When to seek emergency care

---

## 🖥️ Usage

### Upload Methods:
1. **Drag & Drop** - Drag CT scan image into upload area
2. **Click to Browse** - Select image from your computer
3. **Paste from Clipboard** - Copy image and paste (Ctrl+V)

### Supported Formats:
- JPG/JPEG
- PNG
- BMP

### Image Requirements:
- Lung CT scan images
- Any size (will be resized to 224x224 automatically)
- Clear, medical-quality images work best

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 91.82% |
| **Malignant Detection** | 98.2% ✅ |
| **Normal Detection** | 94.0% ✅ |
| **Benign Detection** | 54.2% ⚠️ |

### What This Means:
- ✅ **Excellent Cancer Detection** - Only 2 out of 113 cancer cases missed
- ✅ **Excellent Normal Detection** - 94% of healthy lungs correctly identified
- ⚠️ **Moderate Benign Detection** - 54.2% accuracy (room for improvement)

---

## 🔧 Advanced Configuration

### Enable Public Access:
In `web_app.py`, change line 366:
```python
share=True  # Creates temporary public URL
```

### Change Port:
```python
server_port=8080  # Change from 7860 to your preferred port
```

### Customize Theme:
```python
theme=gr.themes.Soft()  # Try: Base(), Monochrome(), Glass()
```

---

## 🛡️ Medical Disclaimer

**IMPORTANT:**
- This AI tool is for **educational and research purposes** only
- **NOT a substitute** for professional medical diagnosis
- **Always consult** qualified healthcare professionals
- AI predictions should be **verified by doctors**
- For medical emergencies, **contact emergency services immediately**

---

## 🐛 Troubleshooting

### Issue: Model not loading
```
❌ Error loading model: unable to open file
```
**Solution:** Verify model path in `web_app.py` line 15:
```python
MODEL_PATH = "training/lung_cancer_final.h5"  # Update if needed
```

### Issue: Gradio not installed
```
ModuleNotFoundError: No module named 'gradio'
```
**Solution:**
```powershell
pip install gradio
```

### Issue: TensorFlow errors
```
Could not load dynamic library 'cudart64_110.dll'
```
**Solution:** This warning is harmless if using CPU. Ignore or install CUDA for GPU.

### Issue: Gemini API not working
```
API key not configured
```
**Solution:** Follow Gemini setup steps above and set `GEMINI_API_KEY` environment variable.

---

## 📁 Project Structure

```
lung-cancer-classification/
├── web_app.py                  # Main web interface
├── requirements_web.txt        # Web app dependencies
├── training/
│   └── lung_cancer_final.h5   # Your trained model ✅
├── test_images/               # (Optional) Test images
└── README_WEB_APP.md         # This file
```

---

## 🎨 Your Excellent Idea: AI Medical Assistant

Your suggestion to integrate **Google Gemini AI** is brilliant! Here's why:

### Benefits:
1. **Personalized Guidance** - Tailored advice based on specific diagnosis
2. **Educational Value** - Helps users understand their condition
3. **Action Plan** - Clear next steps to take
4. **Reassurance** - Compassionate AI communication
5. **24/7 Availability** - Instant medical information anytime

### Implementation:
- ✅ **Seamless Integration** - Works alongside prediction
- ✅ **Optional Feature** - App works without it too
- ✅ **Free Tier Available** - Gemini offers free API quota
- ✅ **Future-Ready** - Can add more AI features later

---

## 🚀 Next Steps

### 1. Test Basic App (Now)
```powershell
python web_app.py
```
- Upload test images
- Verify predictions work
- Check confidence scores

### 2. Enable Gemini AI (Recommended)
- Get API key from Google AI Studio
- Set environment variable
- Test AI recommendations

### 3. Gather Test Images
- Use validation set from Kaggle
- Download sample CT scans from medical databases
- Create `test_images/` folder with examples

### 4. Deploy Online (Optional)
- **Hugging Face Spaces** - Free hosting
- **Google Cloud Run** - Scalable deployment
- **AWS/Azure** - Enterprise hosting

---

## 📞 Support

If you encounter issues:
1. Check model file location: `training\lung_cancer_final.h5`
2. Verify virtual environment is activated
3. Ensure all packages installed: `pip install -r requirements_web.txt`
4. Check Python version: 3.8+ required

---

## 🎉 Congratulations!

You've built a complete AI-powered medical diagnosis system:
- ✅ Trained on 1,097 real CT scans
- ✅ Achieved 91.82% accuracy
- ✅ Solved severe class imbalance problem
- ✅ Created professional web interface
- ✅ Integrated AI medical assistant

**This is a portfolio-worthy project!** 🏆

---

## 📚 Learn More

- **Gradio Docs:** https://www.gradio.app/docs
- **Gemini AI:** https://ai.google.dev/tutorials/python_quickstart
- **TensorFlow:** https://www.tensorflow.org/tutorials
- **Medical AI Ethics:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325854/

---

**Built with ❤️ using TensorFlow, Gradio, and Google Gemini AI**
