"""
============================================================================
PACKAGE VERIFICATION SCRIPT
============================================================================
Verify all required packages are installed and importable
Run this after setup to ensure everything is working
============================================================================
"""

import sys

print("=" * 70)
print("VERIFYING PACKAGE INSTALLATIONS")
print("=" * 70)

# Track success/failure
all_success = True
failed_imports = []

# Test 1: Core packages
print("\n1. Testing Core Packages...")
try:
    import numpy as np
    print(f"   ✅ numpy: {np.__version__}")
except ImportError as e:
    print(f"   ❌ numpy: FAILED - {e}")
    all_success = False
    failed_imports.append("numpy")

try:
    import pandas as pd
    print(f"   ✅ pandas: {pd.__version__}")
except ImportError as e:
    print(f"   ❌ pandas: FAILED - {e}")
    all_success = False
    failed_imports.append("pandas")

# Test 2: TensorFlow and Keras
print("\n2. Testing TensorFlow/Keras...")
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"   ✅ tensorflow: {tf.__version__}")
    print(f"   ✅ keras: {keras.__version__}")
except ImportError as e:
    print(f"   ❌ tensorflow/keras: FAILED - {e}")
    all_success = False
    failed_imports.append("tensorflow")

# Test 3: Image Processing
print("\n3. Testing Image Processing...")
try:
    import cv2
    print(f"   ✅ opencv-python (cv2): {cv2.__version__}")
except ImportError as e:
    print(f"   ❌ opencv-python: FAILED - {e}")
    all_success = False
    failed_imports.append("opencv-python")

try:
    from PIL import Image
    print(f"   ✅ pillow (PIL): installed")
except ImportError as e:
    print(f"   ❌ pillow: FAILED - {e}")
    all_success = False
    failed_imports.append("pillow")

# Test 4: Configuration and Data
print("\n4. Testing Configuration/Data...")
try:
    import yaml
    print(f"   ✅ PyYAML (yaml): installed")
except ImportError as e:
    print(f"   ❌ PyYAML: FAILED - {e}")
    all_success = False
    failed_imports.append("PyYAML")

try:
    from tqdm import tqdm
    print(f"   ✅ tqdm: installed")
except ImportError as e:
    print(f"   ❌ tqdm: FAILED - {e}")
    all_success = False
    failed_imports.append("tqdm")

# Test 5: Machine Learning
print("\n5. Testing Machine Learning...")
try:
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    print(f"   ✅ scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"   ❌ scikit-learn: FAILED - {e}")
    all_success = False
    failed_imports.append("scikit-learn")

# Test 6: Visualization
print("\n6. Testing Visualization...")
try:
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"   ✅ matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"   ❌ matplotlib: FAILED - {e}")
    all_success = False
    failed_imports.append("matplotlib")

try:
    import seaborn as sns
    print(f"   ✅ seaborn: installed")
except ImportError as e:
    print(f"   ❌ seaborn: FAILED - {e}")
    all_success = False
    failed_imports.append("seaborn")

# Test 7: Project Modules
print("\n7. Testing Project Modules...")
try:
    from models.inverted_residual import InvertedResidualCNN
    print(f"   ✅ models.inverted_residual: OK")
except ImportError as e:
    print(f"   ❌ models.inverted_residual: FAILED - {e}")
    all_success = False

try:
    from models.self_attention import SelfAttentionCNN
    print(f"   ✅ models.self_attention: OK")
except ImportError as e:
    print(f"   ❌ models.self_attention: FAILED - {e}")
    all_success = False

try:
    from models.swnn_classifier import SWNNClassifier
    print(f"   ✅ models.swnn_classifier: OK")
except ImportError as e:
    print(f"   ❌ models.swnn_classifier: FAILED - {e}")
    all_success = False

try:
    from utils.data_augmentation import LungCancerDataAugmentation
    print(f"   ✅ utils.data_augmentation: OK")
except ImportError as e:
    print(f"   ❌ utils.data_augmentation: FAILED - {e}")
    all_success = False

try:
    from utils.feature_fusion import SerialFeatureFusion
    print(f"   ✅ utils.feature_fusion: OK")
except ImportError as e:
    print(f"   ❌ utils.feature_fusion: FAILED - {e}")
    all_success = False

try:
    from utils.ssa_optimization import SalpSwarmOptimizer
    print(f"   ✅ utils.ssa_optimization: OK")
except ImportError as e:
    print(f"   ❌ utils.ssa_optimization: FAILED - {e}")
    all_success = False

# Final Summary
print("\n" + "=" * 70)
if all_success:
    print("✅ ALL PACKAGES VERIFIED SUCCESSFULLY!")
    print("=" * 70)
    print("\n🚀 Your environment is ready for training!")
    print("\nNext steps:")
    print("1. Make sure dataset is in data/raw/")
    print("2. Run: python training/train_complete_pipeline.py")
    sys.exit(0)
else:
    print("❌ SOME PACKAGES FAILED TO IMPORT")
    print("=" * 70)
    print(f"\nFailed imports: {', '.join(failed_imports)}")
    print("\nTo fix, run:")
    print(f"   pip install {' '.join(failed_imports)}")
    sys.exit(1)
