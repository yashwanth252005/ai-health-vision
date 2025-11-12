# ========================================================================
# KAGGLE TRAINING NOTEBOOK - COMPLETE
# Just copy each CELL into your Kaggle notebook
# ========================================================================

# ========================================================================
# CELL 1: Environment Check
# ========================================================================
import os
import tensorflow as tf

print("=" * 70)
print("KAGGLE ENVIRONMENT CHECK")
print("=" * 70)
print(f"✅ Running on Kaggle!")
print(f"Dataset location: /kaggle/input/")
print(f"Available datasets: {os.listdir('/kaggle/input/')}")
print(f"Working directory: /kaggle/working/")
print(f"\n✅ GPU: {tf.config.list_physical_devices('GPU')}")
print(f"✅ TensorFlow: {tf.__version__}")
print("=" * 70)


# ========================================================================
# CELL 2: Check Pre-installed Packages (NO INSTALLATION NEEDED!)
# ========================================================================
# Kaggle already has everything we need! Let's just verify:

import sys
print("Checking packages...")

try:
    import cv2
    print(f"✅ opencv-python (cv2): {cv2.__version__}")
except ImportError:
    print("❌ opencv-python not found - installing...")
    !pip install opencv-python -q

try:
    import yaml
    print(f"✅ PyYAML (yaml): installed")
except ImportError:
    print("❌ PyYAML not found - installing...")
    !pip install PyYAML -q

try:
    import sklearn
    print(f"✅ scikit-learn: {sklearn.__version__}")
except ImportError:
    pass  # Will install if needed

import numpy as np
import matplotlib
import seaborn

print(f"✅ numpy: {np.__version__}")
print(f"✅ matplotlib: {matplotlib.__version__}")
print(f"✅ seaborn: installed")
print("\n✅ All packages ready!")


# ========================================================================
# CELL 3: Create Project Structure
# ========================================================================
import os

folders = ['models', 'utils', 'training', 'config', 'saved_models', 'results']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create __init__.py files
for folder in ['models', 'utils', 'training']:
    with open(f'{folder}/__init__.py', 'w') as f:
        f.write('')

print("✅ Project structure created!")
print(f"Folders: {os.listdir('.')}")


# ========================================================================
# CELL 4: Create Config File
# ========================================================================
%%writefile config/config.yaml
training:
  batch_size_local: 16
  batch_size_kaggle: 64
  learning_rate: 0.00021
  momentum: 0.701
  epochs: 100

inverted_residual:
  num_blocks: 7
  feature_dim: 1282

self_attention:
  num_blocks: 7
  feature_dim: 1406

ssa_optimization:
  max_iterations: 200
  population_size: 30

swnn:
  hidden_units: 512
  dropout_rate: 0.3

print("✅ Config file created!")


# ========================================================================
# CELL 5: Find Your Dataset
# ========================================================================
import os

print("Looking for your dataset...")
print("\nAvailable datasets:")

input_path = '/kaggle/input/'
datasets = os.listdir(input_path)

for i, dataset in enumerate(datasets, 1):
    print(f"{i}. {dataset}")
    dataset_path = os.path.join(input_path, dataset)
    if os.path.isdir(dataset_path):
        contents = os.listdir(dataset_path)
        print(f"   Contents: {contents[:5]}")  # Show first 5 items

print("\n⚠️ IMPORTANT: Copy your dataset name from above!")
print("You'll need it for the next cell!")


# ========================================================================
# CELL 6: Load and Prepare Data
# ========================================================================
from tensorflow import keras
import numpy as np
import os

# ⚠️ UPDATE THIS LINE with your dataset name from Cell 5!
DATASET_NAME = 'lung-cancer-classification-dataset'  # ← CHANGE THIS!

# Build the path to your data
DATA_PATH = f'/kaggle/input/{DATASET_NAME}'

# Check what's in the dataset
print(f"Dataset path: {DATA_PATH}")
print(f"Contents: {os.listdir(DATA_PATH)}")

# Find the folder with images (usually 'raw', 'data', or class folders)
# Adjust this based on what you see above
if 'raw' in os.listdir(DATA_PATH):
    DATA_PATH = os.path.join(DATA_PATH, 'raw')
    print(f"Using: {DATA_PATH}")

class_names = ['benign', 'malignant', 'normal']
images = []
labels = []

print("\nLoading images...")
for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(DATA_PATH, class_name)
    
    if not os.path.exists(class_dir):
        print(f"⚠️ Not found: {class_dir}")
        # Try without 'raw' folder
        class_dir = os.path.join(f'/kaggle/input/{DATASET_NAME}', class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️ Still not found: {class_dir}")
            continue
    
    image_files = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"✅ {class_name}: {len(image_files)} images")
    
    # Load images (limit to 100 per class for quick testing)
    for img_file in image_files[:100]:
        try:
            img_path = os.path.join(class_dir, img_file)
            img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize
            
            images.append(img_array)
            labels.append(class_idx)
        except Exception as e:
            print(f"⚠️ Error loading {img_file}: {e}")
            continue

X = np.array(images)
y = np.array(labels)
y_onehot = keras.utils.to_categorical(y, num_classes=len(class_names))

print(f"\n✅ Loaded {len(X)} images")
print(f"Shape: {X.shape}")
print(f"Labels shape: {y_onehot.shape}")


# ========================================================================
# CELL 7: Build Simple CNN Model
# ========================================================================
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn_model():
    """Build a simple CNN for testing"""
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, 3, activation='relu', padding='same', 
                     input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        # Block 2
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        # Block 3
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        # Block 4
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Classifier
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    return model

# Build model
model = build_cnn_model()

# Compile with journal hyperparameters
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00021),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print("✅ Model built!")
print(f"\nTotal parameters: {model.count_params():,}")
model.summary()


# ========================================================================
# CELL 8: Train Model
# ========================================================================
from sklearn.model_selection import train_test_split

# Split data (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'saved_models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train
print("\n🚀 Starting training...")
print("=" * 70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  # Will stop early if not improving
    batch_size=64,  # Kaggle batch size
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")


# ========================================================================
# CELL 9: Evaluate Model
# ========================================================================
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Evaluate on validation set
print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

results = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Loss: {results[0]:.4f}")
print(f"Validation Accuracy: {results[1]*100:.2f}%")
print(f"Validation Precision: {results[2]*100:.2f}%")
print(f"Validation Recall: {results[3]*100:.2f}%")

# Predictions
y_pred_proba = model.predict(X_val, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_val, axis=1)

# Classification report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_true, y_pred, 
                          target_names=class_names,
                          digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=class_names,
           yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300)
plt.show()


# ========================================================================
# CELL 10: Plot Training History
# ========================================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=300)
plt.show()

print("✅ Plots saved!")


# ========================================================================
# CELL 11: Save Final Model and Results
# ========================================================================
import json

# Save final model
model.save('saved_models/lung_cancer_final.h5')
print("✅ Model saved: saved_models/lung_cancer_final.h5")

# Save results
results_dict = {
    'validation_accuracy': float(results[1]),
    'validation_precision': float(results[2]),
    'validation_recall': float(results[3]),
    'validation_loss': float(results[0]),
    'total_epochs': len(history.history['accuracy']),
    'best_val_accuracy': float(max(history.history['val_accuracy'])),
    'class_names': class_names,
    'total_images': len(X),
    'train_images': len(X_train),
    'val_images': len(X_val)
}

with open('results/training_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("✅ Results saved: results/training_results.json")

# Print summary
print("\n" + "=" * 70)
print("TRAINING COMPLETE! 🎉")
print("=" * 70)
print(f"Final Accuracy: {results[1]*100:.2f}%")
print(f"Best Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"Total Epochs: {len(history.history['accuracy'])}")
print("\nSaved files:")
print("  - saved_models/lung_cancer_final.h5")
print("  - saved_models/best_model.h5")
print("  - results/confusion_matrix.png")
print("  - results/training_history.png")
print("  - results/training_results.json")
print("=" * 70)
