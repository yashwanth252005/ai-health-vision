"""
============================================================================
SACNN TRAINING SCRIPT - HYBRID APPROACH
============================================================================
Train the Self-Attention CNN (SACNN) for feature extraction

ARCHITECTURE:
- 84 layers total
- 7 self-attention blocks
- 7.5M parameters
- Each block has 4 parallel attention paths
- Outputs 1406-dimensional features

HYBRID APPROACH:
- LOCAL: Batch size 16 (RTX 2050 4GB VRAM)
- KAGGLE: Batch size 64 (P100 16GB VRAM)
- Automatic environment detection

JOURNAL REFERENCE: Figure 5
============================================================================
"""

import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.self_attention import SelfAttentionCNN


def detect_environment():
    """
    HYBRID APPROACH: Detect execution environment
    
    Returns:
        dict: Environment configuration with batch size
    """
    is_kaggle = os.path.exists('/kaggle/input') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    if is_kaggle:
        print("🌐 KAGGLE Environment Detected")
        print("   GPU: P100 (16GB)")
        print("   Batch Size: 64")
        return {
            'name': 'kaggle',
            'batch_size': 64,
            'data_path': '/kaggle/input/lung-cancer-data',
            'output_path': '/kaggle/working'
        }
    else:
        print("💻 LOCAL Environment Detected")
        print("   GPU: RTX 2050 (4GB)")
        print("   Batch Size: 16")
        return {
            'name': 'local',
            'batch_size': 16,
            'data_path': 'data',
            'output_path': '.'
        }


def setup_gpu(env_config):
    """
    Configure GPU based on environment
    
    LOCAL: 3.5GB memory limit for 4GB VRAM
    KAGGLE: Full GPU utilization
    """
    print("\nGPU Configuration:")
    print("-" * 50)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit for local environment
            if env_config['name'] == 'local':
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=3584)]
                )
                print("Memory limit: 3.5GB (local)")
            
            # Enable mixed precision
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision: ENABLED")
            
            print(f"GPUs detected: {len(gpus)}")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    else:
        print("⚠️ No GPU - using CPU (slower)")
    
    print("-" * 50)


def load_dataset(env_config):
    """
    Load augmented lung cancer dataset
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, class_names)
    """
    print("\nLoading Dataset:")
    print("-" * 50)
    
    data_path = env_config['data_path']
    augmented_path = os.path.join(data_path, 'augmented')
    
    # Class names
    class_names = ['benign', 'malignant', 'normal']
    
    # Load images
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(augmented_path, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️ Directory not found: {class_dir}")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"{class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            # Load and preprocess image
            img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            
            images.append(img_array)
            labels.append(class_idx)
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\nTotal images loaded: {len(X)}")
    
    # Convert labels to one-hot encoding
    y_onehot = keras.utils.to_categorical(y, num_classes=len(class_names))
    
    # Split into train/validation (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print("-" * 50)
    
    return X_train, y_train, X_val, y_val, class_names


def plot_training_history(history, save_path):
    """
    Plot and save training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('SACNN - Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('SACNN - Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Training history saved: {save_path}")
    plt.close()


def train_sacnn(env_config=None, epochs=100):
    """
    Train SACNN with hybrid batch size support
    
    Args:
        env_config: Environment configuration (auto-detected if None)
        epochs: Number of training epochs
    
    Returns:
        tuple: (trained_model, history, env_config)
    """
    print("\n" + "=" * 70)
    print("SACNN TRAINING - HYBRID APPROACH")
    print("Self-Attention CNN with 7 Attention Blocks")
    print("=" * 70)
    
    # Detect environment if not provided
    if env_config is None:
        env_config = detect_environment()
    
    # Setup GPU
    setup_gpu(env_config)
    
    # Load dataset
    X_train, y_train, X_val, y_val, class_names = load_dataset(env_config)
    
    # Build SACNN model
    print("\nBuilding SACNN Model:")
    print("-" * 50)
    sacnn = SelfAttentionCNN()
    model = sacnn.build_model(include_top=True, num_classes=len(class_names))
    
    print(f"Total layers: {len(model.layers)}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Attention blocks: 7")
    print(f"Parallel paths per block: 4")
    print("-" * 50)
    
    # Compile model with journal hyperparameters
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00021),  # From config
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_dir = os.path.join(env_config['output_path'], 'saved_models')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'sacnn_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(env_config['output_path'], 'logs', 'sacnn'),
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\nTraining SACNN:")
    print("-" * 50)
    print(f"Environment: {env_config['name'].upper()}")
    print(f"Batch Size: {env_config['batch_size']}")
    print(f"Epochs: {epochs}")
    print(f"Self-Attention: ENABLED (Q/K/V transformations)")
    print("-" * 50)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=env_config['batch_size'],  # HYBRID: 16 local, 64 Kaggle
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'sacnn_final.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # Save training history plot
    plot_path = os.path.join(env_config['output_path'], 'results', 'sacnn_training.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot_training_history(history, plot_path)
    
    # Print final results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    final_accuracy = history.history['val_accuracy'][-1]
    best_accuracy = max(history.history['val_accuracy'])
    
    print(f"Final Validation Accuracy: {final_accuracy * 100:.2f}%")
    print(f"Best Validation Accuracy: {best_accuracy * 100:.2f}%")
    print(f"Total Epochs: {len(history.history['accuracy'])}")
    print("=" * 70 + "\n")
    
    return model, history, env_config


def extract_features(model, images):
    """
    Extract 1406-dimensional features from trained SACNN
    
    This removes the classification head and uses only the feature extractor
    
    Args:
        model: Trained SACNN model
        images: Input images
    
    Returns:
        numpy array: Extracted features (N x 1406)
    """
    print("\nExtracting SACNN features...")
    
    # Create feature extractor (remove classification layers)
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output  # Before final dense layers
    )
    
    # Extract features
    features = feature_extractor.predict(images, verbose=1)
    
    print(f"Feature shape: {features.shape}")
    print(f"Expected: (N, 1406)")
    
    return features


if __name__ == '__main__':
    """
    Run SACNN training
    
    USAGE:
    
    LOCAL:
        python training/train_sacnn.py
    
    KAGGLE:
        # Upload to Kaggle - auto-detects environment
        # Uses batch_size=64 automatically
    """
    
    # Train SACNN
    model, history, env_config = train_sacnn(epochs=100)
    
    print("✅ SACNN training complete!")
    print(f"Model saved in: {env_config['output_path']}/saved_models/")
    print("\nNext step: Feature Fusion and SSA Optimization")
    print("Run: python training/train_complete_pipeline.py")
