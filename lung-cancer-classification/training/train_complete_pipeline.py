"""
============================================================================
COMPLETE TRAINING PIPELINE - HYBRID APPROACH
============================================================================
This script trains the ENTIRE lung cancer classification pipeline:
1. Data Augmentation
2. Train IRCNN (feature extractor)
3. Train SACNN (feature extractor)
4. Feature Fusion
5. SSA Optimization
6. Train SWNN (final classifier)

HYBRID APPROACH:
- LOCAL: Batch size 16, mixed precision, optimized for RTX 2050 4GB
- KAGGLE: Batch size 64, full GPU utilization, P100 GPU

The script AUTOMATICALLY DETECTS which environment you're in and adjusts!
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
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from utils.data_augmentation import LungCancerDataAugmentation
from models.inverted_residual import InvertedResidualCNN
from models.self_attention import SelfAttentionCNN
from utils.feature_fusion import SerialFeatureFusion
from utils.ssa_optimization import SalpSwarmOptimizer
from models.swnn_classifier import SWNNClassifier


def detect_environment():
    """
    Detect if running locally or on Kaggle
    
    HYBRID APPROACH:
    - Checks for Kaggle-specific environment variables
    - Adjusts batch size and GPU settings accordingly
    
    Returns:
        dict: Environment configuration
    """
    is_kaggle = os.path.exists('/kaggle/input') or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    if is_kaggle:
        print("🔍 Environment: KAGGLE")
        print("  GPU: P100 (16GB VRAM)")
        print("  Batch Size: 64 (high performance)")
        return {
            'name': 'kaggle',
            'batch_size': 64,
            'data_path': '/kaggle/input/lung-cancer-data/data',
            'output_path': '/kaggle/working',
            'mixed_precision': True,
            'gpu_memory_limit': None  # Use full GPU
        }
    else:
        print("🔍 Environment: LOCAL")
        print("  GPU: RTX 2050 (4GB VRAM)")
        print("  Batch Size: 16 (memory optimized)")
        return {
            'name': 'local',
            'batch_size': 16,
            'data_path': 'data',
            'output_path': '.',
            'mixed_precision': True,
            'gpu_memory_limit': 3584  # 3.5GB limit for 4GB VRAM
        }


def setup_gpu(env_config):
    """
    Configure GPU based on environment
    
    LOCAL: Limited memory growth, mixed precision
    KAGGLE: Full GPU utilization
    """
    print("\n" + "=" * 70)
    print("GPU CONFIGURATION")
    print("=" * 70)
    
    # Check available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth (prevents TensorFlow from allocating all GPU memory)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit for local environment
            if env_config['name'] == 'local' and env_config['gpu_memory_limit']:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=env_config['gpu_memory_limit']
                    )]
                )
                print(f"GPU memory limit: {env_config['gpu_memory_limit']}MB")
            
            # Enable mixed precision for faster training
            if env_config['mixed_precision']:
                policy = keras.mixed_precision.Policy('mixed_float16')
                keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision: ENABLED (faster training)")
            
            print(f"GPUs available: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu.name}")
                
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠️ No GPU detected - using CPU (training will be slower)")
    
    print("=" * 70 + "\n")


def load_and_prepare_data(env_config):
    """
    Load and prepare dataset
    
    Steps:
    1. Check if augmented data exists
    2. If not, run augmentation
    3. Split into train/test (50-50)
    4. Create TensorFlow datasets
    """
    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)
    
    data_path = env_config['data_path']
    augmented_path = os.path.join(data_path, 'augmented')
    
    # Check if augmentation is needed
    if not os.path.exists(augmented_path) or len(os.listdir(augmented_path)) < 3:
        print("Running data augmentation...")
        augmenter = LungCancerDataAugmentation()
        augmenter.augment_dataset(
            input_dir=os.path.join(data_path, 'raw'),
            output_dir=augmented_path,
            multiplier=5  # Creates ~4000 images from ~200 originals
        )
    else:
        print(f"✓ Augmented data found: {augmented_path}")
    
    # Load images and labels
    print("\nLoading dataset...")
    image_paths = []
    labels = []
    class_names = ['benign', 'malignant', 'normal']
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(augmented_path, class_name)
        if os.path.exists(class_dir):
            class_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
    
    print(f"Total images: {len(image_paths)}")
    print(f"Class distribution:")
    for i, name in enumerate(class_names):
        count = labels.count(i)
        print(f"  {name}: {count} images")
    
    # Convert to numpy arrays
    labels = np.array(labels)
    labels_onehot = keras.utils.to_categorical(labels, num_classes=len(class_names))
    
    # Split 50-50 (as per journal)
    from sklearn.model_selection import train_test_split
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        image_paths, labels_onehot,
        test_size=0.5,
        stratify=labels,
        random_state=42
    )
    
    print(f"\nTrain set: {len(X_train_paths)} images")
    print(f"Test set: {len(X_test_paths)} images")
    
    print("=" * 70 + "\n")
    
    return {
        'train_paths': X_train_paths,
        'test_paths': X_test_paths,
        'y_train': y_train,
        'y_test': y_test,
        'class_names': class_names
    }


def load_images(image_paths, target_size=(224, 224)):
    """
    Load and preprocess images
    """
    images = []
    for path in image_paths:
        img = keras.preprocessing.image.load_img(path, target_size=target_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        images.append(img_array)
    return np.array(images)


def train_complete_pipeline(env_config=None):
    """
    Train the complete lung cancer classification pipeline
    
    HYBRID APPROACH:
    - Automatically detects environment (local vs Kaggle)
    - Adjusts batch sizes and GPU settings
    - Saves checkpoints for both environments
    """
    print("\n" + "=" * 70)
    print("LUNG CANCER CLASSIFICATION - COMPLETE TRAINING PIPELINE")
    print("HYBRID APPROACH: Works on LOCAL (RTX 2050) and KAGGLE (P100)")
    print("=" * 70 + "\n")
    
    # Detect environment if not provided
    if env_config is None:
        env_config = detect_environment()
    
    # Setup GPU
    setup_gpu(env_config)
    
    # Load configuration
    config_path = 'config/config.yaml' if env_config['name'] == 'local' else '../input/config/config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("⚠️ Config file not found, using defaults")
        config = {}
    
    # Prepare data
    data = load_and_prepare_data(env_config)
    
    # Create output directory
    os.makedirs(os.path.join(env_config['output_path'], 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(env_config['output_path'], 'results'), exist_ok=True)
    
    # ========================================================================
    # STAGE 1: Train IRCNN (Feature Extractor)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Training IRCNN (94 Layers)")
    print("=" * 70)
    
    print("Loading training images...")
    X_train = load_images(data['train_paths'][:100])  # Start with subset for testing
    X_test = load_images(data['test_paths'][:50])
    
    print("Building IRCNN...")
    ircnn = InvertedResidualCNN()
    ircnn_model = ircnn.build_model(include_top=True)  # With classification head
    
    print(f"Training with batch size: {env_config['batch_size']}")
    history_ircnn = ircnn_model.fit(
        X_train, data['y_train'][:100],
        validation_data=(X_test, data['y_test'][:50]),
        epochs=20,  # Reduced for testing
        batch_size=env_config['batch_size'],
        verbose=1
    )
    
    # Save model
    ircnn_path = os.path.join(env_config['output_path'], 'saved_models', 'ircnn_model.h5')
    ircnn_model.save(ircnn_path)
    print(f"✓ IRCNN saved: {ircnn_path}")
    
    # ========================================================================
    # STAGE 2: Train SACNN (Feature Extractor)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Training SACNN (84 Layers)")
    print("=" * 70)
    
    print("Building SACNN...")
    sacnn = SelfAttentionCNN()
    sacnn_model = sacnn.build_model(include_top=True)
    
    print(f"Training with batch size: {env_config['batch_size']}")
    history_sacnn = sacnn_model.fit(
        X_train, data['y_train'][:100],
        validation_data=(X_test, data['y_test'][:50]),
        epochs=20,
        batch_size=env_config['batch_size'],
        verbose=1
    )
    
    # Save model
    sacnn_path = os.path.join(env_config['output_path'], 'saved_models', 'sacnn_model.h5')
    sacnn_model.save(sacnn_path)
    print(f"✓ SACNN saved: {sacnn_path}")
    
    # ========================================================================
    # STAGE 3: Feature Extraction & Fusion
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Feature Fusion")
    print("=" * 70)
    
    # Rebuild models for feature extraction (without classification head)
    ircnn_features = InvertedResidualCNN().build_model(include_top=False)
    sacnn_features = SelfAttentionCNN().build_model(include_top=False)
    
    # Load weights from trained models
    ircnn_features.set_weights(ircnn_model.get_weights()[:-4])  # Exclude classification layers
    sacnn_features.set_weights(sacnn_model.get_weights()[:-4])
    
    # Create fusion module
    fusion = SerialFeatureFusion(ircnn_features, sacnn_features)
    
    # Extract and fuse features
    print("Extracting features from training set...")
    X_train_fused = fusion.fuse_features(X_train)
    
    print("Extracting features from test set...")
    X_test_fused = fusion.fuse_features(X_test)
    
    # ========================================================================
    # STAGE 4: SSA Optimization
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 4: SSA Feature Optimization")
    print("=" * 70)
    
    ssa = SalpSwarmOptimizer(
        n_features=X_train_fused.shape[1],
        n_salps=10,  # Reduced for testing
        max_iterations=20  # Reduced for testing
    )
    
    best_features, ssa_history = ssa.optimize(
        X_train_fused,
        np.argmax(data['y_train'][:100], axis=1),
        verbose=True
    )
    
    # Select optimized features
    X_train_optimized = X_train_fused[:, best_features]
    X_test_optimized = X_test_fused[:, best_features]
    
    # Save feature indices
    feature_indices_path = os.path.join(env_config['output_path'], 'saved_models', 'ssa_features.npy')
    np.save(feature_indices_path, best_features)
    print(f"✓ Feature indices saved: {feature_indices_path}")
    
    # ========================================================================
    # STAGE 5: Train SWNN (Final Classifier)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 5: Training SWNN Classifier")
    print("=" * 70)
    
    swnn = SWNNClassifier(n_input_features=X_train_optimized.shape[1])
    swnn_model = swnn.build_model()
    
    history_swnn = swnn.train(
        swnn_model,
        X_train_optimized, data['y_train'][:100],
        X_test_optimized, data['y_test'][:50],
        epochs=50,
        batch_size=env_config['batch_size'],
        verbose=1
    )
    
    # Save SWNN model
    swnn_path = os.path.join(env_config['output_path'], 'saved_models', 'swnn_model.h5')
    swnn_model.save(swnn_path)
    print(f"✓ SWNN saved: {swnn_path}")
    
    # ========================================================================
    # STAGE 6: Final Evaluation
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 6: Final Evaluation")
    print("=" * 70)
    
    results = swnn.evaluate(
        swnn_model,
        X_test_optimized,
        data['y_test'][:50],
        class_names=data['class_names']
    )
    
    # Save results
    results_path = os.path.join(env_config['output_path'], 'results', 'final_results.json')
    results_to_save = {
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'environment': env_config['name'],
        'batch_size': env_config['batch_size'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"✓ Results saved: {results_path}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Environment: {env_config['name'].upper()}")
    print(f"Batch Size: {env_config['batch_size']}")
    print(f"Final Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Target Accuracy: 95.0%")
    print("\nModels saved in: saved_models/")
    print("Results saved in: results/")
    print("=" * 70 + "\n")
    
    return {
        'ircnn_model': ircnn_model,
        'sacnn_model': sacnn_model,
        'swnn_model': swnn_model,
        'results': results,
        'env_config': env_config
    }


if __name__ == '__main__':
    """
    Run complete training pipeline
    
    USAGE:
    
    LOCAL:
        python training/train_complete_pipeline.py
    
    KAGGLE:
        # Upload this script to Kaggle
        # It will automatically detect Kaggle environment
        # and use batch_size=64 with P100 GPU
    """
    # Train the complete pipeline
    pipeline_results = train_complete_pipeline()
    
    print("\n✅ All done! Your lung cancer classifier is ready!")
    print(f"Final accuracy: {pipeline_results['results']['accuracy'] * 100:.2f}%")
