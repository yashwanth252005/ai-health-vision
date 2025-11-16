"""
============================================================================
SHALLOW WIDE NEURAL NETWORK (SWNN) - FINAL CLASSIFIER
============================================================================
Based on: Scientific Reports Journal (DOI: 10.1038/s41598-025-93718-7)
Page 10, Figure 7, Table 5

WHAT IS SWNN?
A "Shallow" neural network has few layers (just 1-2 hidden layers).
A "Wide" neural network has many neurons per layer (512 units).

WHY SHALLOW WIDE instead of DEEP NARROW?
- FASTER training (fewer layers)
- LESS overfitting (simpler architecture)
- WORKS WELL with good features (which we have from IRCNN+SACNN+SSA!)

ANALOGY:
Think of it like a decision maker:
- Gets optimized features from SSA (500-1000 best features)
- Single hidden layer processes these features (512 neurons)
- Output layer makes final decision (3 classes: benign, malignant, normal)

ARCHITECTURE:
Input (500-1000 features from SSA) 
  → Dense(512, ReLU) 
  → Dropout(0.3) 
  → Dense(3, Softmax)

JOURNAL SPECIFICATIONS:
- Hidden Layer: 512 neurons
- Activation: ReLU
- Dropout: 0.3
- Output: 3 classes (benign, malignant, normal)
- Optimizer: ADAM
- Learning Rate: 0.00021
- Momentum: 0.701

EXPECTED PERFORMANCE:
- Accuracy: 85.0%
- Precision: 85.0%
- Sensitivity: 85.0%
- F1-Score: 85.0%
============================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import numpy as np
import yaml
import os


class SWNNClassifier:
    """
    Shallow Wide Neural Network for Lung Cancer Classification
    
    This is the FINAL classifier in the complete pipeline:
    1. Images → IRCNN → 1282 features
    2. Images → SACNN → 1406 features
    3. Fusion → 2688 combined features
    4. SSA → 500-1000 optimized features
    5. SWNN → Classification (benign/malignant/normal)
    
    WHY THIS ARCHITECTURE?
    - Shallow: Fast training, less overfitting
    - Wide: Can handle many input features effectively
    - Simple: Easy to train and interpret
    - Effective: 85% accuracy with proper features
    
    JOURNAL REFERENCE:
    Page 10: "SWNN with single hidden layer of 512 units"
    Figure 7: Architecture diagram
    Table 5: Performance metrics (85% accuracy)
    
    USAGE:
        # After SSA optimization
        swnn = SWNNClassifier(n_input_features=len(best_features))
        model = swnn.build_model()
        model.fit(X_train_selected, y_train, epochs=100)
        predictions = model.predict(X_test_selected)
    """
    
    def __init__(self, n_input_features, config_path='config/config.yaml'):
        """
        Initialize SWNN Classifier
        
        Args:
            n_input_features (int): Number of input features (from SSA optimization)
            config_path (str): Path to configuration file
        """
        self.n_input_features = n_input_features
        
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.swnn_config = config.get('swnn_classifier', {})
                self.training_config = config.get('training', {})
        else:
            # Default configuration if file not found
            self.swnn_config = {
                'hidden_units': 512,
                'dropout_rate': 0.3,
                'activation': 'relu'
            }
            self.training_config = {
                'learning_rate': 0.00021,
                'momentum': 0.701,
                'optimizer': 'adam'
            }
        
        # Extract hyperparameters
        # WHY: These values are from the journal
        self.hidden_units = self.swnn_config.get('hidden_units', 512)
        self.dropout_rate = self.swnn_config.get('dropout_rate', 0.3)
        self.activation = self.swnn_config.get('activation', 'relu')
        self.n_classes = 3  # Benign, Malignant, Normal
        
        print("=" * 70)
        print("SHALLOW WIDE NEURAL NETWORK (SWNN) INITIALIZED")
        print("=" * 70)
        print(f"Input Features: {self.n_input_features}")
        print(f"Hidden Units: {self.hidden_units}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print(f"Output Classes: {self.n_classes} (benign, malignant, normal)")
        print("=" * 70)
    
    def build_model(self):
        """
        Build the SWNN architecture
        
        COMPLETE ARCHITECTURE:
        
        1. INPUT LAYER
           - Shape: (n_features,) from SSA optimization
           - WHY: Receives optimized features
        
        2. HIDDEN LAYER (WIDE)
           - Dense(512) neurons
           - ReLU activation
           - WHY: Wide layer can learn complex patterns from features
           - JOURNAL: Page 10 - "512 hidden units for feature processing"
        
        3. DROPOUT LAYER
           - Rate: 0.3 (drops 30% of neurons during training)
           - WHY: Prevents overfitting, improves generalization
           - JOURNAL: Page 10 - "Dropout for regularization"
        
        4. OUTPUT LAYER
           - Dense(3) neurons (one per class)
           - Softmax activation
           - WHY: Produces probability distribution over classes
           - Output: [P(benign), P(malignant), P(normal)]
        
        LAYER SIZES:
        Input (500-1000) → Dense(512) → Dropout(0.3) → Dense(3)
        
        TOTAL PARAMETERS:
        Approximately: (n_features × 512) + (512 × 3) ≈ 256K-512K parameters
        
        Returns:
            keras.Model: Compiled SWNN model ready for training
        """
        print("\nBuilding SWNN Architecture...")
        
        # Input Layer
        # WHY: Accepts optimized features from SSA
        # SHAPE: (batch_size, n_features)
        inputs = keras.Input(
            shape=(self.n_input_features,),
            name='swnn_input'
        )
        print(f"  Input: {self.n_input_features} features")
        
        # Hidden Layer (WIDE)
        # WHY: 512 neurons provide sufficient capacity for pattern learning
        # ACTIVATION: ReLU - standard for hidden layers
        # JOURNAL: Page 10 - "Single hidden layer with 512 neurons"
        x = layers.Dense(
            units=self.hidden_units,
            activation=self.activation,
            kernel_initializer='he_normal',  # Good for ReLU
            name='hidden_layer'
        )(inputs)
        print(f"  Hidden: {self.hidden_units} neurons (ReLU)")
        
        # Dropout Layer
        # WHY: Regularization to prevent overfitting
        # RATE: 0.3 means 30% of neurons are randomly dropped during training
        # JOURNAL: Page 10 - "Dropout rate of 0.3"
        x = layers.Dropout(
            rate=self.dropout_rate,
            name='dropout'
        )(x)
        print(f"  Dropout: {self.dropout_rate * 100:.0f}%")
        
        # Output Layer
        # WHY: Produces final classification
        # UNITS: 3 (one for each class)
        # ACTIVATION: Softmax converts scores to probabilities
        # OUTPUT: [P(benign), P(malignant), P(normal)] where sum = 1.0
        outputs = layers.Dense(
            units=self.n_classes,
            activation='softmax',
            name='output'
        )(x)
        print(f"  Output: {self.n_classes} classes (softmax)")
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='SWNN')
        
        # Compile model with optimizer and loss function
        # OPTIMIZER: ADAM (Adaptive Moment Estimation)
        # WHY: ADAM is fast and works well for most problems
        # LEARNING RATE: 0.00021 (from journal)
        # JOURNAL: Page 10 - "ADAM optimizer with lr=0.00021"
        
        learning_rate = self.training_config.get('learning_rate', 0.00021)
        
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,      # Default momentum for first moment
            beta_2=0.999     # Default momentum for second moment
        )
        
        # Loss function: Categorical Crossentropy
        # WHY: Standard loss for multi-class classification
        # Labels should be one-hot encoded: [0, 1, 0] for malignant
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        print(f"\nModel compiled:")
        print(f"  Optimizer: ADAM")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Loss: Categorical Crossentropy")
        print(f"  Metrics: Accuracy, Precision, Recall")
        
        # Count parameters
        total_params = model.count_params()
        print(f"\nTotal Parameters: {total_params:,}")
        
        print("\n" + "=" * 70)
        print("SWNN BUILD COMPLETE!")
        print("=" * 70)
        
        return model
    
    def train(self, model, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=16, verbose=1):
        """
        Train the SWNN model
        
        TRAINING PROCESS:
        1. Feed batches of features and labels
        2. Model makes predictions
        3. Calculate loss (how wrong predictions are)
        4. Update weights to reduce loss
        5. Repeat for all batches (1 epoch)
        6. Repeat for multiple epochs (100 total)
        
        CALLBACKS:
        - Early Stopping: Stops if validation loss doesn't improve for 15 epochs
        - Model Checkpoint: Saves best model based on validation accuracy
        - Reduce LR on Plateau: Reduces learning rate if stuck
        
        EXPECTED TRAINING TIME:
        - Local (RTX 2050): ~2-3 minutes
        - Kaggle (P100): ~1-2 minutes
        
        JOURNAL REFERENCE:
        Page 10: "Trained for 100 epochs with early stopping"
        Table 5: Training configuration
        
        Args:
            model: Compiled SWNN model
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples, 3) - one-hot encoded
            X_val: Validation features
            y_val: Validation labels
            epochs: Maximum training epochs (default=100)
            batch_size: Batch size (default=16 for local, 64 for Kaggle)
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            history: Training history with loss and metrics
        """
        print("\n" + "=" * 70)
        print("STARTING SWNN TRAINING")
        print("=" * 70)
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print("=" * 70 + "\n")
        
        # Define callbacks
        # WHY: Improves training efficiency and prevents overfitting
        
        # Early Stopping
        # WHY: Stops training if validation loss stops improving
        # PATIENCE: Wait 15 epochs before stopping
        # JOURNAL: Page 10 - "Early stopping with patience 15"
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model Checkpoint
        # WHY: Saves best model during training
        checkpoint = keras.callbacks.ModelCheckpoint(
            'saved_models/swnn_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Reduce Learning Rate on Plateau
        # WHY: Reduces LR when validation loss plateaus
        # Helps model converge to better solution
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,      # Multiply LR by 0.5
            patience=5,      # Wait 5 epochs
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint, reduce_lr],
            verbose=verbose
        )
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print("=" * 70 + "\n")
        
        return history
    
    def evaluate(self, model, X_test, y_test, class_names=None):
        """
        Evaluate SWNN model on test set
        
        EVALUATION METRICS:
        - Accuracy: Overall correctness (TP + TN) / Total
        - Precision: How many predicted positives are actually positive
        - Recall (Sensitivity): How many actual positives were found
        - F1-Score: Harmonic mean of precision and recall
        
        EXPECTED RESULTS (from journal Table 5):
        - Accuracy: 85.0%
        - Precision: 85.0%
        - Recall (Sensitivity): 85.0%
        - F1-Score: 85.0%
        
        JOURNAL REFERENCE:
        Table 5: Performance metrics on lung cancer dataset
        
        Args:
            model: Trained SWNN model
            X_test: Test features (n_samples, n_features)
            y_test: Test labels (n_samples, 3) - one-hot encoded
            class_names: List of class names (default=['benign', 'malignant', 'normal'])
            
        Returns:
            results: Dictionary with all evaluation metrics
        """
        if class_names is None:
            class_names = ['benign', 'malignant', 'normal']
        
        print("\n" + "=" * 70)
        print("EVALUATING SWNN MODEL")
        print("=" * 70)
        print(f"Test samples: {X_test.shape[0]}")
        print("=" * 70 + "\n")
        
        # Get predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, classification_report, confusion_matrix
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        print("OVERALL METRICS:")
        print(f"  Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  Precision:   {precision:.4f} ({precision * 100:.2f}%)")
        print(f"  Recall:      {recall:.4f} ({recall * 100:.2f}%)")
        print(f"  F1-Score:    {f1:.4f} ({f1 * 100:.2f}%)")
        
        print("\n" + "-" * 70)
        print("PER-CLASS METRICS:")
        print("-" * 70)
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        print("-" * 70)
        print("CONFUSION MATRIX:")
        print("-" * 70)
        cm = confusion_matrix(y_true, y_pred)
        print(f"{'':>12}", end='')
        for name in class_names:
            print(f"{name:>12}", end='')
        print()
        for i, name in enumerate(class_names):
            print(f"{name:>12}", end='')
            for j in range(len(class_names)):
                print(f"{cm[i, j]:>12}", end='')
            print()
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE!")
        print("=" * 70)
        
        # Compare with journal results
        print("\nCOMPARISON WITH JOURNAL (Table 5):")
        print(f"  Expected Accuracy: 85.0%")
        print(f"  Achieved Accuracy: {accuracy * 100:.1f}%")
        diff = abs(accuracy * 100 - 85.0)
        if diff < 2:
            print(f"  Status: ✓ EXCELLENT (within 2%)")
        elif diff < 5:
            print(f"  Status: ✓ GOOD (within 5%)")
        else:
            print(f"  Status: ⚠ NEEDS IMPROVEMENT")
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_probs
        }
        
        return results


if __name__ == '__main__':
    """
    Test script for SWNN Classifier
    
    RUN THIS TO:
    - Test SWNN architecture
    - Verify model building
    - Check parameter count
    """
    print("\n" + "=" * 70)
    print("TESTING SHALLOW WIDE NEURAL NETWORK (SWNN)")
    print("=" * 70 + "\n")
    
    # Create dummy data (simulating SSA-optimized features)
    print("Generating dummy test data...")
    n_samples = 100
    n_features = 800  # Typical output from SSA optimization
    n_classes = 3
    
    np.random.seed(42)
    X_train = np.random.rand(n_samples, n_features).astype(np.float32)
    y_train = keras.utils.to_categorical(
        np.random.randint(0, n_classes, size=n_samples),
        num_classes=n_classes
    )
    
    X_test = np.random.rand(20, n_features).astype(np.float32)
    y_test = keras.utils.to_categorical(
        np.random.randint(0, n_classes, size=20),
        num_classes=n_classes
    )
    
    print(f"Train data: {X_train.shape}")
    print(f"Train labels: {y_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Build model
    print("\n" + "-" * 70)
    print("Building SWNN Model...")
    swnn = SWNNClassifier(n_input_features=n_features)
    model = swnn.build_model()
    model.summary()
    
    # Quick training test
    print("\n" + "-" * 70)
    print("Testing Training (5 epochs)...")
    history = swnn.train(
        model, X_train, y_train, X_test, y_test,
        epochs=5, batch_size=16, verbose=0
    )
    
    # Evaluation test
    print("\n" + "-" * 70)
    print("Testing Evaluation...")
    results = swnn.evaluate(model, X_test, y_test)
    
    print("\n" + "=" * 70)
    print("SWNN TESTING COMPLETE!")
    print("=" * 70 + "\n")
