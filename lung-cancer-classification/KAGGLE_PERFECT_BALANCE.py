# ========================================================================
# 🎯 PERFECT BALANCE - Equal Treatment for All 3 Classes
# ========================================================================

import gc
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

# ========================================================================
# STEP 1: Reset
# ========================================================================
keras.backend.clear_session()
for var in ['model', 'history', 'X_train', 'X_val', 'y_train', 'y_val',
            'X_train_balanced', 'y_train_balanced', 'y_train_balanced_onehot']:
    try:
        exec(f"del {var}")
    except:
        pass
gc.collect()

print("=" * 70)
print("🔄 SESSION RESET")
print("=" * 70)

# ========================================================================
# STEP 2: Create Perfectly Balanced Dataset (300 per class)
# ========================================================================
print("\n📊 Creating PERFECTLY balanced dataset")
print("Strategy: EXACTLY 300 samples per class")

X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

y_train_labels = np.argmax(y_train, axis=1)

# Separate by class
X_benign = X_train[y_train_labels == 0]
X_malignant = X_train[y_train_labels == 1]
X_normal = X_train[y_train_labels == 2]

print(f"\nOriginal training counts:")
print(f"  Benign: {len(X_benign)}")
print(f"  Malignant: {len(X_malignant)}")
print(f"  Normal: {len(X_normal)}")

# Augmenter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmenter = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

def balance_class(X_class, target_count, class_idx, class_name):
    """Balance a class to exact target count"""
    current = len(X_class)
    
    print(f"\n  Balancing {class_name}: {current} → {target_count}")
    
    if current >= target_count:
        # Undersample
        indices = np.random.choice(current, size=target_count, replace=False)
        return X_class[indices], np.full(target_count, class_idx)
    
    # Oversample
    copies = [X_class]
    labels = [np.full(current, class_idx)]
    
    remaining = target_count - current
    print(f"    Need {remaining} synthetic samples...")
    
    batch_num = 0
    while remaining > 0:
        batch_size = min(remaining, current)
        indices = np.random.choice(current, size=batch_size, replace=True)
        batch = X_class[indices]
        
        # Augment
        augmented = []
        for img in batch:
            img_exp = np.expand_dims(img, 0)
            aug_img = next(augmenter.flow(img_exp, batch_size=1))[0]
            augmented.append(aug_img)
        
        copies.append(np.array(augmented))
        labels.append(np.full(len(augmented), class_idx))
        remaining -= batch_size
        batch_num += 1
        
        if batch_num % 2 == 0:
            print(f"    Progress: {target_count - remaining}/{target_count}")
    
    X_out = np.concatenate(copies, axis=0)[:target_count]
    y_out = np.concatenate(labels, axis=0)[:target_count]
    print(f"    ✅ Done: {len(X_out)} samples")
    return X_out, y_out

# Balance ALL classes to EXACTLY 300 each
TARGET = 300

print(f"\n🎯 Balancing ALL classes to EXACTLY {TARGET} samples:")
print("=" * 70)

X_benign_bal, y_benign_bal = balance_class(X_benign, TARGET, 0, "Benign")
X_malignant_bal, y_malignant_bal = balance_class(X_malignant, TARGET, 1, "Malignant")
X_normal_bal, y_normal_bal = balance_class(X_normal, TARGET, 2, "Normal")

# Combine
X_train_balanced = np.concatenate([
    X_benign_bal,
    X_malignant_bal,
    X_normal_bal
], axis=0)

y_train_balanced = np.concatenate([
    y_benign_bal,
    y_malignant_bal,
    y_normal_bal
], axis=0)

# Shuffle
shuffle_idx = np.random.permutation(len(X_train_balanced))
X_train_balanced = X_train_balanced[shuffle_idx]
y_train_balanced = y_train_balanced[shuffle_idx]

# One-hot
y_train_balanced_onehot = keras.utils.to_categorical(y_train_balanced, num_classes=3)

print(f"\n" + "=" * 70)
print("✅ PERFECTLY BALANCED DATASET CREATED!")
print("=" * 70)
print(f"Total: {len(X_train_balanced)} images")
print(f"\nFinal distribution:")
for i, name in enumerate(class_names):
    count = np.sum(y_train_balanced == i)
    pct = (count / len(y_train_balanced)) * 100
    print(f"  {name}: {count} ({pct:.2f}%)")

# Verify perfect balance
unique_counts = [np.sum(y_train_balanced == i) for i in range(3)]
if len(set(unique_counts)) == 1:
    print(f"\n🎯 PERFECT BALANCE CONFIRMED: Each class has {unique_counts[0]} samples!")
else:
    print(f"\n⚠️ WARNING: Counts not equal: {unique_counts}")

# ========================================================================
# STEP 3: Build Model
# ========================================================================
print("\n" + "=" * 70)
print("🏗️ Building Model")
print("=" * 70)

from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    
    layers.Conv2D(256, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print(f"✅ Model: {model.count_params():,} parameters")

# ========================================================================
# STEP 4: Train on PERFECTLY Balanced Data
# ========================================================================
print("\n" + "=" * 70)
print("🚀 TRAINING ON PERFECTLY BALANCED DATA")
print("=" * 70)
print(f"Each class: EXACTLY {TARGET} samples")
print("No class weights - pure equal treatment!")
print("Longer patience to learn all 3 classes")
print("=" * 70)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,  # More patience
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=8,
        mode='max',
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'saved_models/best_model_perfect.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

print(f"\nTraining configuration:")
print(f"  Training samples: {len(X_train_balanced)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Batch size: 64")
print(f"  Max epochs: 50")
print()

history = model.fit(
    X_train_balanced, y_train_balanced_onehot,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print("\n🎯 NOW RUN CELL 9 FOR EVALUATION!")
print("\nExpected results:")
print("  ✅ Benign Detection: 30-50%")
print("  ✅ Malignant Detection: 60-80%")
print("  ✅ Normal Detection: 40-60%")
print("  🎯 ALL 3 classes should be detected!")
print("=" * 70)
