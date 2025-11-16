"""
============================================================================
MODEL EVALUATION SCRIPT
============================================================================
Comprehensive evaluation of trained lung cancer classification models

METRICS:
- Accuracy
- Precision (per class and weighted)
- Recall (per class and weighted)
- F1-Score (per class and weighted)
- Confusion Matrix
- Classification Report

COMPARISON:
- Compares results with journal paper (target: 85% accuracy)
============================================================================
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_test_data(data_path='data/augmented', test_split=0.5):
    """
    Load test dataset
    
    Args:
        data_path: Path to augmented data
        test_split: Fraction of data for testing (0.5 = 50%)
    
    Returns:
        tuple: (X_test, y_test, class_names)
    """
    print("Loading test data...")
    
    class_names = ['benign', 'malignant', 'normal']
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_path, class_name)
        if not os.path.exists(class_dir):
            continue
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            
            images.append(img_array)
            labels.append(class_idx)
    
    X = np.array(images)
    y = np.array(labels)
    
    # Split (use second half as test set)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=test_split,
        stratify=y,
        random_state=42
    )
    
    print(f"Test set: {len(X_test)} images")
    
    return X_test, y_test, class_names


def evaluate_model(model, X_test, y_test, class_names, model_name='Model'):
    """
    Evaluate a trained model
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (integer labels, NOT one-hot)
        class_names: List of class names
        model_name: Name of model for reporting
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating {model_name}")
    print(f"{'=' * 70}")
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\n" + "-" * 70)
    print("OVERALL METRICS")
    print("-" * 70)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name.upper()}:")
        print(f"  Precision: {precision_per_class[i] * 100:.2f}%")
        print(f"  Recall:    {recall_per_class[i] * 100:.2f}%")
        print(f"  F1-Score:  {f1_per_class[i] * 100:.2f}%")
    
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    print(cm)
    
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("=" * 70 + "\n")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'true_labels': y_test.tolist()
    }


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()


def plot_per_class_metrics(metrics, class_names, save_path):
    """
    Plot per-class precision, recall, and F1-score
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x = np.arange(len(class_names))
    width = 0.6
    
    # Precision
    axes[0].bar(x, metrics['precision_per_class'], width, color='skyblue')
    axes[0].set_title('Precision per Class', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names)
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Score')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[1].bar(x, metrics['recall_per_class'], width, color='lightcoral')
    axes[1].set_title('Recall per Class', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Score')
    axes[1].grid(axis='y', alpha=0.3)
    
    # F1-Score
    axes[2].bar(x, metrics['f1_per_class'], width, color='lightgreen')
    axes[2].set_title('F1-Score per Class', fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names)
    axes[2].set_ylim([0, 1])
    axes[2].set_ylabel('Score')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics saved: {save_path}")
    plt.close()


def compare_with_journal(metrics, journal_accuracy=0.95):
    """
    Compare results with journal paper
    
    Args:
        metrics: Evaluation metrics
        journal_accuracy: Target accuracy from journal (default 85%)
    """
    print("\n" + "=" * 70)
    print("COMPARISON WITH JOURNAL PAPER")
    print("=" * 70)
    
    our_accuracy = metrics['accuracy']
    difference = (our_accuracy - journal_accuracy) * 100
    
    print(f"Our Accuracy:      {our_accuracy * 100:.2f}%")
    print(f"Journal Accuracy:  {journal_accuracy * 100:.2f}%")
    print(f"Difference:        {difference:+.2f}%")
    
    if our_accuracy >= journal_accuracy:
        print("\n✅ SUCCESS: Our model matches or exceeds journal performance!")
    elif our_accuracy >= journal_accuracy - 0.05:
        print("\n⚠️ CLOSE: Our model is within 5% of journal performance")
    else:
        print("\n❌ NEEDS IMPROVEMENT: Consider hyperparameter tuning")
    
    print("=" * 70 + "\n")


def evaluate_complete_pipeline(model_path, data_path='data/augmented', output_dir='results'):
    """
    Evaluate the complete trained pipeline
    
    Args:
        model_path: Path to saved model
        data_path: Path to test data
        output_dir: Directory to save results
    """
    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE EVALUATION")
    print("=" * 70 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load test data
    X_test, y_test, class_names = load_test_data(data_path)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, class_names, 
                            model_name='Lung Cancer Classifier')
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        class_names,
        cm_path,
        title='Lung Cancer Classification - Confusion Matrix'
    )
    
    # Plot per-class metrics
    metrics_path = os.path.join(output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(metrics, class_names, metrics_path)
    
    # Compare with journal
    compare_with_journal(metrics)
    
    # Save results to JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'metrics': metrics,
        'class_names': class_names,
        'test_size': len(X_test),
        'journal_comparison': {
            'our_accuracy': metrics['accuracy'],
            'journal_accuracy': 0.95,
            'difference': metrics['accuracy'] - 0.95
        }
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved: {results_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"All results saved in: {output_dir}/")
    print("=" * 70 + "\n")
    
    return metrics


if __name__ == '__main__':
    """
    Run evaluation
    
    USAGE:
        python evaluation/evaluate_model.py
    
    Or specify model path:
        python evaluation/evaluate_model.py --model saved_models/swnn_model.h5
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, default='saved_models/swnn_model.h5',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='data/augmented',
                       help='Path to test data')
    parser.add_argument('--output', type=str, default='results/evaluation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_complete_pipeline(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output
    )
    
    print(f"✅ Final Accuracy: {metrics['accuracy'] * 100:.2f}%")
