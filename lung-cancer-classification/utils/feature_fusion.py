"""
============================================================================
FEATURE FUSION MODULE - PEARSON CORRELATION COEFFICIENT
============================================================================
Based on: Scientific Reports Journal (DOI: 10.1038/s41598-025-93718-7)
Equation 4, Page 8, Figure 6

WHAT IS FEATURE FUSION?
Feature fusion combines features from BOTH CNN models (IRCNN + SACNN) into
a single, more powerful feature representation. Think of it like getting
two expert opinions and combining them for better diagnosis.

WHY PEARSON CORRELATION?
The journal uses Pearson Correlation Coefficient to measure how features
from both models relate to each other. This ensures we keep features that
provide complementary (different but useful) information.

JOURNAL EQUATION 4:
r(U,V) = Σ(Ui - Ū)(Vi - V̄) / √[Σ(Ui - Ū)² × Σ(Vi - V̄)²]

WHERE:
- U: Features from IRCNN (1282 dimensions)
- V: Features from SACNN (1406 dimensions)
- Ū: Mean of U features
- V̄: Mean of V features
- r: Correlation coefficient (-1 to 1)

FUSION STRATEGY:
1. Extract features from both CNNs
2. Calculate correlation between feature pairs
3. Select strongly correlated features (high absolute r value)
4. Concatenate selected features → Final 2688-dimensional vector

PURPOSE:
Combines complementary information from both architectures:
- IRCNN: Efficient, lightweight, local patterns
- SACNN: Self-attention, global context, long-range dependencies
============================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml
import os


class SerialFeatureFusion:
    """
    Serial-Based Feature Fusion using Pearson Correlation Coefficient
    
    This class implements the feature fusion strategy from the journal.
    It combines features from two different CNN architectures using
    correlation analysis to select the most informative features.
    
    WORKFLOW:
    1. Load trained IRCNN and SACNN models
    2. Extract features from both models
    3. Calculate Pearson correlation between features
    4. Select features with strong correlation
    5. Concatenate features into unified representation
    
    JOURNAL REFERENCE:
    Page 8: "Serial-based fusion using Pearson correlation"
    Equation 4: Correlation coefficient formula
    Figure 6: Feature fusion architecture diagram
    
    USAGE:
        fusion = SerialFeatureFusion(ircnn_model, sacnn_model)
        fused_features = fusion.fuse_features(images)
        # Output: (batch_size, 2688) feature vectors
    """
    
    def __init__(self, ircnn_model, sacnn_model, correlation_threshold=0.3):
        """
        Initialize Feature Fusion Module
        
        Args:
            ircnn_model: Trained Inverted Residual CNN (outputs 1282-dim features)
            sacnn_model: Trained Self-Attention CNN (outputs 1406-dim features)
            correlation_threshold: Minimum correlation for feature selection (default=0.3)
                                  WHY: Journal uses 0.3 to filter weakly correlated features
        """
        self.ircnn_model = ircnn_model
        self.sacnn_model = sacnn_model
        self.correlation_threshold = correlation_threshold
        
        # Get feature dimensions from models
        # IRCNN: 1282 dimensions (from journal specifications)
        # SACNN: 1406 dimensions (from journal specifications)
        self.ircnn_dim = 1282
        self.sacnn_dim = 1406
        
        # Expected fused dimension
        # WHY: Sum of both feature dimensions
        self.fused_dim = self.ircnn_dim + self.sacnn_dim  # 2688 dimensions
        
        print("=" * 70)
        print("FEATURE FUSION MODULE INITIALIZED")
        print("=" * 70)
        print(f"IRCNN Features: {self.ircnn_dim} dimensions")
        print(f"SACNN Features: {self.sacnn_dim} dimensions")
        print(f"Fused Features: {self.fused_dim} dimensions")
        print(f"Correlation Threshold: {self.correlation_threshold}")
        print("=" * 70)
    
    def pearson_correlation(self, u, v):
        """
        Calculate Pearson Correlation Coefficient between two feature vectors
        
        FORMULA (from Journal Equation 4):
        r(U,V) = Σ(Ui - Ū)(Vi - V̄) / √[Σ(Ui - Ū)² × Σ(Vi - V̄)²]
        
        STEP-BY-STEP:
        1. Calculate means: Ū = mean(U), V̄ = mean(V)
        2. Compute deviations: (Ui - Ū), (Vi - V̄)
        3. Numerator: Sum of products of deviations
        4. Denominator: Square root of (sum of squared deviations for U) × (sum for V)
        5. Result: r = numerator / denominator
        
        INTERPRETATION:
        - r close to +1: Strong positive correlation (features move together)
        - r close to -1: Strong negative correlation (features move opposite)
        - r close to 0: Weak correlation (features independent)
        
        WHY THIS MATTERS:
        We want features that are CORRELATED but not IDENTICAL. High correlation
        means both CNNs are capturing related (but complementary) information.
        
        Args:
            u: Feature vector from IRCNN (shape: batch_size × 1282)
            v: Feature vector from SACNN (shape: batch_size × 1406)
            
        Returns:
            Correlation coefficient r (scalar value between -1 and 1)
        """
        # Ensure inputs are numpy arrays
        u = np.array(u)
        v = np.array(v)
        
        # Step 1: Calculate means
        # WHY: We need to center the data around zero
        u_mean = np.mean(u)
        v_mean = np.mean(v)
        
        # Step 2: Calculate deviations from mean
        # WHY: Measures how much each value differs from average
        u_dev = u - u_mean
        v_dev = v - v_mean
        
        # Step 3: Numerator - Sum of products of deviations
        # WHY: Measures how u and v vary together
        numerator = np.sum(u_dev * v_dev)
        
        # Step 4: Denominator - Square root of product of sum of squared deviations
        # WHY: Normalizes the correlation to [-1, 1] range
        u_squared = np.sum(u_dev ** 2)
        v_squared = np.sum(v_dev ** 2)
        denominator = np.sqrt(u_squared * v_squared)
        
        # Avoid division by zero
        if denominator == 0:
            return 0.0
        
        # Step 5: Calculate correlation coefficient
        r = numerator / denominator
        
        return r
    
    def calculate_feature_correlations(self, ircnn_features, sacnn_features):
        """
        Calculate correlation between ALL pairs of features from both CNNs
        
        PROCESS:
        For each IRCNN feature (1282 features):
            For each SACNN feature (1406 features):
                Calculate Pearson correlation
                
        RESULT: Correlation matrix (1282 × 1406)
        Each cell [i, j] contains correlation between IRCNN feature i and SACNN feature j
        
        WHY THIS IS IMPORTANT:
        We need to know which features from IRCNN relate to which features from SACNN.
        This helps us understand what information is shared vs. unique.
        
        JOURNAL REFERENCE:
        Page 8: "Feature correlation analysis for optimal fusion"
        
        Args:
            ircnn_features: Features from IRCNN (batch_size, 1282)
            sacnn_features: Features from SACNN (batch_size, 1406)
            
        Returns:
            correlation_matrix: numpy array (1282, 1406) containing all correlations
        """
        print("\nCalculating feature correlations...")
        print(f"IRCNN features shape: {ircnn_features.shape}")
        print(f"SACNN features shape: {sacnn_features.shape}")
        
        # Initialize correlation matrix
        # WHY: Stores correlation between each pair of features
        n_ircnn = ircnn_features.shape[1]  # 1282
        n_sacnn = sacnn_features.shape[1]  # 1406
        correlation_matrix = np.zeros((n_ircnn, n_sacnn))
        
        # Calculate correlation for each feature pair
        # NOTE: This can be slow for large feature sets
        # OPTIMIZATION: Could use numpy's corrcoef for faster computation
        for i in range(n_ircnn):
            for j in range(n_sacnn):
                # Get feature vectors across all samples
                u = ircnn_features[:, i]
                v = sacnn_features[:, j]
                
                # Calculate correlation
                correlation_matrix[i, j] = self.pearson_correlation(u, v)
        
        print(f"Correlation matrix shape: {correlation_matrix.shape}")
        print(f"Max correlation: {np.max(np.abs(correlation_matrix)):.4f}")
        print(f"Min correlation: {np.min(np.abs(correlation_matrix)):.4f}")
        print(f"Mean correlation: {np.mean(np.abs(correlation_matrix)):.4f}")
        
        return correlation_matrix
    
    def select_correlated_features(self, correlation_matrix):
        """
        Select feature pairs with correlation above threshold
        
        STRATEGY:
        1. Find all correlations with |r| > threshold
        2. Select those feature indices
        3. Return indices for both IRCNN and SACNN
        
        WHY ABSOLUTE VALUE?
        Both positive AND negative correlations are useful!
        - Positive: Features increase together
        - Negative: Features move opposite (still informative!)
        
        JOURNAL REFERENCE:
        Page 8: "Features with |r| > 0.3 are selected for fusion"
        
        Args:
            correlation_matrix: Correlation values (1282 × 1406)
            
        Returns:
            ircnn_indices: List of IRCNN feature indices to keep
            sacnn_indices: List of SACNN feature indices to keep
        """
        # Find where |correlation| > threshold
        # WHY: We want STRONG correlations (positive or negative)
        strong_correlations = np.abs(correlation_matrix) > self.correlation_threshold
        
        # Get indices of selected features
        ircnn_indices, sacnn_indices = np.where(strong_correlations)
        
        # Remove duplicates and sort
        ircnn_indices = np.unique(ircnn_indices)
        sacnn_indices = np.unique(sacnn_indices)
        
        print(f"\nFeature Selection:")
        print(f"Selected IRCNN features: {len(ircnn_indices)} / {correlation_matrix.shape[0]}")
        print(f"Selected SACNN features: {len(sacnn_indices)} / {correlation_matrix.shape[1]}")
        
        return ircnn_indices, sacnn_indices
    
    def fuse_features(self, images, use_correlation_selection=False):
        """
        Main fusion function - Combines features from both CNNs
        
        COMPLETE PROCESS:
        1. Extract features from IRCNN → 1282-dim vector
        2. Extract features from SACNN → 1406-dim vector
        3. (Optional) Select correlated features only
        4. Concatenate features → 2688-dim vector (or less if selection used)
        
        TWO MODES:
        - Simple Fusion (use_correlation_selection=False): Just concatenate all features
        - Smart Fusion (use_correlation_selection=True): Select correlated features first
        
        JOURNAL APPROACH:
        The journal uses simple concatenation for final fusion, but correlation
        analysis is performed for understanding feature relationships.
        
        WHY CONCATENATE?
        Keeps ALL information from both models. The SSA optimization (next step)
        will select the most important features from this combined set.
        
        Args:
            images: Input images (batch_size, 224, 224, 3)
            use_correlation_selection: Whether to filter by correlation (default=False)
            
        Returns:
            fused_features: Combined feature vectors (batch_size, 2688 or less)
        """
        print("\n" + "=" * 70)
        print("FEATURE FUSION IN PROGRESS")
        print("=" * 70)
        
        # Step 1: Extract features from IRCNN
        # WHY: Get lightweight, efficient features
        print("\n[1/4] Extracting IRCNN features...")
        ircnn_features = self.ircnn_model.predict(images, verbose=0)
        print(f"  IRCNN features shape: {ircnn_features.shape}")
        assert ircnn_features.shape[1] == self.ircnn_dim, \
            f"Expected {self.ircnn_dim} IRCNN features, got {ircnn_features.shape[1]}"
        
        # Step 2: Extract features from SACNN
        # WHY: Get self-attention, global context features
        print("\n[2/4] Extracting SACNN features...")
        sacnn_features = self.sacnn_model.predict(images, verbose=0)
        print(f"  SACNN features shape: {sacnn_features.shape}")
        assert sacnn_features.shape[1] == self.sacnn_dim, \
            f"Expected {self.sacnn_dim} SACNN features, got {sacnn_features.shape[1]}"
        
        # Step 3: (Optional) Feature selection based on correlation
        if use_correlation_selection:
            print("\n[3/4] Performing correlation-based feature selection...")
            
            # Calculate correlations
            correlation_matrix = self.calculate_feature_correlations(
                ircnn_features,
                sacnn_features
            )
            
            # Select features
            ircnn_indices, sacnn_indices = self.select_correlated_features(
                correlation_matrix
            )
            
            # Filter features
            ircnn_features = ircnn_features[:, ircnn_indices]
            sacnn_features = sacnn_features[:, sacnn_indices]
            
            print(f"  Selected IRCNN features: {ircnn_features.shape}")
            print(f"  Selected SACNN features: {sacnn_features.shape}")
        else:
            print("\n[3/4] Skipping correlation selection (using all features)")
        
        # Step 4: Concatenate features
        # WHY: Combines complementary information from both models
        # JOURNAL: Page 8 - "Serial concatenation of feature vectors"
        print("\n[4/4] Concatenating features...")
        fused_features = np.concatenate([ircnn_features, sacnn_features], axis=1)
        
        print(f"  Fused features shape: {fused_features.shape}")
        print(f"  Feature dimension: {fused_features.shape[1]}")
        
        print("\n" + "=" * 70)
        print("FEATURE FUSION COMPLETE!")
        print("=" * 70)
        
        return fused_features
    
    def analyze_feature_importance(self, images, n_samples=100):
        """
        Analyze which features contribute most to the fusion
        
        This is a diagnostic tool to understand:
        - Which IRCNN features are most important
        - Which SACNN features are most important
        - How features correlate between models
        
        USAGE:
            fusion.analyze_feature_importance(training_images, n_samples=100)
        
        Args:
            images: Sample images for analysis
            n_samples: Number of samples to use (default=100)
            
        Returns:
            Dictionary with analysis results
        """
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)
        
        # Extract features
        print(f"\nAnalyzing {n_samples} samples...")
        ircnn_features = self.ircnn_model.predict(images[:n_samples], verbose=0)
        sacnn_features = self.sacnn_model.predict(images[:n_samples], verbose=0)
        
        # Calculate feature statistics
        ircnn_mean = np.mean(ircnn_features, axis=0)
        ircnn_std = np.std(ircnn_features, axis=0)
        sacnn_mean = np.mean(sacnn_features, axis=0)
        sacnn_std = np.std(sacnn_features, axis=0)
        
        # Calculate correlations
        correlation_matrix = self.calculate_feature_correlations(
            ircnn_features,
            sacnn_features
        )
        
        # Find most correlated features
        max_corr_idx = np.unravel_index(
            np.argmax(np.abs(correlation_matrix)),
            correlation_matrix.shape
        )
        max_correlation = correlation_matrix[max_corr_idx]
        
        # Results
        results = {
            'ircnn_feature_means': ircnn_mean,
            'ircnn_feature_stds': ircnn_std,
            'sacnn_feature_means': sacnn_mean,
            'sacnn_feature_stds': sacnn_std,
            'correlation_matrix': correlation_matrix,
            'max_correlation': max_correlation,
            'max_corr_indices': max_corr_idx,
            'mean_abs_correlation': np.mean(np.abs(correlation_matrix))
        }
        
        print("\nAnalysis Results:")
        print(f"Max correlation: {max_correlation:.4f}")
        print(f"  Between IRCNN feature {max_corr_idx[0]} and SACNN feature {max_corr_idx[1]}")
        print(f"Mean absolute correlation: {results['mean_abs_correlation']:.4f}")
        print(f"IRCNN features with high variance: {np.sum(ircnn_std > 0.5)}")
        print(f"SACNN features with high variance: {np.sum(sacnn_std > 0.5)}")
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        
        return results


if __name__ == '__main__':
    """
    Test script for Feature Fusion Module
    
    RUN THIS TO:
    - Test fusion with dummy models
    - Verify feature dimensions
    - Check correlation calculations
    """
    print("\n" + "=" * 70)
    print("TESTING FEATURE FUSION MODULE")
    print("=" * 70 + "\n")
    
    # Create dummy models for testing
    print("Creating dummy CNN models...")
    
    # Dummy IRCNN (outputs 1282-dim features)
    ircnn_input = keras.Input(shape=(224, 224, 3))
    ircnn_x = keras.layers.Conv2D(32, 3, activation='relu')(ircnn_input)
    ircnn_x = keras.layers.GlobalAveragePooling2D()(ircnn_x)
    ircnn_output = keras.layers.Dense(1282)(ircnn_x)
    ircnn_model = keras.Model(ircnn_input, ircnn_output, name='DummyIRCNN')
    
    # Dummy SACNN (outputs 1406-dim features)
    sacnn_input = keras.Input(shape=(224, 224, 3))
    sacnn_x = keras.layers.Conv2D(32, 3, activation='relu')(sacnn_input)
    sacnn_x = keras.layers.GlobalAveragePooling2D()(sacnn_x)
    sacnn_output = keras.layers.Dense(1406)(sacnn_x)
    sacnn_model = keras.Model(sacnn_input, sacnn_output, name='DummySACNN')
    
    print("✓ Dummy models created\n")
    
    # Initialize fusion module
    print("Initializing Feature Fusion...")
    fusion = SerialFeatureFusion(ircnn_model, sacnn_model, correlation_threshold=0.3)
    
    # Create dummy images
    print("\nGenerating dummy test data...")
    dummy_images = np.random.rand(10, 224, 224, 3).astype(np.float32)
    print(f"Test images shape: {dummy_images.shape}")
    
    # Test simple fusion
    print("\n" + "-" * 70)
    print("TEST 1: Simple Fusion (no correlation selection)")
    print("-" * 70)
    fused_simple = fusion.fuse_features(dummy_images, use_correlation_selection=False)
    print(f"\nResult: {fused_simple.shape}")
    print(f"Expected: (10, 2688)")
    print(f"Match: {'✓ YES' if fused_simple.shape == (10, 2688) else '✗ NO'}")
    
    # Test correlation-based fusion
    print("\n" + "-" * 70)
    print("TEST 2: Smart Fusion (with correlation selection)")
    print("-" * 70)
    fused_smart = fusion.fuse_features(dummy_images, use_correlation_selection=True)
    print(f"\nResult: {fused_smart.shape}")
    print(f"Features selected: {fused_smart.shape[1]} / 2688")
    
    # Test feature importance analysis
    print("\n" + "-" * 70)
    print("TEST 3: Feature Importance Analysis")
    print("-" * 70)
    importance = fusion.analyze_feature_importance(dummy_images, n_samples=10)
    
    print("\n" + "=" * 70)
    print("FEATURE FUSION TESTING COMPLETE!")
    print("=" * 70 + "\n")
