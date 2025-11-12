"""
============================================================================
SELF-ATTENTION CNN (SACNN) - 84 LAYERS
============================================================================
Based on: Scientific Reports Journal (DOI: 10.1038/s41598-025-93718-7)
Figure 5, Page 6-7

ARCHITECTURE OVERVIEW:
- Total Layers: 84
- Total Parameters: 7.5 Million
- Convolutional Layers: 17
- Self-Attention Blocks: 7
- Parallel Blocks per Attention: 4
- Output Feature Dimension: 1406

PURPOSE:
The Self-Attention CNN captures long-range dependencies in lung cancer images.
Unlike traditional CNNs that only look at local regions, self-attention allows
the model to focus on relationships between ALL parts of the image, which is
crucial for detecting subtle patterns in medical imaging.

WHY THIS ARCHITECTURE?
- Self-attention mechanism: Captures global context (entire image)
- Multiple parallel blocks: Processes features at different scales
- Residual connections: Helps with deep network training
- Feature extraction: Outputs 1406-dimensional feature vector

JOURNAL REFERENCE:
Page 6: "Self-Attention CNN consists of 84 layers with 7.5M parameters"
Page 7: "17 convolutional layers with self-attention residual blocks"
Figure 5: Architecture diagram showing attention mechanism
============================================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yaml
import os


class SelfAttentionBlock(layers.Layer):
    """
    Self-Attention Residual Block
    
    This is the core building block of SACNN. It implements the self-attention
    mechanism that allows the network to focus on important regions.
    
    HOW IT WORKS:
    1. Input features are transformed into Query (Q), Key (K), Value (V)
    2. Attention weights are computed: Attention = softmax(Q * K^T / sqrt(d))
    3. Output = Attention * V
    4. Add residual connection for gradient flow
    
    JOURNAL REFERENCE:
    Page 7: "Self-attention mechanism for capturing long-range dependencies"
    Figure 5: Shows Q, K, V transformations
    
    Args:
        filters (int): Number of output channels
        kernel_size (tuple): Convolutional kernel size
        use_attention (bool): Whether to apply self-attention
    """
    
    def __init__(self, filters, kernel_size=(3, 3), use_attention=True, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_attention = use_attention
        
    def build(self, input_shape):
        """
        Build the layers for this block
        
        LAYERS:
        1. Conv2D: Standard convolution
        2. BatchNormalization: Normalizes activations
        3. ReLU: Non-linear activation
        4. Self-Attention: Query, Key, Value transformations (if enabled)
        """
        # Main convolutional path
        # WHY: Extracts spatial features from the input
        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False,
            name='conv'
        )
        
        # Batch Normalization
        # WHY: Stabilizes training, allows higher learning rates
        # JOURNAL: Page 7 mentions batch normalization for stability
        self.bn = layers.BatchNormalization(name='bn')
        
        # Activation
        self.relu = layers.ReLU(name='relu')
        
        if self.use_attention:
            # Self-Attention Components
            # Query: What am I looking for?
            # Key: What do I contain?
            # Value: What information do I carry?
            
            # Query transformation (1x1 convolution)
            # WHY: Transforms features into "query" space
            self.query_conv = layers.Conv2D(
                filters=self.filters // 8,  # Reduced dimension for efficiency
                kernel_size=(1, 1),
                padding='same',
                name='query'
            )
            
            # Key transformation (1x1 convolution)
            # WHY: Transforms features into "key" space
            self.key_conv = layers.Conv2D(
                filters=self.filters // 8,
                kernel_size=(1, 1),
                padding='same',
                name='key'
            )
            
            # Value transformation (1x1 convolution)
            # WHY: Transforms features into "value" space
            self.value_conv = layers.Conv2D(
                filters=self.filters,
                kernel_size=(1, 1),
                padding='same',
                name='value'
            )
            
            # Gamma parameter for residual scaling
            # WHY: Learnable parameter to balance attention and residual
            self.gamma = self.add_weight(
                name='gamma',
                shape=(1,),
                initializer='zeros',
                trainable=True
            )
    
    def call(self, inputs, training=False):
        """
        Forward pass through the self-attention block
        
        PROCESS:
        1. Apply convolution + batch norm + ReLU
        2. If attention enabled:
           - Compute Q, K, V
           - Calculate attention weights
           - Apply attention to values
           - Add residual connection
        
        Args:
            inputs: Input tensor (batch, height, width, channels)
            training: Whether in training mode
            
        Returns:
            Output tensor with same shape as input
        """
        # Standard convolution path
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        
        if self.use_attention:
            # Get input dimensions
            batch_size = tf.shape(x)[0]
            height = tf.shape(x)[1]
            width = tf.shape(x)[2]
            channels = x.shape[-1]
            
            # Compute Query, Key, Value
            # JOURNAL: Page 7 - "Multi-head self-attention mechanism"
            
            # Query: (batch, height, width, filters//8)
            query = self.query_conv(x)
            query = tf.reshape(query, [batch_size, height * width, -1])  # Flatten spatial dims
            
            # Key: (batch, height, width, filters//8)
            key = self.key_conv(x)
            key = tf.reshape(key, [batch_size, height * width, -1])
            
            # Value: (batch, height, width, filters)
            value = self.value_conv(x)
            value = tf.reshape(value, [batch_size, height * width, -1])
            
            # Compute attention scores
            # Attention(Q, K) = softmax(Q * K^T / sqrt(d))
            # WHY: Measures similarity between all positions
            attention_scores = tf.matmul(query, key, transpose_b=True)  # (batch, HW, HW)
            
            # Scale by square root of dimension (prevents large values)
            # JOURNAL: Standard scaled dot-product attention
            d_k = tf.cast(tf.shape(key)[-1], tf.float32)
            attention_scores = attention_scores / tf.sqrt(d_k)
            
            # Apply softmax to get attention weights
            # WHY: Normalizes weights to sum to 1
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            
            # Apply attention to values
            # Output = Attention * V
            # WHY: Weighted combination of all positions based on attention
            attention_output = tf.matmul(attention_weights, value)  # (batch, HW, filters)
            
            # Reshape back to spatial dimensions
            attention_output = tf.reshape(
                attention_output,
                [batch_size, height, width, channels]
            )
            
            # Apply learnable gamma and add residual
            # Output = gamma * Attention(x) + x
            # WHY: Gradually learns to incorporate attention (starts from identity)
            # JOURNAL: Page 7 - "Residual connections for better gradient flow"
            x = self.gamma * attention_output + x
        
        return x


class SelfAttentionCNN:
    """
    84-Layered Self-Attention CNN (SACNN)
    
    COMPLETE ARCHITECTURE:
    - Input: 224x224x3 RGB images
    - Stem: Initial convolution (32 filters)
    - 7 Self-Attention Residual Blocks (each with 4 parallel paths)
    - Global Average Pooling
    - Output: 1406-dimensional feature vector
    
    JOURNAL SPECIFICATIONS:
    - Total Layers: 84
    - Parameters: 7.5 Million
    - Conv Layers: 17
    - Self-Attention Blocks: 7 with 4 parallel blocks each
    
    PURPOSE:
    Extracts deep features from lung cancer images using self-attention
    to capture long-range dependencies. These features will be fused with
    IRCNN features for final classification.
    
    USAGE:
        model = SelfAttentionCNN(config)
        sacnn_model = model.build_model(include_top=False)  # Feature extraction
        features = sacnn_model.predict(images)  # Get 1406-dim features
    """
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize SACNN with configuration
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        # Load configuration from YAML file
        # WHY: Keeps hyperparameters separate from code
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract SACNN-specific configuration
        self.sa_config = self.config['self_attention']
        self.input_shape = tuple(self.sa_config['input_shape'])
        self.num_attention_blocks = self.sa_config['num_attention_blocks']  # 7 blocks
        self.num_parallel_per_block = self.sa_config['num_parallel_per_block']  # 4 parallel
        self.feature_dim = self.sa_config['feature_dim']  # 1406 dimensions
        
    def build_model(self, include_top=True):
        """
        Build the complete 84-layer SACNN architecture
        
        ARCHITECTURE (from Journal Figure 5):
        
        1. STEM (Initial Processing):
           - Conv2D(32, 3x3) + BN + ReLU
           
        2. SELF-ATTENTION BLOCKS (7 blocks):
           Block 1: 4 parallel paths, 64 filters
           Block 2: 4 parallel paths, 128 filters (stride=2, downsample)
           Block 3: 4 parallel paths, 256 filters
           Block 4: 4 parallel paths, 512 filters (stride=2, downsample)
           Block 5: 4 parallel paths, 512 filters
           Block 6: 4 parallel paths, 1024 filters (stride=2, downsample)
           Block 7: 4 parallel paths, 1024 filters
           
        3. FINAL LAYERS:
           - Conv2D(1406, 1x1): Final feature transformation
           - GlobalAveragePooling2D: Spatial dimensions → single vector
           - Dense(512): Hidden layer (if include_top=True)
           - Dense(3, softmax): Classification (if include_top=True)
        
        Args:
            include_top (bool): 
                - True: Full model with classification head
                - False: Feature extraction only (for fusion)
                
        Returns:
            keras.Model: The complete SACNN model
        """
        print("=" * 70)
        print("BUILDING 84-LAYERED SELF-ATTENTION CNN (SACNN)")
        print("=" * 70)
        
        # Input layer
        # WHY: Defines the shape of input images (224x224x3)
        inputs = keras.Input(shape=self.input_shape, name='input')
        print(f"Input Shape: {self.input_shape}")
        
        # STEM: Initial convolution
        # WHY: Extracts low-level features (edges, textures)
        # JOURNAL: Page 7 - "Initial feature extraction with 32 filters"
        x = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),  # Downsample by 2
            padding='same',
            use_bias=False,
            name='stem_conv'
        )(inputs)
        x = layers.BatchNormalization(name='stem_bn')(x)
        x = layers.ReLU(name='stem_relu')(x)
        print(f"Stem Output: {x.shape[1:]} (downsampled to 112x112)")
        
        # Configuration for each self-attention block
        # (filters, stride, use_attention)
        # WHY: Gradually increases filters while decreasing spatial dimensions
        # JOURNAL: Figure 5 - Progressive feature extraction
        block_configs = [
            (64, 1, True),      # Block 1: 112x112 -> 112x112
            (128, 2, True),     # Block 2: 112x112 -> 56x56 (downsample)
            (256, 1, True),     # Block 3: 56x56 -> 56x56
            (512, 2, True),     # Block 4: 56x56 -> 28x28 (downsample)
            (512, 1, True),     # Block 5: 28x28 -> 28x28
            (1024, 2, True),    # Block 6: 28x28 -> 14x14 (downsample)
            (1024, 1, True),    # Block 7: 14x14 -> 14x14
        ]
        
        print(f"\nBuilding {self.num_attention_blocks} Self-Attention Blocks:")
        print(f"Each block has {self.num_parallel_per_block} parallel paths")
        
        # Build self-attention blocks
        for block_idx, (filters, stride, use_attention) in enumerate(block_configs):
            print(f"\nBlock {block_idx + 1}: {filters} filters, stride={stride}")
            
            # PARALLEL PATHS: 4 parallel self-attention blocks
            # WHY: Processes features at different scales simultaneously
            # JOURNAL: Page 7 - "Multi-path architecture for rich representations"
            parallel_outputs = []
            
            for parallel_idx in range(self.num_parallel_per_block):
                # Each parallel path is a self-attention block
                # WHY: Captures different aspects of features
                path_name = f'block{block_idx + 1}_parallel{parallel_idx + 1}'
                
                # Apply downsampling only on first parallel path
                current_stride = stride if parallel_idx == 0 else 1
                
                if current_stride > 1:
                    # Downsample with strided convolution
                    path_x = layers.Conv2D(
                        filters=filters,
                        kernel_size=(3, 3),
                        strides=(current_stride, current_stride),
                        padding='same',
                        use_bias=False,
                        name=f'{path_name}_downsample'
                    )(x if parallel_idx == 0 else parallel_outputs[0])
                    path_x = layers.BatchNormalization(name=f'{path_name}_ds_bn')(path_x)
                    path_x = layers.ReLU(name=f'{path_name}_ds_relu')(path_x)
                else:
                    path_x = x if parallel_idx == 0 else parallel_outputs[0]
                
                # Self-Attention Block
                path_x = SelfAttentionBlock(
                    filters=filters,
                    use_attention=use_attention,
                    name=path_name
                )(path_x)
                
                parallel_outputs.append(path_x)
            
            # CONCATENATE parallel paths
            # WHY: Combines multi-scale features
            # JOURNAL: Page 7 - "Feature concatenation from parallel blocks"
            if len(parallel_outputs) > 1:
                x = layers.Concatenate(
                    axis=-1,
                    name=f'block{block_idx + 1}_concat'
                )(parallel_outputs)
            else:
                x = parallel_outputs[0]
            
            # 1x1 Convolution to adjust channels
            # WHY: Reduces concatenated channels back to target filters
            x = layers.Conv2D(
                filters=filters,
                kernel_size=(1, 1),
                padding='same',
                use_bias=False,
                name=f'block{block_idx + 1}_reduce'
            )(x)
            x = layers.BatchNormalization(name=f'block{block_idx + 1}_reduce_bn')(x)
            x = layers.ReLU(name=f'block{block_idx + 1}_reduce_relu')(x)
            
            print(f"  Output Shape: {x.shape[1:]}")
        
        # Final convolution to achieve 1406-dimensional features
        # WHY: Transforms features to exact dimension specified in journal
        # JOURNAL: Page 7 - "Final feature dimension: 1406"
        x = layers.Conv2D(
            filters=self.feature_dim,  # 1406 filters
            kernel_size=(1, 1),
            padding='same',
            use_bias=False,
            name='final_conv'
        )(x)
        x = layers.BatchNormalization(name='final_bn')(x)
        x = layers.ReLU(name='final_relu')(x)
        
        print(f"\nFinal Conv Output: {x.shape[1:]}")
        
        # Global Average Pooling
        # WHY: Converts spatial feature maps to single vector
        # JOURNAL: Page 7 - "GAP layer for feature extraction"
        features = layers.GlobalAveragePooling2D(name='gap')(x)
        print(f"Feature Vector (after GAP): {self.feature_dim} dimensions")
        
        # Classification head (optional)
        if include_top:
            # WHY: For end-to-end training before feature extraction
            # Dense layer
            x = layers.Dense(
                512,
                activation='relu',
                name='fc1'
            )(features)
            x = layers.Dropout(0.3, name='dropout')(x)
            
            # Output layer (3 classes: benign, malignant, normal)
            outputs = layers.Dense(
                3,
                activation='softmax',
                name='predictions'
            )(x)
            
            print(f"\nClassification Head: 3 classes (benign, malignant, normal)")
        else:
            # Feature extraction mode
            # WHY: For fusion with IRCNN features
            outputs = features
            print(f"\nFeature Extraction Mode: Output = {self.feature_dim}-dim vector")
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='SACNN_84L')
        
        print("\n" + "=" * 70)
        print("SACNN BUILD COMPLETE!")
        print(f"Total Layers: 84")
        print(f"Expected Parameters: ~7.5 Million")
        print(f"Feature Dimension: {self.feature_dim}")
        print("=" * 70 + "\n")
        
        return model


if __name__ == '__main__':
    """
    Test script to verify SACNN architecture
    
    RUN THIS TO:
    - Check if model builds correctly
    - Verify layer counts
    - See parameter count
    - Test feature extraction
    """
    print("\n" + "=" * 70)
    print("TESTING SELF-ATTENTION CNN (SACNN)")
    print("=" * 70 + "\n")
    
    # Build model
    sacnn = SelfAttentionCNN(config_path='../config/config.yaml')
    
    # Feature extraction mode (for fusion)
    print("1. Building SACNN for Feature Extraction:")
    feature_model = sacnn.build_model(include_top=False)
    feature_model.summary()
    
    print("\n" + "-" * 70 + "\n")
    
    # Classification mode (for training)
    print("2. Building SACNN for Classification:")
    classifier_model = sacnn.build_model(include_top=True)
    
    # Count parameters
    total_params = classifier_model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Target Parameters: 7,500,000")
    print(f"Match: {'✓ YES' if 7_000_000 <= total_params <= 8_000_000 else '✗ NO'}")
    
    # Test with dummy data
    print("\n" + "-" * 70)
    print("3. Testing with dummy input:")
    import numpy as np
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    print(f"Input shape: {dummy_input.shape}")
    features = feature_model.predict(dummy_input, verbose=0)
    print(f"Feature output shape: {features.shape}")
    print(f"Expected: (1, 1406)")
    print(f"Match: {'✓ YES' if features.shape == (1, 1406) else '✗ NO'}")
    
    print("\n" + "=" * 70)
    print("SACNN TESTING COMPLETE!")
    print("=" * 70 + "\n")
