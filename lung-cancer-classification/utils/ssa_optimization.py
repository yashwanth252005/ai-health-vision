"""
============================================================================
SALP SWARM ALGORITHM (SSA) - FEATURE OPTIMIZATION
============================================================================
Based on: Scientific Reports Journal (DOI: 10.1038/s41598-025-93718-7)
Page 9, Algorithm 1, Table 3

WHAT IS SALP SWARM ALGORITHM?
SSA is a bio-inspired optimization algorithm that mimics the swarming
behavior of salps in the ocean. Salps form chains where:
- LEADER salp explores new areas (finds food)
- FOLLOWER salps follow the leader in a chain formation

WHY USE SSA FOR FEATURE SELECTION?
After fusion, we have 2688 features. Not all are equally important!
SSA helps us find the BEST SUBSET of features that give highest accuracy.

HOW IT WORKS:
1. Start with random feature subsets (salp positions)
2. Leader salp moves toward best solution (food source)
3. Follower salps follow the leader
4. Iterate 200 times to find optimal features
5. Return best feature subset

JOURNAL SPECIFICATIONS:
- Population Size: 30 salps
- Max Iterations: 200
- Fitness Function: Standard Error Mean (SEM)
- Search Space: 2688-dimensional (one dimension per feature)

PURPOSE:
Reduces 2688 fused features to ~500-1000 most important features.
This improves:
- Classification accuracy (removes noise)
- Training speed (fewer features)
- Model generalization (prevents overfitting)
============================================================================
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import yaml
import os


class SalpSwarmOptimizer:
    """
    Salp Swarm Algorithm for Feature Selection
    
    This implements the SSA optimization from the journal to select
    the most important features from the fused feature set.
    
    ANALOGY:
    Imagine you have 2688 ingredients (features) to make the best dish
    (classifier). SSA is like a team of 30 chefs (salps) trying different
    combinations for 200 rounds to find the perfect recipe!
    
    WORKFLOW:
    1. Initialize: 30 random feature subsets
    2. Evaluate: Test each subset's classification accuracy
    3. Update: Move salps toward better solutions
    4. Repeat: For 200 iterations
    5. Return: Best feature subset found
    
    JOURNAL REFERENCE:
    Page 9: "SSA with 200 iterations and population 30"
    Algorithm 1: Complete SSA pseudocode
    Table 3: SSA parameters and settings
    
    USAGE:
        ssa = SalpSwarmOptimizer(
            n_features=2688,
            n_salps=30,
            max_iterations=200
        )
        best_features = ssa.optimize(X_train, y_train)
        X_selected = X_train[:, best_features]
    """
    
    def __init__(self, n_features, n_salps=30, max_iterations=200, config_path='config/config.yaml'):
        """
        Initialize SSA Optimizer
        
        Args:
            n_features (int): Total number of features (2688 from fusion)
            n_salps (int): Population size (default=30, from journal)
            max_iterations (int): Number of optimization iterations (default=200, from journal)
            config_path (str): Path to configuration file
        """
        self.n_features = n_features
        self.n_salps = n_salps
        self.max_iterations = max_iterations
        
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                ssa_config = config.get('ssa_optimization', {})
                
                # Override with config values if available
                self.n_salps = ssa_config.get('population_size', n_salps)
                self.max_iterations = ssa_config.get('max_iterations', max_iterations)
        
        # Initialize salp positions (feature selections)
        # Each salp represents a binary vector: 1=feature selected, 0=feature not selected
        # WHY: We need to try different combinations of features
        # SHAPE: (n_salps, n_features) = (30, 2688)
        self.salp_positions = np.random.rand(self.n_salps, self.n_features)
        self.salp_positions = (self.salp_positions > 0.5).astype(int)  # Binary: 0 or 1
        
        # Initialize fitness values for each salp
        # WHY: Tracks how good each feature subset is
        self.fitness_values = np.zeros(self.n_salps)
        
        # Best solution found so far (food source in SSA terminology)
        # WHY: Leader salp moves toward this
        self.food_position = None
        self.food_fitness = float('-inf')  # Start with worst possible fitness
        
        # History tracking for analysis
        self.fitness_history = []
        self.best_fitness_history = []
        
        print("=" * 70)
        print("SALP SWARM ALGORITHM (SSA) INITIALIZED")
        print("=" * 70)
        print(f"Total Features: {self.n_features}")
        print(f"Population Size: {self.n_salps} salps")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Initial feature selection: ~{np.mean(self.salp_positions) * 100:.1f}% features")
        print("=" * 70)
    
    def fitness_function(self, features, X, y):
        """
        Fitness function to evaluate feature subset quality
        
        This uses Standard Error Mean (SEM) as mentioned in the journal.
        Lower SEM = Better feature subset
        
        WHAT IS SEM?
        SEM measures prediction error with confidence:
        SEM = Standard Deviation / √(Number of Samples)
        
        WHY SEM?
        - Considers both accuracy AND consistency
        - Penalizes unstable feature subsets
        - Works well with cross-validation
        
        PROCESS:
        1. Select features based on binary vector
        2. Train simple classifier (Random Forest)
        3. Evaluate with 5-fold cross-validation
        4. Calculate SEM from CV scores
        5. Return negative SEM (we maximize fitness, minimize SEM)
        
        JOURNAL REFERENCE:
        Page 9: "Fitness evaluated using Standard Error Mean"
        Table 3: "Fitness function: SEM"
        
        Args:
            features: Binary vector (1=include feature, 0=exclude)
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            fitness: Negative SEM (higher is better)
        """
        # Select features where binary vector is 1
        # WHY: Only use features this salp has "chosen"
        selected_indices = np.where(features == 1)[0]
        
        # Ensure at least some features are selected
        # WHY: Can't train with zero features!
        if len(selected_indices) < 10:
            return float('-inf')  # Very bad fitness
        
        # Extract selected features
        X_selected = X[:, selected_indices]
        
        try:
            # Train simple classifier
            # WHY: Fast evaluation during optimization
            # NOTE: Random Forest is fast and doesn't need much tuning
            clf = RandomForestClassifier(
                n_estimators=10,  # Few trees for speed
                max_depth=5,      # Shallow for speed
                random_state=42,
                n_jobs=-1         # Use all CPU cores
            )
            
            # Evaluate with cross-validation
            # WHY: Get reliable estimate of performance
            # JOURNAL: Uses 5-fold CV for evaluation
            cv_scores = cross_val_score(
                clf, X_selected, y,
                cv=5,              # 5-fold cross-validation
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Calculate Standard Error Mean
            # SEM = std(scores) / sqrt(number of folds)
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            sem = std_score / np.sqrt(len(cv_scores))
            
            # Fitness = Accuracy - SEM
            # WHY: We want high accuracy AND low variance
            # JOURNAL: Page 9 - "Fitness based on accuracy and stability"
            fitness = mean_score - sem
            
            return fitness
            
        except Exception as e:
            # If evaluation fails, return bad fitness
            print(f"Warning: Fitness evaluation failed: {e}")
            return float('-inf')
    
    def update_food_position(self, X, y):
        """
        Update the food source (best solution found so far)
        
        In SSA, the "food source" represents the best feature subset.
        All salps are attracted to this position.
        
        PROCESS:
        1. Evaluate fitness for all salps
        2. Find salp with best fitness
        3. Update food position if better than current best
        
        WHY THIS MATTERS:
        The food position guides the entire swarm. Better food = better features!
        
        Args:
            X: Feature matrix
            y: Labels
        """
        for i in range(self.n_salps):
            # Evaluate fitness for this salp's feature selection
            fitness = self.fitness_function(self.salp_positions[i], X, y)
            self.fitness_values[i] = fitness
            
            # Update food position if this is the best so far
            # WHY: Leader salp follows best solution
            if fitness > self.food_fitness:
                self.food_fitness = fitness
                self.food_position = self.salp_positions[i].copy()
    
    def update_salp_positions(self, iteration):
        """
        Update positions of all salps in the chain
        
        SSA DYNAMICS (from journal Algorithm 1):
        
        LEADER SALP (first in chain):
        - Moves toward food source with randomness
        - Formula: X1 = F + c2 * c3
          where F = food position, c2/c3 = random coefficients
        
        FOLLOWER SALPS (rest of chain):
        - Follow the salp in front of them
        - Formula: Xi = 0.5 * (Xi + Xi-1)
          where Xi-1 = position of salp ahead
        
        WHY THIS WORKS:
        - Leader explores new areas (exploitation + exploration)
        - Followers exploit good regions found by leader
        - Balance between searching and refining
        
        JOURNAL REFERENCE:
        Page 9: Algorithm 1, Lines 8-15
        Equations for leader and follower updates
        
        Args:
            iteration: Current iteration number (0 to max_iterations)
        """
        # Calculate coefficient c1 (decreases linearly from 2 to 0)
        # WHY: More exploration early, more exploitation later
        # JOURNAL: Page 9 - "c1 decreases from 2 to 0"
        c1 = 2 * np.exp(-(4 * iteration / self.max_iterations) ** 2)
        
        # Update LEADER salp (index 0)
        # WHY: Leader explores space around food source
        for j in range(self.n_features):
            # Random coefficients
            # WHY: Adds stochasticity to avoid local optima
            c2 = np.random.rand()  # [0, 1]
            c3 = np.random.rand()  # [0, 1]
            
            # Update formula from Algorithm 1
            # If c3 < 0.5: move toward food
            # If c3 >= 0.5: move away from food
            if c3 < 0.5:
                # Move toward food position
                self.salp_positions[0, j] = self.food_position[j] + c1 * ((1 - 0) * c2 + 0)
            else:
                # Move away from food position
                self.salp_positions[0, j] = self.food_position[j] - c1 * ((1 - 0) * c2 + 0)
        
        # Update FOLLOWER salps (indices 1 to n_salps-1)
        # WHY: Followers maintain chain formation
        # JOURNAL: Page 9 - "Followers update based on previous salp"
        for i in range(1, self.n_salps):
            # Formula: Xi = 0.5 * (Xi + Xi-1)
            # WHY: Current position + position of salp ahead, averaged
            self.salp_positions[i] = 0.5 * (
                self.salp_positions[i] + self.salp_positions[i - 1]
            )
        
        # Clip positions to [0, 1] and threshold to binary
        # WHY: Positions must be binary (feature selected or not)
        self.salp_positions = np.clip(self.salp_positions, 0, 1)
        self.salp_positions = (self.salp_positions > 0.5).astype(int)
    
    def optimize(self, X, y, verbose=True):
        """
        Main optimization loop - Find best features using SSA
        
        COMPLETE PROCESS:
        1. Initialize salps with random feature selections
        2. For each iteration (200 total):
           a. Evaluate fitness of all salps
           b. Update food position (best solution)
           c. Update leader salp position
           d. Update follower salp positions
           e. Record history
        3. Return best feature subset found
        
        EXPECTED RESULT:
        - Starts with ~1344 features (50% of 2688)
        - Ends with ~500-1000 optimal features
        - Accuracy improves with each iteration
        
        JOURNAL REFERENCE:
        Page 9: Algorithm 1 - Complete SSA pseudocode
        Table 4: Expected feature reduction results
        
        Args:
            X: Feature matrix (n_samples, 2688)
            y: Labels (n_samples,)
            verbose: Whether to print progress (default=True)
            
        Returns:
            best_features: Indices of selected features (numpy array)
            optimization_history: Dictionary with fitness history
        """
        print("\n" + "=" * 70)
        print("STARTING SSA OPTIMIZATION")
        print("=" * 70)
        print(f"Training samples: {X.shape[0]}")
        print(f"Total features: {X.shape[1]}")
        print(f"Optimization iterations: {self.max_iterations}")
        print("=" * 70 + "\n")
        
        # Main optimization loop
        # WHY: Iteratively improves feature selection
        for iteration in range(self.max_iterations):
            # Step 1: Update food position (best solution)
            # WHY: Find current best feature subset
            self.update_food_position(X, y)
            
            # Step 2: Update salp positions
            # WHY: Move swarm toward better solutions
            self.update_salp_positions(iteration)
            
            # Track history
            self.fitness_history.append(self.fitness_values.copy())
            self.best_fitness_history.append(self.food_fitness)
            
            # Print progress
            if verbose and (iteration + 1) % 10 == 0:
                n_selected = np.sum(self.food_position)
                print(f"Iteration {iteration + 1:3d}/{self.max_iterations} | "
                      f"Best Fitness: {self.food_fitness:.4f} | "
                      f"Features Selected: {n_selected:4d}/{self.n_features}")
        
        # Get final best features
        # WHY: These are the optimized feature indices
        best_feature_indices = np.where(self.food_position == 1)[0]
        
        print("\n" + "=" * 70)
        print("SSA OPTIMIZATION COMPLETE!")
        print("=" * 70)
        print(f"Initial features: {self.n_features}")
        print(f"Selected features: {len(best_feature_indices)}")
        print(f"Feature reduction: {(1 - len(best_feature_indices)/self.n_features)*100:.1f}%")
        print(f"Best fitness: {self.food_fitness:.4f}")
        print(f"Expected accuracy: ~{self.food_fitness * 100:.1f}%")
        print("=" * 70 + "\n")
        
        # Prepare optimization history
        optimization_history = {
            'fitness_history': np.array(self.fitness_history),
            'best_fitness_history': np.array(self.best_fitness_history),
            'n_selected_features': len(best_feature_indices),
            'feature_reduction_percent': (1 - len(best_feature_indices)/self.n_features) * 100,
            'final_fitness': self.food_fitness
        }
        
        return best_feature_indices, optimization_history


if __name__ == '__main__':
    """
    Test script for SSA Optimizer
    
    RUN THIS TO:
    - Test SSA with dummy data
    - Verify feature selection
    - Check optimization progress
    """
    print("\n" + "=" * 70)
    print("TESTING SALP SWARM ALGORITHM")
    print("=" * 70 + "\n")
    
    # Create dummy data
    print("Generating dummy test data...")
    n_samples = 200
    n_features = 100  # Using 100 instead of 2688 for faster testing
    
    # Random features and labels
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 3, size=n_samples)  # 3 classes
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Initialize SSA
    print("\n" + "-" * 70)
    print("Initializing SSA Optimizer...")
    ssa = SalpSwarmOptimizer(
        n_features=n_features,
        n_salps=10,          # Reduced for faster testing
        max_iterations=20    # Reduced for faster testing
    )
    
    # Run optimization
    print("\n" + "-" * 70)
    print("Running Optimization...")
    best_features, history = ssa.optimize(X, y, verbose=True)
    
    # Results
    print("\n" + "-" * 70)
    print("Optimization Results:")
    print(f"Selected {len(best_features)} features out of {n_features}")
    print(f"Feature indices (first 10): {best_features[:10]}")
    print(f"Best fitness: {history['final_fitness']:.4f}")
    print(f"Feature reduction: {history['feature_reduction_percent']:.1f}%")
    
    # Test selected features
    print("\n" + "-" * 70)
    print("Testing selected features...")
    X_selected = X[:, best_features]
    print(f"Selected features shape: {X_selected.shape}")
    
    print("\n" + "=" * 70)
    print("SSA TESTING COMPLETE!")
    print("=" * 70 + "\n")
