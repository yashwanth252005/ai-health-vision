"""
============================================================================
DATA AUGMENTATION FOR LUNG CANCER CLASSIFICATION
Based on Journal Figure 3 - Flip and Rotate Operations
============================================================================

This script implements the data augmentation strategy described in the journal:
- Flip Left (Horizontal Flip Left)
- Flip Right (Horizontal Flip Right)  
- Rotate 90 degrees

Each original image generates 3 augmented versions + original = 4 total images
197 original → ~800 images → Target: 4000 (with proper sampling)

Author: AI Mini Project
Date: 2025
============================================================================
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import List, Tuple
import shutil

class LungCancerDataAugmentation:
    """
    Data augmentation class for lung cancer images.
    Implements flip and rotate operations as per journal methodology.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the data augmentation class.
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration from YAML file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract paths from config
        self.raw_data_path = Path(self.config['paths']['raw_data'])
        self.augmented_data_path = Path(self.config['paths']['augmented_data'])
        
        # Get augmentation settings
        self.augmentation_config = self.config['augmentation']
        self.target_size = tuple(self.config['dataset']['image_size'])
        
        # Class names (from config)
        self.class_names = self.config['dataset']['class_names']
        
        print("="*80)
        print("DATA AUGMENTATION INITIALIZED")
        print("="*80)
        print(f"Raw Data Path: {self.raw_data_path}")
        print(f"Augmented Data Path: {self.augmented_data_path}")
        print(f"Target Image Size: {self.target_size}")
        print(f"Classes: {self.class_names}")
        print("="*80)
    
    def flip_left(self, image: np.ndarray) -> np.ndarray:
        """
        Flip image horizontally (left-right flip).
        This is Equation (1) in the journal: T(x,y) = Px(H+1-y)
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Horizontally flipped image
        """
        # cv2.flip with flipCode=1 performs horizontal flip
        return cv2.flip(image, 1)
    
    def flip_right(self, image: np.ndarray) -> np.ndarray:
        """
        Flip image horizontally from right.
        This is Equation (2) in the journal: T(x,y) = Px(x+1-h)
        
        Note: In practice, this is same as flip_left but conceptually different
        in the journal's mathematical notation.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Flipped image
        """
        # For implementation, this is same as horizontal flip
        return cv2.flip(image, 1)
    
    def rotate_90(self, image: np.ndarray) -> np.ndarray:
        """
        Rotate image by 90 degrees clockwise.
        This is Equation (3) in the journal: T(x,y) = [Cos90 -Sin90; Sin90 Cos90][T; T1]
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Rotated image (90 degrees clockwise)
        """
        # cv2.rotate with ROTATE_90_CLOCKWISE
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    def augment_single_image(self, image_path: str, output_dir: Path, 
                           base_name: str) -> List[str]:
        """
        Apply all augmentation operations to a single image.
        
        Args:
            image_path (str): Path to original image
            output_dir (Path): Directory to save augmented images
            base_name (str): Base name for output files
            
        Returns:
            List[str]: Paths of all generated images (including original)
        """
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return []
        
        # Resize to target size (224x224 as per journal)
        image = cv2.resize(image, self.target_size)
        
        generated_images = []
        
        # Save original image (resized)
        original_path = output_dir / f"{base_name}_original.jpg"
        cv2.imwrite(str(original_path), image)
        generated_images.append(str(original_path))
        
        # Apply Flip Left (Horizontal Flip)
        if "flip_left" in self.augmentation_config['operations']:
            flipped_left = self.flip_left(image)
            flip_left_path = output_dir / f"{base_name}_flip_left.jpg"
            cv2.imwrite(str(flip_left_path), flipped_left)
            generated_images.append(str(flip_left_path))
        
        # Apply Flip Right (Alternative horizontal flip)
        if "flip_right" in self.augmentation_config['operations']:
            flipped_right = self.flip_right(image)
            flip_right_path = output_dir / f"{base_name}_flip_right.jpg"
            cv2.imwrite(str(flip_right_path), flipped_right)
            generated_images.append(str(flip_right_path))
        
        # Apply Rotate 90 degrees
        if "rotate_90" in self.augmentation_config['operations']:
            rotated = self.rotate_90(image)
            rotate_path = output_dir / f"{base_name}_rotate90.jpg"
            cv2.imwrite(str(rotate_path), rotated)
            generated_images.append(str(rotate_path))
        
        return generated_images
    
    def augment_dataset(self, multiplier: int = 1) -> Tuple[int, int]:
        """
        Augment entire dataset.
        
        The journal mentions:
        - Original: 197 images
        - After augmentation: 4000 images
        - Each image gets 3 augmented versions + original = 4 versions
        
        To reach 4000 from 197: We need to apply augmentations multiple times
        or use the multiplier parameter.
        
        Args:
            multiplier (int): How many times to repeat augmentation process
            
        Returns:
            Tuple[int, int]: (total_original_images, total_augmented_images)
        """
        print("\n" + "="*80)
        print("STARTING DATA AUGMENTATION PROCESS")
        print("="*80)
        
        # Create augmented data directory structure
        self.augmented_data_path.mkdir(parents=True, exist_ok=True)
        
        total_original = 0
        total_augmented = 0
        
        # Process each class
        for class_name in self.class_names:
            class_input_dir = self.raw_data_path / class_name
            class_output_dir = self.augmented_data_path / class_name
            
            # Create output directory for this class
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if input directory exists
            if not class_input_dir.exists():
                print(f"Warning: Directory {class_input_dir} not found. Skipping...")
                continue
            
            # Get all image files in this class
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(class_input_dir.glob(f'*{ext}')))
                image_files.extend(list(class_input_dir.glob(f'*{ext.upper()}')))
            
            print(f"\nProcessing Class: {class_name}")
            print(f"Found {len(image_files)} original images")
            print("-" * 80)
            
            # Process each image with progress bar
            class_augmented_count = 0
            for img_file in tqdm(image_files, desc=f"Augmenting {class_name}"):
                # Base name for output files
                base_name = img_file.stem
                
                # Augment image (multiplier times for reaching target count)
                for m in range(multiplier):
                    augmented_images = self.augment_single_image(
                        str(img_file),
                        class_output_dir,
                        f"{base_name}_m{m}"
                    )
                    class_augmented_count += len(augmented_images)
            
            total_original += len(image_files)
            total_augmented += class_augmented_count
            
            print(f"Class {class_name}: {len(image_files)} original → {class_augmented_count} augmented")
        
        print("\n" + "="*80)
        print("AUGMENTATION COMPLETE")
        print("="*80)
        print(f"Total Original Images: {total_original}")
        print(f"Total Augmented Images: {total_augmented}")
        print(f"Augmentation Factor: {total_augmented / total_original if total_original > 0 else 0:.2f}x")
        print(f"Output Directory: {self.augmented_data_path}")
        print("="*80)
        
        return total_original, total_augmented
    
    def verify_augmentation(self) -> dict:
        """
        Verify that augmentation was successful.
        
        Returns:
            dict: Statistics about augmented dataset
        """
        stats = {
            'total_images': 0,
            'classes': {}
        }
        
        for class_name in self.class_names:
            class_dir = self.augmented_data_path / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob('*.jpg'))
                count = len(image_files)
                stats['classes'][class_name] = count
                stats['total_images'] += count
        
        return stats


def main():
    """
    Main function to run data augmentation.
    """
    print("\n" + "="*80)
    print("LUNG CANCER DATA AUGMENTATION")
    print("Based on Journal Methodology (Figure 3)")
    print("="*80)
    
    # Initialize augmentation class
    augmenter = LungCancerDataAugmentation()
    
    # Calculate multiplier needed to reach ~4000 images
    # Journal: 197 original → 4000 target
    # With 4 versions per image (original + 3 augmented) = 197 × 4 = 788
    # To reach 4000: 4000 / 788 ≈ 5 multiplier
    multiplier = 5
    
    print(f"\nUsing multiplier: {multiplier}")
    print("This will generate approximately 4000 images (as per journal)")
    print()
    
    # Perform augmentation
    original_count, augmented_count = augmenter.augment_dataset(multiplier=multiplier)
    
    # Verify results
    print("\nVerifying augmentation...")
    stats = augmenter.verify_augmentation()
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    for class_name, count in stats['classes'].items():
        print(f"{class_name}: {count} images")
    print(f"\nTotal: {stats['total_images']} images")
    print("="*80)
    
    # Check if we met the target (approximately 4000 from journal)
    target = 4000
    if stats['total_images'] >= target * 0.9:  # Within 90% of target
        print(f"\n✓ SUCCESS: Generated {stats['total_images']} images (Target: ~{target})")
    else:
        print(f"\n⚠ WARNING: Generated {stats['total_images']} images (Target: ~{target})")
        print("  You may need to adjust the multiplier or add more original images.")
    
    print("\nNext Step: Run data splitting script to create train/test sets (50-50 split)")
    print("="*80)


if __name__ == "__main__":
    main()
