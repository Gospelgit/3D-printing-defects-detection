
3D Printing Defect Detection - Data Loading Utilities

This module contains utilities for loading and preprocessing data
from the 3D printing defect dataset in COCO format.
Modified to work with the simplified detection model.
"""

import os
import json
import numpy as np
import tensorflow as tf
import cv2
from pycocotools.coco import COCO

# Defining fixed number of anchors to match the simplified model
MAX_DETECTIONS = 100

class COCODataLoader:
    """
    Data loader for COCO format annotations for object detection.
    Handles loading images and annotations from a COCO format dataset
    with annotations split across train/valid/test subdirectories.
    """
    
    def __init__(self, 
                 dataset_dir, 
                 annotation_file='_annotations.coco.json',
                 img_size=(416, 416),
                 augment_train=True):
        """
        Initialize the COCO data loader.
        
        Args:
            dataset_dir: Directory containing the dataset
            annotation_file: Name of the COCO format annotation file
            img_size: Target image size (height, width)
            augment_train: Whether to apply data augmentation during training
        """
        self.dataset_dir = dataset_dir
        self.annotation_file = annotation_file
        self.img_size = img_size
        self.augment_train = augment_train
        
        # Fixed number of detections
        self.max_detections = MAX_DETECTIONS
        
        # Set up paths for annotation files in subdirectories
        self.train_ann_path = os.path.join(dataset_dir, 'train', annotation_file)
        self.valid_ann_path = os.path.join(dataset_dir, 'valid', annotation_file)
        self.test_ann_path = os.path.join(dataset_dir, 'test', annotation_file)
        
        # Check if annotation files exist
        self.has_train_ann = os.path.exists(self.train_ann_path)
        self.has_valid_ann = os.path.exists(self.valid_ann_path)
        self.has_test_ann = os.path.exists(self.test_ann_path)
        
        # Ensure at least one annotation file exists
        if not any([self.has_train_ann, self.has_valid_ann, self.has_test_ann]):
            root_ann_path = os.path.join(dataset_dir, annotation_file)
            if os.path.exists(root_ann_path):
                # Use root annotation file if available
                self.root_ann_path = root_ann_path
                self.has_root_ann = True
            else:
                raise FileNotFoundError(f"No annotation files found in {dataset_dir} or its subdirectories")
        else:
            self.has_root_ann = False
            self.root_ann_path = None
        
        # Load and parse annotations
        self._initialize_annotations()
    
    def _initialize_annotations(self):
        """Load and parse COCO format annotations from all available annotation files."""
        # Initialize empty lists to store combined data
        self.img_ids = []
        self.categories = []
        self.cat_id_to_idx = {}
        self.idx_to_cat_id = {}
        self.img_id_to_path = {}  # Mapping from image ID to file path
        
        # Dictionary to store COCO API instances for each split
        self.coco_api = {}
        
        # If using a single root annotation file
        if self.has_root_ann:
            print(f"Loading annotations from root file: {self.root_ann_path}")
            self.coco_api['root'] = COCO(self.root_ann_path)
            self.img_ids.extend(self.coco_api['root'].getImgIds())
            
            # Get category information
            cat_ids = self.coco_api['root'].getCatIds()
            self.categories = self.coco_api['root'].loadCats(cat_ids)
            
            # Create image ID to path mapping
            for img_id in self.img_ids:
                img_info = self.coco_api['root'].loadImgs(img_id)[0]
                self.img_id_to_path[img_id] = os.path.join(self.dataset_dir, img_info['file_name'])
        
        # Load annotations from train/valid/test subdirectories
        else:
            # Process training annotations
            if self.has_train_ann:
                print(f"Loading annotations from training set: {self.train_ann_path}")
                self.coco_api['train'] = COCO(self.train_ann_path)
                train_img_ids = self.coco_api['train'].getImgIds()
                self.img_ids.extend(train_img_ids)
                
                # Get category information if not already loaded
                if not self.categories:
                    cat_ids = self.coco_api['train'].getCatIds()
                    self.categories = self.coco_api['train'].loadCats(cat_ids)
                
                # Create image ID to path mapping for training images
                for img_id in train_img_ids:
                    img_info = self.coco_api['train'].loadImgs(img_id)[0]
                    self.img_id_to_path[img_id] = os.path.join(self.dataset_dir, 'train', img_info['file_name'])
            
            # Process validation annotations
            if self.has_valid_ann:
                print(f"Loading annotations from validation set: {self.valid_ann_path}")
                self.coco_api['valid'] = COCO(self.valid_ann_path)
                valid_img_ids = self.coco_api['valid'].getImgIds()
                self.img_ids.extend(valid_img_ids)
                
                # Get category information if not already loaded
                if not self.categories:
                    cat_ids = self.coco_api['valid'].getCatIds()
                    self.categories = self.coco_api['valid'].loadCats(cat_ids)
                
                # Create image ID to path mapping for validation images
                for img_id in valid_img_ids:
                    img_info = self.coco_api['valid'].loadImgs(img_id)[0]
                    self.img_id_to_path[img_id] = os.path.join(self.dataset_dir, 'valid', img_info['file_name'])
            
            # Process test annotations
            if self.has_test_ann:
                print(f"Loading annotations from test set: {self.test_ann_path}")
                self.coco_api['test'] = COCO(self.test_ann_path)
                test_img_ids = self.coco_api['test'].getImgIds()
                self.img_ids.extend(test_img_ids)
                
                # Get category information if not already loaded
                if not self.categories:
                    cat_ids = self.coco_api['test'].getCatIds()
                    self.categories = self.coco_api['test'].loadCats(cat_ids)
                
                # Create image ID to path mapping for test images
                for img_id in test_img_ids:
                    img_info = self.coco_api['test'].loadImgs(img_id)[0]
                    self.img_id_to_path[img_id] = os.path.join(self.dataset_dir, 'test', img_info['file_name'])
        
        # Create category ID to index mapping
        self.cat_id_to_idx = {cat['id']: i for i, cat in enumerate(self.categories)}
        self.idx_to_cat_id = {i: cat['id'] for i, cat in enumerate(self.categories)}
        
        self.num_classes = len(self.categories)
        
        print(f"Number of images: {len(self.img_ids)}")
        print(f"Categories: {[cat['name'] for cat in self.categories]}")
        print(f"Number of classes: {self.num_classes}")
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        Split the dataset into training, validation, and test sets.
        If the dataset already has predefined splits, use those instead.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            seed: Random seed for reproducibility
            
        Returns:
            train_ids, val_ids, test_ids: Lists of image IDs for each split
        """
        # If we have predefined splits, use them
        if self.has_train_ann and self.has_valid_ann and self.has_test_ann:
            train_ids = self.coco_api['train'].getImgIds()
            val_ids = self.coco_api['valid'].getImgIds()
            test_ids = self.coco_api['test'].getImgIds()
            
            print(f"Using predefined splits: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test")
            return train_ids, val_ids, test_ids
        
        # If we only have some of the splits, fill in the missing ones
        if self.has_train_ann and self.has_valid_ann:
            train_ids = self.coco_api['train'].getImgIds()
            val_ids = self.coco_api['valid'].getImgIds()
            
            # Use all remaining images as test set
            test_ids = [img_id for img_id in self.img_ids if img_id not in train_ids and img_id not in val_ids]
            
            print(f"Using predefined train/val splits: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test")
            return train_ids, val_ids, test_ids
        
        # If we only have train and test, create validation from part of train
        if self.has_train_ann and self.has_test_ann:
            train_ids = self.coco_api['train'].getImgIds()
            test_ids = self.coco_api['test'].getImgIds()
            
            # Split train into train and validation
            np.random.seed(seed)
            train_ids_shuffled = train_ids.copy()
            np.random.shuffle(train_ids_shuffled)
            
            n_val = int(len(train_ids) * val_ratio / (train_ratio + val_ratio))
            val_ids = train_ids_shuffled[:n_val]
            train_ids = train_ids_shuffled[n_val:]
            
            print(f"Using predefined train/test splits: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test")
            return train_ids, val_ids, test_ids
        
        # Otherwise, create a new split
        np.random.seed(seed)
        
        # Shuffle image IDs
        img_ids = self.img_ids.copy()
        np.random.shuffle(img_ids)
        
        # Split the data
        n_total = len(img_ids)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        train_ids = img_ids[:n_train]
        val_ids = img_ids[n_train:n_train + n_val]
        test_ids = img_ids[n_train + n_val:]
        
        print(f"Created random split: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test")
        
        return train_ids, val_ids, test_ids
    
    def _get_coco_api_for_img(self, img_id):
        """
        Get the COCO API instance for a given image ID.
        
        Args:
            img_id: Image ID
            
        Returns:
            COCO API instance
        """
        # If we have a single root annotation file
        if self.has_root_ann:
            return self.coco_api['root']
        
        # Otherwise, find which split contains this image ID
        for split, coco in self.coco_api.items():
            if img_id in coco.getImgIds():
                return coco
        
        # If not found, return None
        return None
    
    def _load_image(self, img_id):
        """
        Load and preprocess an image.
        
        Args:
            img_id: Image ID
            
        Returns:
            Preprocessed image, original height, original width, image info
        """
        # Get COCO API instance for this image ID
        coco = self._get_coco_api_for_img(img_id)
        if coco is None:
            print(f"Warning: No COCO API instance found for image ID {img_id}")
            return None, 0, 0, None
        
        # Get image info
        img_info = coco.loadImgs(img_id)[0]
        
        # Use the path from our mapping if available
        if img_id in self.img_id_to_path:
            img_path = self.img_id_to_path[img_id]
        else:
            # Determine possible image paths
            possible_paths = [
                os.path.join(self.dataset_dir, img_info['file_name']),
                os.path.join(self.dataset_dir, 'train', img_info['file_name']),
                os.path.join(self.dataset_dir, 'valid', img_info['file_name']),
                os.path.join(self.dataset_dir, 'test', img_info['file_name'])
            ]
            
            # Find the first existing path
            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    break
        
        if img_path is None or not os.path.exists(img_path):
            print(f"Warning: Image not found for ID {img_id}, file name: {img_info['file_name']}")
            return None, 0, 0, img_info
        
        # Read and preprocess image
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read image: {img_path}")
                return None, 0, 0, img_info
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get original dimensions
            original_height, original_width = img.shape[:2]
            
            # Resize image
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            
            # Normalization
            img = img.astype(np.float32) / 255.0
            
            return img, original_height, original_width, img_info
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, 0, 0, img_info
    
    def _load_annotations(self, img_id):
        """
        Load annotations for an image.
        
        Args:
            img_id: Image ID
            
        Returns:
            List of annotations
        """
        # Get COCO API instance for this image ID
        coco = self._get_coco_api_for_img(img_id)
        if coco is None:
            print(f"Warning: No COCO API instance found for image ID {img_id}")
            return []
        
        try:
            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)
            return annotations
        except Exception as e:
            print(f"Error loading annotations for image {img_id}: {e}")
            return []
    
    def _process_annotations(self, annotations, original_height, original_width):
        """
        Process annotations into target format.
        
        Args:
            annotations: List of COCO annotations
            original_height: Original image height
            original_width: Original image width
            
        Returns:
            Processed annotations as numpy arrays (boxes, classes)
        """
        if not annotations or original_height == 0 or original_width == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)
            
        boxes = []
        classes = []
        
        for ann in annotations:
            # Skip invalid annotations
            if 'bbox' not in ann or 'category_id' not in ann:
                continue
                
            # COCO format: [x, y, width, height]
            bbox = ann['bbox']
            category_id = ann['category_id']
            
            # Skip if the annotation is for a category we don't know
            if category_id not in self.cat_id_to_idx:
                continue
            
            # Convert to [x_min, y_min, x_max, y_max]
            x_min = bbox[0] / original_width
            y_min = bbox[1] / original_height
            x_max = (bbox[0] + bbox[2]) / original_width
            y_max = (bbox[1] + bbox[3]) / original_height
            
            # Clip to [0, 1]
            x_min = max(0, min(1, x_min))
            y_min = max(0, min(1, y_min))
            x_max = max(0, min(1, x_max))
            y_max = max(0, min(1, y_max))
            
            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Class index
            class_idx = self.cat_id_to_idx[category_id]
            
            boxes.append([x_min, y_min, x_max, y_max])
            classes.append(class_idx)
        
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)
            
        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)
    
    def _apply_augmentation(self, image, boxes):
        """
        Apply data augmentation to image and boxes.
        
        Args:
            image: Input image
            boxes: Bounding boxes
            
        Returns:
            Augmented image and boxes
        """
        # Only apply augmentation randomly
        if not self.augment_train or np.random.random() > 0.5:
            return image, boxes
        
        # Skip augmentation if no boxes are present
        if len(boxes) == 0:
            return image, boxes
            
        height, width = image.shape[:2]
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            if len(boxes) > 0:
                boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        
        # Random color augmentation
        if np.random.random() > 0.5:
            # Adjust brightness
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 1)
            
            # Adjust contrast
            contrast = np.random.uniform(0.8, 1.2)
            mean = np.mean(image, axis=(0, 1))
            image = np.clip((image - mean) * contrast + mean, 0, 1)
            
            # Adjust saturation
            saturation = np.random.uniform(0.8, 1.2)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 1)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image, boxes
    
    def load_data(self, img_ids):
        """
        Load images and annotations for a set of image IDs.
        
        Args:
            img_ids: List of image IDs to load
            
        Returns:
            images, annotations
        """
        images = []
        all_boxes = []
        all_classes = []
        
        for img_id in img_ids:
            # Load image
            try:
                img, original_height, original_width, img_info = self._load_image(img_id)
                if img is None:
                    continue
            except Exception as e:
                print(f"Warning: Error loading image for ID {img_id}: {str(e)}")
                continue
            
            # Load annotations
            annotations = self._load_annotations(img_id)
            
            # Process annotations
            try:
                boxes, classes = self._process_annotations(annotations, original_height, original_width)
            except Exception as e:
                print(f"Warning: Error processing annotations for ID {img_id}: {str(e)}")
                continue
            
            # Apply augmentation
            if self.augment_train:
                img, boxes = self._apply_augmentation(img, boxes)
            
            images.append(img)
            all_boxes.append(boxes)
            all_classes.append(classes)
        
        # Convert to numpy arrays
        if not images:
            print("Warning: No valid images found")
            return np.array([]), [], []
            
        images = np.array(images)
        
        return images, all_boxes, all_classes
    
    def create_tf_dataset(self, img_ids, batch_size=8, is_training=True, **kwargs):
        """
        Create a TensorFlow dataset for training or evaluation.
        
        Args:
            img_ids: List of image IDs
            batch_size: Batch size
            is_training: Whether the dataset is for training
            **kwargs: Additional arguments for backward compatibility
            
        Returns:
            TensorFlow dataset
        """
        # Ignore 'anchors' parameter if passed
        if 'anchors' in kwargs:
            # Just log that we're ignoring it
            print("Ignoring 'anchors' parameter - using fixed output sizes")
        
        # Set augmentation flag based on is_training
        self.augment_train = is_training and self.augment_train
        
        # Load all data
        images, all_boxes, all_classes = self.load_data(img_ids)
        
        if len(images) == 0:
            raise ValueError("No valid images found for dataset creation")
        
        # Create target tensors with fixed sizes
        regression_targets, classification_targets = self._generate_fixed_targets(
            all_boxes, all_classes
        )
        
        # Verify shapes match expected
        print(f"Regression targets shape: {regression_targets.shape}")
        print(f"Classification targets shape: {classification_targets.shape}")
        
        dataset = tf.data.Dataset.from_tensor_slices(
            (images, {'regression_output': regression_targets, 
                     'classification_output': classification_targets})
        )
        
        # Configure dataset
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(images))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return dataset
    
    def _generate_fixed_targets(self, all_boxes, all_classes):
        """
        Generate fixed-size regression and classification targets for all images.
        This matches the simplified model with MAX_DETECTIONS output.
        
        Args:
            all_boxes: List of bounding boxes for each image
            all_classes: List of class IDs for each image
            
        Returns:
            regression_targets, classification_targets
        """
        num_images = len(all_boxes)
        
        if num_images == 0:
            raise ValueError("No images to generate targets from")
        
        # Initialize targets with fixed size - MAX_DETECTIONS
        regression_targets = np.zeros((num_images, self.max_detections, 4), dtype=np.float32)
        classification_targets = np.zeros((num_images, self.max_detections, self.num_classes), dtype=np.float32)
        
        # Process each image
        for i in range(num_images):
            boxes = all_boxes[i]
            classes = all_classes[i]
            
            if len(boxes) == 0:
                continue
            
            # Limit the number of boxes to MAX_DETECTIONS
            num_boxes = min(len(boxes), self.max_detections)
            
            # Add boxes to regression targets (first come, first served)
            regression_targets[i, :num_boxes] = boxes[:num_boxes]
            
            # Create classification targets (one-hot encoding)
            for j in range(num_boxes):
                classification_targets[i, j, classes[j]] = 1.0
        
        return regression_targets, classification_targets
    
    # For backward compatibility
    def get_anchors(self, image_height, image_width):
        """
        Legacy method - not used in simplified model.
        Kept for backward compatibility.
        """
        print("Warning: get_anchors() is deprecated in simplified model")
        return np.zeros((self.max_detections, 4), dtype=np.float32)
    
    def visualize_sample(self, img_id, output_path=None, show=True):
        """
        Visualize a sample image with its annotations.
        
        Args:
            img_id: Image ID
            output_path: Path to save the visualization
            show: Whether to display the image
            
        Returns:
            Visualization image
        """
        # Load image and annotations
        img, original_height, original_width, img_info = self._load_image(img_id)
        if img is None:
            print(f"Error: Unable to load image for ID {img_id}")
            return None
        
        annotations = self._load_annotations(img_id)
        boxes, classes = self._process_annotations(annotations, original_height, original_width)
        
        # Convert normalized boxes to pixel coordinates
        img_height, img_width = img.shape[:2]
        boxes_pixel = []
        for box in boxes:
            x_min = int(box[0] * img_width)
            y_min = int(box[1] * img_height)
            x_max = int(box[2] * img_width)
            y_max = int(box[3] * img_height)
            boxes_pixel.append([x_min, y_min, x_max, y_max])
        
        # Create visualization image
        vis_img = (img * 255).astype(np.uint8).copy()
        
        # Draw boxes and labels
        for i, box in enumerate(boxes_pixel):
            x_min, y_min, x_max, y_max = box
            class_id = classes[i]
            class_name = self.categories[class_id]['name']
            
            # Draw box
            cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}"
            cv2.putText(vis_img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save or show image
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"Saved visualization to {output_path}")
        
        if show:
            try:
                from matplotlib import pyplot as plt
                plt.figure(figsize=(10, 10))
                plt.imshow(vis_img)
                plt.axis('off')
                plt.title(f"Image ID: {img_id}")
                plt.show()
            except ImportError:
                print("Matplotlib not available for visualization")
        
        return vis_img
