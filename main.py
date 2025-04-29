import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, Reshape, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define object detection model (simplified example using RetinaNet-like approach)
def build_detection_model(input_size=(416, 416, 3), num_classes=1):
    """
    Build a simplified object detection model based on RetinaNet concepts
    
    Args:
        input_size: Input image dimensions
        num_classes: Number of object classes to detect
    """
    # Base feature extractor (ResNet50)
    inputs = Input(shape=input_size)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Feature Pyramid Network (simplified)
    C5 = base_model.output
    P5 = Conv2D(256, kernel_size=1, strides=1, padding='same')(C5)
    
    # Detection head
    bbox_features = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(P5)
    bbox_features = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(bbox_features)
    
    # Outputs: [x, y, width, height, objectness, class_probs...]
    num_anchors = 9  # 3 scales × 3 aspect ratios
    detection_outputs = Conv2D(num_anchors * (5 + num_classes), kernel_size=3, padding='same')(bbox_features)
    output_shape = tf.concat([tf.shape(detection_outputs)[:-1], [num_anchors, 5 + num_classes]], axis=0)
    detection_outputs = Reshape((-1, 5 + num_classes))(detection_outputs)
    
    model = Model(inputs, detection_outputs)
    return model

# Load COCO format annotations
def load_coco_dataset(dataset_dir, annotation_file='_annotations.coco.json'):
    """
    Load dataset in COCO format
    
    Args:
        dataset_dir: Directory containing images and annotations
        annotation_file: COCO format annotation filename
    """
    # Load annotations
    with open(os.path.join(dataset_dir, annotation_file)) as f:
        coco_data = json.load(f)
    
    # Process images and annotations
    images = []
    annotations = []
    
    # Create lookup dictionaries
    image_dict = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Process each image and its annotations
    for img_id, img_info in image_dict.items():
        img_path = os.path.join(dataset_dir, 'train', img_info['file_name'])
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset_dir, 'valid', img_info['file_name'])
            if not os.path.exists(img_path):
                img_path = os.path.join(dataset_dir, 'test', img_info['file_name'])
        
        if os.path.exists(img_path):
            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (416, 416))
            img = img.astype(np.float32) / 255.0
            
            # Get annotations for this image
            img_annotations = annotations_by_image.get(img_id, [])
            boxes = []
            for ann in img_annotations:
                # COCO format: [x, y, width, height]
                bbox = ann['bbox']
                category_id = ann['category_id']
                
                # Convert to [x_min, y_min, x_max, y_max, class_id]
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = bbox[0] + bbox[2]
                y_max = bbox[1] + bbox[3]
                
                # Normalize coordinates (0-1)
                x_min = x_min / img_info['width']
                y_min = y_min / img_info['height']
                x_max = x_max / img_info['width']
                y_max = y_max / img_info['height']
                
                boxes.append([x_min, y_min, x_max, y_max, category_id])
            
            images.append(img)
            annotations.append(np.array(boxes))
    
    return np.array(images), annotations, categories

# Visualization function
def visualize_predictions(image, true_boxes, pred_boxes, categories, threshold=0.5):
    """
    Visualize detection predictions
    
    Args:
        image: Input image (normalized 0-1)
        true_boxes: Ground truth boxes [x_min, y_min, x_max, y_max, class_id]
        pred_boxes: Predicted boxes [x_min, y_min, x_max, y_max, objectness, class_probs...]
        categories: Dictionary mapping category IDs to names
        threshold: Confidence threshold for showing predictions
    """
    plt.figure(figsize=(10, 10))
    
    # Convert image back to 0-255 range if normalized
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    plt.imshow(image)
    
    # Plot ground truth boxes
    for box in true_boxes:
        x_min, y_min, x_max, y_max, class_id = box
        width = x_max - x_min
        height = y_max - y_min
        
        # Convert normalized coordinates to pixel values
        x_min = int(x_min * image.shape[1])
        y_min = int(y_min * image.shape[0])
        width = int(width * image.shape[1])
        height = int(height * image.shape[0])
        
        rect = plt.Rectangle((x_min, y_min), width, height, 
                            fill=False, edgecolor='green', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(x_min, y_min - 5, categories[class_id], 
                color='green', fontsize=10, backgroundcolor='black')
    
    # Plot predicted boxes
    for box in pred_boxes:
        if box[4] > threshold:  # Check objectness confidence
            x_min, y_min, x_max, y_max = box[:4]
            objectness = box[4]
            class_id = np.argmax(box[5:])
            class_prob = box[5 + class_id]
            confidence = objectness * class_prob
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Convert normalized coordinates to pixel values
            x_min = int(x_min * image.shape[1])
            y_min = int(y_min * image.shape[0])
            width = int(width * image.shape[1])
            height = int(height * image.shape[0])
            
            rect = plt.Rectangle((x_min, y_min), width, height, 
                                fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x_min, y_min - 5, 
                    f"{categories[class_id]} {confidence:.2f}", 
                    color='red', fontsize=10, backgroundcolor='black')
    
    plt.axis('off')
    return plt

# Main function to train and evaluate
def train_and_evaluate_model(dataset_dir, epochs=50, batch_size=8):
    """
    Train and evaluate the detection model
    
    Args:
        dataset_dir: Directory containing the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    # Load dataset
    images, annotations, categories = load_coco_dataset(dataset_dir)
    
    # Split dataset
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    
    train_split = int(0.7 * len(indices))
    val_split = int(0.85 * len(indices))
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    X_train = images[train_indices]
    y_train = [annotations[i] for i in train_indices]
    
    X_val = images[val_indices]
    y_val = [annotations[i] for i in val_indices]
    
    X_test = images[test_indices]
    y_test = [annotations[i] for i in test_indices]
    
    # Build model
    model = build_detection_model(input_size=(416, 416, 3), num_classes=len(categories))
    
    # Compile model (Note: this is simplified - actual detection models use complex loss functions)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mse',  # This is oversimplified - should use proper detection loss
        metrics=['accuracy']  # Also oversimplified
    )
    
    # Train model (Note: This is a placeholder - actual implementation would need a custom training loop)
    # model.fit(...)
    
    # Display model architecture
    model.summary()
    
    print("Note: This code example is a simplified framework.")
    print("A complete implementation would require custom loss functions, anchor generation,")
    print("non-maximum suppression, and other object detection components.")
    
    # Return the model and dataset information
    return model, (X_test, y_test, categories)

# Example usage
if __name__ == "__main__":
    dataset_dir = "path/to/3d_printing_pictures"
    model, test_data = train_and_evaluate_model(dataset_dir)
    
    # Example visualization
    X_test, y_test, categories = test_data
    if len(X_test) > 0:
        # This is a placeholder - would need actual predictions
        sample_image = X_test[0]
        sample_boxes = y_test[0]
        
        # In a real implementation, you would get predictions from the model
        # pred_boxes = model.predict(np.expand_dims(sample_image, axis=0))[0]
        
        # For demonstration, use the true boxes as "predictions"
        pred_boxes = np.concatenate([sample_boxes[:, :4], 
                                    np.ones((len(sample_boxes), 1)),  # objectness
                                    np.eye(len(categories))[sample_boxes[:, 4].astype(int)]], axis=1)
        
        plt = visualize_predictions(sample_image, sample_boxes, pred_boxes, categories)
        plt.savefig("example_detection.png")
        plt.close()
