import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

# U-Net model definitions
def build_unet(input_size=(256, 256, 1), num_classes=12):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bottom
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Output layer
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv7)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Loading all datasets from HDF5 files
def load_all_hdf5_datasets(hdf5_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Load all datasets from HDF5 files and split into train, val, and test sets
    
    Args:
        hdf5_files: List of HDF5 file paths
        train_ratio, val_ratio, test_ratio: Ratios for splitting the data
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing train, val, and test datasets
    """
    np.random.seed(seed)
    
    
    all_images = []
    all_masks = []
    
    # Loading data from each HDF5 file
    for hdf5_file in hdf5_files:
        print(f"Loading dataset from {hdf5_file}...")
        try:
            with h5py.File(hdf5_file, 'r') as f:

                # Exploring the structure of the HDF5 file
                print(f"HDF5 file structure: {list(f.keys())}")
                
              
                if 'images' in f and 'masks' in f:
                    images = f['images'][:]
                    masks = f['masks'][:]
                    
                    # Normalizing images to [0,1] range
                    images = images.astype(np.float32) / 255.0
                    
                    # Adding channel dimension if not present
                    if len(images.shape) == 3:  # (num_samples, height, width)
                        images = np.expand_dims(images, axis=-1)
                   

                    if len(masks.shape) == 3:  # (num_samples, height, width)
                        # Assuming masks contain class indices
                        masks_one_hot = tf.keras.utils.to_categorical(masks, num_classes=12)
                    elif len(masks.shape) == 4 and masks.shape[-1] == 12:
                        # Masks are already one-hot encoded
                        masks_one_hot = masks
                    else:
                        raise ValueError(f"Unexpected mask shape: {masks.shape}")
                    
                    all_images.append(images)
                    all_masks.append(masks_one_hot)
                    
                    print(f"Loaded {images.shape[0]} samples from {hdf5_file}")
                else:
                    print(f"Warning: Expected 'images' and 'masks' datasets not found in {hdf5_file}")
                    # Try to find the actual dataset keys
                    print(f"Available keys: {list(f.keys())}")
                    
        except Exception as e:
            print(f"Error loading {hdf5_file}: {str(e)}")
    
    # Combine all data
    if all_images and all_masks:
        all_images = np.vstack(all_images)
        all_masks = np.vstack(all_masks)
        print(f"Combined dataset: {all_images.shape[0]} samples")
    else:
        raise ValueError("No data was loaded from the HDF5 files")
    
    # Shuffling and split the data
    indices = np.arange(all_images.shape[0])
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * len(indices))
    val_end = train_end + int(val_ratio * len(indices))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Creating the train, val, and test sets
    X_train = all_images[train_indices]
    y_train = all_masks[train_indices]
    
    X_val = all_images[val_indices]
    y_val = all_masks[val_indices]
    
    X_test = all_images[test_indices]
    y_test = all_masks[test_indices]
    
    print(f"Split into {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
    
    return {
        'train': {'images': X_train, 'masks': y_train},
        'val': {'images': X_val, 'masks': y_val},
        'test': {'images': X_test, 'masks': y_test}
    }

# Process model training, validation and testing
def train_and_evaluate_model(hdf5_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Main function to train, validate and test the segmentation model using HDF5 datasets
    
    Args:
        hdf5_files: List of HDF5 file paths
        train_ratio, val_ratio, test_ratio: Ratios for splitting the data
    """
    # Load all datasets
    print("Loading all datasets...")
    all_data = load_all_hdf5_datasets(hdf5_files, train_ratio, val_ratio, test_ratio)
    
    # Extract train, val, and test sets
    X_train = all_data['train']['images']
    y_train = all_data['train']['masks']
    X_val = all_data['val']['images']
    y_val = all_data['val']['masks']
    X_test = all_data['test']['images']
    y_test = all_data['test']['masks']
    
    print(f"Dataset loaded: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
    print(f"Image shape: {X_train.shape[1:]}, Mask shape: {y_train.shape[1:]}")
    
    # Build and compile model
    input_shape = X_train.shape[1:]
    model = build_unet(input_size=input_shape, num_classes=12)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Show model summary
    model.summary()
    
    # Train the model
    print("Training the model...")
    history = model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5', 
                save_best_only=True, 
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Load the best model for evaluation
    model.load_weights('best_model.h5')
    
    # Evaluating on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Calculating IoU for each class
    print("Calculating IoU metrics...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=-1)
    y_true_classes = np.argmax(y_test, axis=-1)
    
    def calculate_iou(y_true, y_pred, class_id):
        true_mask = (y_true == class_id).flatten()
        pred_mask = (y_pred == class_id).flatten()
        
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        iou = intersection / union if union > 0 else 0
        return iou
    
    # Calculating mean IoU across all classes
    class_ious = []
    for class_id in range(12):
        class_iou = calculate_iou(y_true_classes, y_pred_classes, class_id)
        class_ious.append(class_iou)
        print(f"IoU for class {class_id}: {class_iou:.4f}")
    
    mean_iou = np.mean(class_ious)
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # Saving some test predictions
    print("Generating prediction visualizations...")
    os.makedirs('predictions', exist_ok=True)
    n_samples = min(10, len(X_test))
    for i in range(n_samples):
        pred = model.predict(np.expand_dims(X_test[i], axis=0))[0]
        pred_mask = np.argmax(pred, axis=-1)
        
        true_mask = np.argmax(y_test[i], axis=-1)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        plt.axis('off')
        
        plt.subplot(132)
        plt.title('True Mask')
        plt.imshow(true_mask, cmap='nipy_spectral')
        plt.axis('off')
        
        plt.subplot(133)
        plt.title('Predicted Mask')
        plt.imshow(pred_mask, cmap='nipy_spectral')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'predictions/test_example_{i}.png')
        plt.close()
    
    print("Evaluation complete!")
    return model, history, (test_loss, test_accuracy, mean_iou)

# Example usage
if __name__ == "__main__":
    # List all 5 dataset HDF5 files
    hdf5_files = [
        r"C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-08-23 TCR Phase 1 Build 1.hdf5",
        r"C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-09-15 TCR Phase 1 Build 2.hdf5",
        r"C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-10-01 TCR Phase 1 Build 3.hdf5",
        r"C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-10-19 TCR Phase 1 Build 4.hdf5",
        r"C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-11-12 TCR Phase 1 Build 5.hdf5"
    ]
    
    # Train and evaluate the model
    model, history, metrics = train_and_evaluate_model(hdf5_files)
    
    # Save the final model
    model.save('final_defect_segmentation_model.h5')
    
    # Print final results
    test_loss, test_accuracy, mean_iou = metrics
    print(f"\nFinal Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
