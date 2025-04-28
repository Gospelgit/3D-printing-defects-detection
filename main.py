import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

# Using U-net model CNN model for defect segmentation, defintion:
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

# Function to load training data
def load_dataset(dataset_dir, split_file='dataset_split.json'):
    import json
    
    with open(os.path.join(dataset_dir, split_file)) as f:
        splits = json.load(f)
    
    # Load training images and masks
    X_train, y_train = [], []
    for img_file in splits['train']:
        img_path = os.path.join(dataset_dir, 'images', img_file)
        mask_path = os.path.join(dataset_dir, 'masks', img_file)
        
        img = plt.imread(img_path)
        mask = plt.imread(mask_path)
        
        # Normalize image to [0,1]
        img = img.astype(np.float32) / 255.0
        
        # Ensure mask is properly formatted
        if len(mask.shape) > 2 and mask.shape[2] >= 3:
            mask = mask[:,:,0]  # Take first channel if it's RGB
        
        # Reshape for model input
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        
        # Convert mask to one-hot encoding (for multi-class segmentation)
        mask_onehot = tf.keras.utils.to_categorical(mask, num_classes=12)
        
        X_train.append(img)
        y_train.append(mask_onehot)
    
    return np.array(X_train), np.array(y_train)

# Train the model
def train_segmentation_model(dataset_dir):
    # Load data
    X_train, y_train = load_dataset(dataset_dir)
    
    # Build model
    model = build_unet(input_size=(X_train.shape[1], X_train.shape[2], 1), num_classes=12)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=50,
        validation_split=0.2,
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
            )
        ]
    )
    
    return model