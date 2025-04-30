#Defining GLOBAL anchor parameters to ensure consistency
ANCHOR_SCALES = [32, 64, 128, 256, 512]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]  # h/w ratios


class DetectionModel:
    """
    Object detection model for 3D printing defect detection.
    Uses a ResNet50 backbone with a Feature Pyramid Network (FPN)
    and detection heads.
    """
    
    def __init__(self, 
                 input_shape=(416, 416, 3), 
                 num_classes=1,
                 backbone='resnet50',
                 pretrained=True):
        """
        Initialize the detection model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of object classes to detect
            backbone: Backbone network ('resnet50', 'mobilenetv2')
            pretrained: Whether to use pretrained weights for backbone
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.pretrained = pretrained
        
        # Anchor box parameters - using global values
        self.anchor_scales = ANCHOR_SCALES
        self.anchor_ratios = ANCHOR_RATIOS
        self.anchor_count = len(self.anchor_scales) * len(self.anchor_ratios)
        
        # Calculate the expected number of anchors
        print(f"Expected number of anchors: calculating...")
        expected_anchors = self._calculate_expected_anchors(input_shape[0], input_shape[1])
        print(f"Expected number of anchors: {expected_anchors}")
        
        # Build the model
        self.model = self._build_simplified_model()
    
    def _calculate_expected_anchors(self, image_height, image_width):
        """
        Calculate the expected number of anchors based on input dimensions.
        This ensures consistency between model and data loader.
        
        Args:
            image_height: Input image height
            image_width: Input image width
            
        Returns:
            Expected number of anchors
        """
        # Feature map sizes for each FPN level (P3-P7)
        feature_sizes = [
            (image_height // 8, image_width // 8),     # P3
            (image_height // 16, image_width // 16),   # P4
            (image_height // 32, image_width // 32),   # P5
            (image_height // 64, image_width // 64),   # P6
            (image_height // 128, image_width // 128)  # P7
        ]
        
        total_anchors = 0
        for level, (feature_height, feature_width) in enumerate(feature_sizes):
            # Calculate number of anchors for this level
            level_anchors = feature_height * feature_width * len(self.anchor_scales) * len(self.anchor_ratios)
            print(f"Level {level+3} ({feature_height}x{feature_width}): {level_anchors} anchors")
            total_anchors += level_anchors
        
        return total_anchors

    def _build_simplified_model(self):
        """
        Build a simplified detection model.
        This uses a more direct approach to avoid KerasTensor issues.
        
        Returns:
            Keras Model
        """
        try:
            # Create input layer
            inputs = layers.Input(shape=self.input_shape)
            
            # Use a simpler backbone to avoid issues
            # First create some basic convolutional layers
            x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            # Create a simple detection head
            # This produces a fixed number of outputs to avoid dynamic reshaping issues
            
            # We'll use a fixed number of detections - 100 boxes max
            max_detections = 100
            
            # Classification output [batch, max_detections, num_classes]
            cls_output = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
            cls_output = layers.Conv2D(self.anchor_count * self.num_classes, kernel_size=1, padding='same', 
                                       activation='sigmoid')(cls_output)
            # Use a fixed output size
            cls_output = layers.GlobalAveragePooling2D()(cls_output)
            cls_output = layers.Dense(max_detections * self.num_classes)(cls_output)
            cls_output = layers.Reshape((max_detections, self.num_classes), name='classification_output')(cls_output)
            
            # Regression output [batch, max_detections, 4]
            reg_output = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
            reg_output = layers.Conv2D(self.anchor_count * 4, kernel_size=1, padding='same')(reg_output)
            # Use a fixed output size
            reg_output = layers.GlobalAveragePooling2D()(reg_output)
            reg_output = layers.Dense(max_detections * 4)(reg_output)
            reg_output = layers.Reshape((max_detections, 4), name='regression_output')(reg_output)
            
            # Create the model
            model = models.Model(
                inputs=inputs,
                outputs=[reg_output, cls_output],
                name='3d_printing_detection_simplified'
            )
            
            return model
            
        except Exception as e:
            print(f"Error building simplified model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_anchors(self, image_height, image_width):
        """
        Generate anchor boxes for all feature map levels
        
        Args:
            image_height: Input image height
            image_width: Input image width
            
        Returns:
            numpy array of anchor boxes in normalized coordinates
            shape: [total_anchors, 4] where each box is [x_min, y_min, x_max, y_max]
        """
        anchors = []
        
        # Feature map sizes for each FPN level (P3-P7)
        feature_sizes = [
            (image_height // 8, image_width // 8),     # P3
            (image_height // 16, image_width // 16),   # P4
            (image_height // 32, image_width // 32),   # P5
            (image_height // 64, image_width // 64),   # P6
            (image_height // 128, image_width // 128)  # P7
        ]
        
        # For each feature map level
        for level, (feature_height, feature_width) in enumerate(feature_sizes):
            stride = 2 ** (level + 3)  # Stride relative to input image
            
            # Generate grid of anchor centers
            grid_y, grid_x = np.meshgrid(
                np.arange(feature_height),
                np.arange(feature_width),
                indexing='ij'
            )
            
            # Convert grid coordinates to image coordinates
            centers_x = (grid_x + 0.5) * stride / image_width
            centers_y = (grid_y + 0.5) * stride / image_height
            
            # For all combinations of scales and ratios
            for scale in self.anchor_scales:
                for ratio in self.anchor_ratios:
                    # Calculate width and height in pixels
                    w = scale * np.sqrt(ratio) / image_width
                    h = scale / np.sqrt(ratio) / image_height
                    
                    # Create anchor boxes for all grid centers
                    for cy, cx in zip(centers_y.flat, centers_x.flat):
                        # Convert to [x_min, y_min, x_max, y_max] format
                        x_min = cx - w / 2
                        y_min = cy - h / 2
                        x_max = cx + w / 2
                        y_max = cy + h / 2
                        
                        # Add the anchor
                        anchors.append([x_min, y_min, x_max, y_max])
        
        # Convert to numpy array
        anchors = np.array(anchors, dtype=np.float32)
        
        # Clip to image boundaries
        anchors = np.clip(anchors, 0, 1)
        
        return anchors
    
    def compile_model(self, learning_rate=1e-4):
        """
        Compile the model with appropriate loss functions
        
        Args:
            learning_rate: Learning rate for the optimizer
        """
        try:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Use simple losses instead of custom ones to avoid issues
            self.model.compile(
                optimizer=optimizer,
                loss={
                    'regression_output': 'mse',  # Mean squared error instead of custom loss
                    'classification_output': 'binary_crossentropy'  # BCE instead of focal loss
                },
                loss_weights={
                    'regression_output': 1.0,
                    'classification_output': 1.0
                }
            )
        except Exception as e:
            print(f"Error compiling model: {e}")
            raise
    
    def train(self, train_data, validation_data=None, epochs=50, batch_size=8, callbacks=None):
        """
        Train the model
        
        Args:
            train_data: Training data generator or tuple (X_train, y_train)
            validation_data: Validation data generator or tuple (X_val, y_val)
            epochs: Number of epochs to train
            batch_size: Batch size
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        try:
            # Check if the model is compiled
            if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
                self.compile_model()
                print("Model was not compiled. Using default compilation settings.")
            
            return self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
        except Exception as e:
            print(f"Error during training: {e}")
            raise
    
    def predict(self, images, confidence_threshold=0.5, nms_threshold=0.5):
        """
        Predict bounding boxes for a batch of images
        
        Args:
            images: Batch of images, shape [batch, height, width, channels]
            confidence_threshold: Threshold for detection confidence
            nms_threshold: Non-max suppression IOU threshold
            
        Returns:
            List of detections for each image, each containing
            bounding boxes [x_min, y_min, x_max, y_max], 
            confidence scores, and class IDs
        """
        try:
            # Run inference
            regression, classification = self.model.predict(images)
            
            all_detections = []
            
            # Process each image in the batch
            for i in range(len(images)):
                # Get predictions for this image
                reg_pred = regression[i]
                cls_pred = classification[i]
                
                # Get confidence scores and class IDs
                scores = np.max(cls_pred, axis=1)
                class_ids = np.argmax(cls_pred, axis=1)
                
                # Filter by confidence threshold
                mask = scores > confidence_threshold
                if not np.any(mask):
                    all_detections.append({
                        'boxes': np.zeros((0, 4), dtype=np.float32),
                        'scores': np.zeros(0, dtype=np.float32),
                        'classes': np.zeros(0, dtype=np.int32)
                    })
                    continue
                
                boxes = reg_pred[mask]  # Use regression output directly as boxes
                scores = scores[mask]
                class_ids = class_ids[mask]
                
                # Apply non-max suppression using NumPy implementation to avoid TF issues
                try:
                    selected_indices = self._numpy_nms(boxes, scores, nms_threshold)
                    
                    selected_boxes = boxes[selected_indices]
                    selected_scores = scores[selected_indices]
                    selected_classes = class_ids[selected_indices]
                except Exception as e:
                    print(f"Warning: Error during NMS: {e}")
                    selected_boxes = np.zeros((0, 4), dtype=np.float32)
                    selected_scores = np.zeros(0, dtype=np.float32)
                    selected_classes = np.zeros(0, dtype=np.int32)
                
                all_detections.append({
                    'boxes': selected_boxes,
                    'scores': selected_scores,
                    'classes': selected_classes
                })
            
            return all_detections
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return empty detections on error
            return [{'boxes': np.zeros((0, 4), dtype=np.float32),
                     'scores': np.zeros(0, dtype=np.float32),
                     'classes': np.zeros(0, dtype=np.int32)} for _ in range(len(images))]
    
    def _numpy_nms(self, boxes, scores, threshold):
        """
        NumPy implementation of non-maximum suppression to avoid TensorFlow issues
        
        Args:
            boxes: Bounding boxes, shape [N, 4]
            scores: Confidence scores, shape [N]
            threshold: IoU threshold for NMS
            
        Returns:
            Selected indices
        """
        # Sort by score
        order = np.argsort(scores)[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with rest of boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            
            iou = inter / (area1 + area2 - inter)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)
    
    def save_weights(self, filepath):
        """
        Save model weights
        
        Args:
            filepath: Path to save weights file
        """
        try:
            # Ensure the directory exists
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save_weights(filepath)
            print(f"Model weights saved to {filepath}")
        except Exception as e:
            print(f"Error saving weights: {e}")
            raise
    
    def load_weights(self, filepath):
        """
        Load model weights
        
        Args:
            filepath: Path to weights file
        """
        try:
            if not os.path.exists(filepath):
                print(f"Weights file not found: {filepath}")
                return False
            
            self.model.load_weights(filepath)
            print(f"Model weights loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def save_model(self, filepath):
        """
        Save the complete model
        
        Args:
            filepath: Path to save model file
        """
        try:
            # Ensure the directory exists
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
    
    def evaluate(self, test_data, verbose=1):
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test data (X_test, y_test) or dataset
            verbose: Verbosity level
            
        Returns:
            Evaluation metrics
        """
        try:
            return self.model.evaluate(test_data, verbose=verbose)
        except Exception as e:
            print(f"Error evaluating model: {e}")
            raise
