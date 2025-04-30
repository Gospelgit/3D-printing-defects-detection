
# Set TF log level to reduce warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import custom modules
try:
    from models.detection_model import DetectionModel
    from utils.data_loader import COCODataLoader
    from utils.metrics import calculate_detection_metrics, print_metrics_summary, plot_precision_recall_curves
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure the current directory or PYTHONPATH includes the project directory.")
    sys.path.append(os.getcwd())  # Add current directory to path
    from models.detection_model import DetectionModel
    from utils.data_loader import COCODataLoader
    from utils.metrics import calculate_detection_metrics, print_metrics_summary, plot_precision_recall_curves

# Set up argument parser
parser = argparse.ArgumentParser(description='Train 3D printing defect detection model')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset directory')
parser.add_argument('--annotation_file', type=str, default='_annotations.coco.json', help='COCO annotation filename')
parser.add_argument('--img_size', type=int, nargs=2, default=[416, 416], help='Image size (height, width)')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval (epochs)')
parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience (epochs)')
parser.add_argument('--num_classes', type=int, default=None, help='Number of classes (optional)')
parser.add_argument('--gpu', type=str, default=None, help='GPU device to use (e.g., "0" or "0,1")')
args = parser.parse_args()


def set_gpu_settings(gpu):
    """
    Configure GPU settings.
    
    Args:
        gpu: GPU device(s) to use
    """
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        # Set memory growth to avoid taking all GPU memory
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Using GPU(s): {gpu}")
        except Exception as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("Using default device configuration")


def create_training_callbacks(output_dir, early_stopping_patience=10):
    """
    Create training callbacks.
    
    Args:
        output_dir: Output directory for checkpoints and logs
        early_stopping_patience: Patience for early stopping
        
    Returns:
        List of callbacks
    """
    # Create checkpoint callback
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Use .weights.h5 extension when using save_weights_only=True
    checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}_loss_{loss:.4f}.weights.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    # Create tensorboard callback
    log_dir = os.path.join(output_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Create early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Create learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Create CSV logger
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(output_dir, 'training_log.csv'),
        append=True
    )
    
    return [checkpoint_callback, tensorboard_callback, early_stopping_callback, lr_scheduler, csv_logger]


def save_model_architecture(model, output_dir):
    """
    Save model architecture summary and diagram.
    
    Args:
        model: Keras model
        output_dir: Output directory
    """
    # Save model summary to file
    summary_path = os.path.join(output_dir, 'model_summary.txt')
    
    original_stdout = sys.stdout
    with open(summary_path, 'w') as f:
        sys.stdout = f
        model.summary()
        sys.stdout = original_stdout
    
    print(f"Model summary saved to {summary_path}")
    
    # Save model architecture diagram - wrapped in try/except for better error handling
    try:
        from tensorflow.keras.utils import plot_model
        
        plot_path = os.path.join(output_dir, 'model_architecture.png')
        plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        print(f"Model architecture diagram saved to {plot_path}")
    except Exception as e:
        print(f"Warning: Could not save model architecture diagram: {str(e)}")
        print("This is non-critical and will not affect model training.")


def save_training_config(args, output_dir):
    """
    Save training configuration.
    
    Args:
        args: Command line arguments
        output_dir: Output directory
    """
    # Convert args to dictionary
    config = {key: value for key, value in vars(args).items() if not key.startswith('_')}
    
    # Convert non-serializable types
    if isinstance(config.get('img_size'), tuple) or isinstance(config.get('img_size'), list):
        config['img_size'] = list(config['img_size'])
    
    # Add timestamp
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to file
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Training configuration saved to {config_path}")


def visualize_dataset_samples(data_loader, img_ids, output_dir, num_samples=5):
    """
    Visualize some samples from the dataset.
    
    Args:
        data_loader: COCODataLoader instance
        img_ids: List of image IDs
        output_dir: Output directory
        num_samples: Number of samples to visualize
    """
    samples_dir = os.path.join(output_dir, 'dataset_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Select random samples
    if len(img_ids) > num_samples:
        sample_ids = np.random.choice(img_ids, num_samples, replace=False)
    else:
        sample_ids = img_ids
    
    successful_samples = 0
    for i, img_id in enumerate(sample_ids):
        try:
            # Load image and annotations
            img, original_height, original_width, img_info = data_loader._load_image(img_id)
            if img is None:
                print(f"Warning: Failed to load image for sample {img_id}")
                continue

            annotations = data_loader._load_annotations(img_id)
            if annotations is None or len(annotations) == 0:
                print(f"Warning: No annotations found for sample {img_id}")
            
            # Create a visualization
            vis_img = (img * 255).astype(np.uint8).copy()
            
            # Draw bounding boxes
            for ann in annotations:
                # Skip annotations without bbox data
                if 'bbox' not in ann or 'category_id' not in ann:
                    continue
                
                # COCO format: [x, y, width, height]
                bbox = ann['bbox']
                category_id = ann['category_id']
                
                # Get class name and color
                if category_id in data_loader.cat_id_to_idx:
                    cat_idx = data_loader.cat_id_to_idx[category_id]
                    cat_name = data_loader.categories[cat_idx]['name']
                else:
                    cat_name = f"Unknown ({category_id})"
                
                # Generate color based on category ID
                np.random.seed(category_id)
                color = tuple(map(int, np.random.randint(0, 256, 3)))
                
                # Convert to pixel coordinates
                x, y, w, h = bbox
                x = int(x * vis_img.shape[1] / original_width)
                y = int(y * vis_img.shape[0] / original_height)
                w = int(w * vis_img.shape[1] / original_width)
                h = int(h * vis_img.shape[0] / original_height)
                
                # Draw rectangle
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                cv2.putText(vis_img, cat_name, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save the visualization
            output_path = os.path.join(samples_dir, f'sample_{i+1}.png')
            cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            successful_samples += 1
            
        except Exception as e:
            print(f"Warning: Could not visualize sample {img_id}: {str(e)}")
    
    if successful_samples > 0:
        print(f"Dataset samples saved to {samples_dir} ({successful_samples}/{len(sample_ids)} successful)")
    else:
        print("Warning: Failed to visualize any dataset samples")


def main():
    # Set GPU configuration
    set_gpu_settings(args.gpu)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Save training configuration
    save_training_config(args, args.output_dir)
    
    # Load data
    print(f"Loading dataset from {args.dataset_path}")
    try:
        data_loader = COCODataLoader(
            dataset_dir=args.dataset_path,
            annotation_file=args.annotation_file,
            img_size=tuple(args.img_size),
            augment_train=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Set number of classes
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = data_loader.num_classes
    
    print(f"Number of classes: {num_classes}")
    print(f"Categories: {[cat['name'] for cat in data_loader.categories]}")
    
    # Split dataset
    train_ids, val_ids, test_ids = data_loader.split_dataset(
        train_ratio=0.7, val_ratio=0.15, seed=42
    )
    
    # Visualize some dataset samples
    visualize_dataset_samples(data_loader, train_ids, args.output_dir)
    
    # Initialize model
    print("Building model...")
    try:
        model = DetectionModel(
            input_shape=(args.img_size[0], args.img_size[1], 3),
            num_classes=num_classes,
            backbone='resnet50',
            pretrained=args.pretrained
        )
    except Exception as e:
        print(f"Error building model: {e}")
        sys.exit(1)
    
    # Get anchors from the model - this is crucial to ensure the same anchors are used in both model and data loader
    print("Generating anchors...")
    anchors = model.get_anchors(args.img_size[0], args.img_size[1])
    print(f"Generated {len(anchors)} anchors")
    
    # Create datasets using the anchors from the model
    print("Creating datasets...")
    try:
        train_dataset = data_loader.create_tf_dataset(
            train_ids, batch_size=args.batch_size, is_training=True, anchors=anchors
        )
        
        val_dataset = data_loader.create_tf_dataset(
            val_ids, batch_size=args.batch_size, is_training=False, anchors=anchors
        )
    except Exception as e:
        print(f"Error creating datasets: {e}")
        sys.exit(1)
    
    # Compile model
    print("Compiling model...")
    model.compile_model(learning_rate=args.learning_rate)
    
    # Save model architecture
    save_model_architecture(model.model, args.output_dir)
    
    # Create callbacks
    callbacks = create_training_callbacks(
        args.output_dir, early_stopping_patience=args.early_stopping
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        try:
            model.load_weights(args.resume)
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Starting training from scratch")
    
    # Train model
    print("Starting training...")
    try:
        history = model.train(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save the current model state
        model.save_weights(os.path.join(args.output_dir, 'interrupted_weights.weights.h5'))
        print("Saved model weights at interruption point")
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    # Save training history
    try:
        history_dict = {key: [float(x) for x in values] for key, values in history.history.items()}
        history_path = os.path.join(args.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save training history: {e}")
    
    # Plot training history
    try:
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot (if available)
        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    except Exception as e:
        print(f"Warning: Could not plot training history: {e}")
    
    # Save final model
    try:
        model.save_model(os.path.join(args.output_dir, 'final_model.h5'))
        model.save_weights(os.path.join(args.output_dir, 'final_weights.weights.h5'))
        print("Model saved successfully")
    except Exception as e:
        print(f"Warning: Could not save final model: {e}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    try:
        # Use the same anchors for test dataset creation
        test_dataset = data_loader.create_tf_dataset(
            test_ids, batch_size=args.batch_size, is_training=False, anchors=anchors
        )
        
        # Load test images and ground truths
        test_images, test_boxes, test_classes = data_loader.load_data(test_ids)
        
        if len(test_images) == 0:
            print("Warning: No test images found. Skipping evaluation.")
            return
        
        # Make predictions on test set
        print("Making predictions on test set...")
        test_predictions = []
        for i in tqdm(range(len(test_images))):
            # Make prediction
            detections = model.predict(
                np.expand_dims(test_images[i], axis=0),
                confidence_threshold=0.5,
                nms_threshold=0.5
            )[0]
            
            test_predictions.append(detections)
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_detection_metrics(
            test_predictions,
            list(zip(test_boxes, test_classes)),
            data_loader.categories,
            conf_threshold=0.5,
            iou_threshold=0.5
        )
        
        # Print metrics summary
        print_metrics_summary(metrics, data_loader.categories)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_json = {k: v for k, v in metrics.items() if k != 'precision_recall_curves' and k != 'confusion_matrix'}
            if 'confusion_matrix' in metrics:
                metrics_json['confusion_matrix'] = metrics['confusion_matrix'].tolist()
            json.dump(metrics_json, f, indent=4)
        
        # Plot precision-recall curves
        pr_fig = plot_precision_recall_curves(metrics['precision_recall_curves'], data_loader.categories)
        pr_fig.savefig(os.path.join(args.output_dir, 'precision_recall_curves.png'))
        
        # Save some example detections
        examples_dir = os.path.join(args.output_dir, 'test_examples')
        os.makedirs(examples_dir, exist_ok=True)
        
        print("Saving example detections...")
        num_examples = min(10, len(test_images))
        for i in range(num_examples):
            try:
                # Create visualization
                vis_img = data_loader.visualize_detections(
                    test_images[i], test_predictions[i], score_threshold=0.5
                )
                
                # Save visualization
                output_path = os.path.join(examples_dir, f'detection_{i+1}.png')
                cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Warning: Could not save detection example {i+1}: {e}")
        
        print(f"Example detections saved to {examples_dir}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    print("\nTraining and evaluation complete!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
