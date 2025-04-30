"""
3D Printing Defect Detection - Evaluation Metrics

This module contains utilities for evaluating object detection models,
including mean Average Precision (mAP) and precision-recall curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: First box [x_min, y_min, x_max, y_max]
        box2: Second box [x_min, y_min, x_max, y_max]
        
    Returns:
        IoU value
    """
    # Calculate intersection
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    
    # Check if boxes intersect
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    # Calculate areas
    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union = area1 + area2 - intersection
    iou = intersection / union
    
    return iou


def calculate_precision_recall(detections, ground_truths, iou_threshold=0.5):
    """
    Calculate precision-recall curve for a class.
    
    Args:
        detections: List of detections, each as [box, score]
        ground_truths: List of ground truth boxes
        iou_threshold: IoU threshold for a positive detection
        
    Returns:
        precision, recall, average precision
    """
    # Sort detections by confidence score (descending)
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    
    # Initialize variables
    num_gt = len(ground_truths)
    num_detections = len(detections)
    
    # If no ground truths, return zeros
    if num_gt == 0:
        if num_detections == 0:
            return 1.0, 1.0, 1.0  # Perfect score for no detections and no ground truths
        else:
            return 0.0, 0.0, 0.0  # All detections are false positives
    
    # If no detections, return zeros
    if num_detections == 0:
        return 0.0, 0.0, 0.0
    
    # Array to keep track of which ground truths have been detected
    gt_detected = [False] * num_gt
    
    # Arrays for precision and recall
    precision = np.zeros(num_detections)
    recall = np.zeros(num_detections)
    
    # Number of true positives so far
    true_positives = 0
    
    # Process each detection
    for i, (box, _) in enumerate(detections):
        # Find the best ground truth match
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(ground_truths):
            if not gt_detected[j]:
                iou = calculate_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        # Check if this detection is a true positive
        if best_iou >= iou_threshold:
            gt_detected[best_gt_idx] = True
            true_positives += 1
        
        # Calculate precision and recall
        precision[i] = true_positives / (i + 1)
        recall[i] = true_positives / num_gt
    
    # Calculate average precision using all points
    # Add start and end points to precision/recall
    prec = np.concatenate(([1.0], precision, [0.0]))
    rec = np.concatenate(([0.0], recall, [1.0]))
    
    # Ensure precision decreases as recall increases for interpolation
    for i in range(prec.size - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    
    # Find indices where recall changes
    i = np.where(rec[1:] != rec[:-1])[0]
    
    # Calculate AP (area under precision-recall curve)
    ap = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])
    
    return precision, recall, ap


def calculate_map(all_detections, all_ground_truths, categories, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) across all classes.
    
    Args:
        all_detections: List of detections for each image
        all_ground_truths: List of ground truths for each image
        categories: Category information
        iou_threshold: IoU threshold for a positive detection
        
    Returns:
        mAP value, AP for each class, precision-recall curves
    """
    # Organize detections and ground truths by class
    detections_by_class = defaultdict(list)
    ground_truths_by_class = defaultdict(list)
    
    # Process each image
    for img_idx in range(len(all_detections)):
        detections = all_detections[img_idx]
        ground_truths = all_ground_truths[img_idx]
        
        # Get detection components
        if isinstance(detections, dict):
            boxes = detections['boxes']
            scores = detections['scores']
            classes = detections['classes']
        else:
            boxes, scores, classes = detections
        
        # Group by class
        for box, score, cls in zip(boxes, scores, classes):
            detections_by_class[int(cls)].append([box, score])
        
        # Process ground truths
        if isinstance(ground_truths, dict):
            gt_boxes = ground_truths['boxes']
            gt_classes = ground_truths['classes']
        else:
            gt_boxes, gt_classes = ground_truths
        
        # Group by class
        for box, cls in zip(gt_boxes, gt_classes):
            ground_truths_by_class[int(cls)].append(box)
    
    # Calculate AP for each class
    ap_per_class = {}
    precision_recall_curves = {}
    
    for cls in set(detections_by_class.keys()) | set(ground_truths_by_class.keys()):
        detections = detections_by_class[cls]
        ground_truths = ground_truths_by_class[cls]
        
        precision, recall, ap = calculate_precision_recall(
            detections, ground_truths, iou_threshold
        )
        
        ap_per_class[cls] = ap
        precision_recall_curves[cls] = (precision, recall)
    
    # Calculate mAP
    if ap_per_class:
        mean_ap = sum(ap_per_class.values()) / len(ap_per_class)
    else:
        mean_ap = 0.0
    
    return mean_ap, ap_per_class, precision_recall_curves


def plot_precision_recall_curves(precision_recall_curves, categories):
    """
    Plot precision-recall curves for each class.
    
    Args:
        precision_recall_curves: Dictionary of (precision, recall) tuples for each class
        categories: Category information
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    for cls, (precision, recall) in precision_recall_curves.items():
        # Get class name
        if isinstance(categories, list):
            # List of dicts with 'id' and 'name' keys
            cls_name = next((cat['name'] for cat in categories if cat['id'] == cls), f"Class {cls}")
        elif isinstance(categories, dict):
            # Dict mapping ID to name
            cls_name = categories.get(cls, f"Class {cls}")
        else:
            cls_name = f"Class {cls}"
        
        # Plot precision-recall curve
        ax.plot(recall, precision, linewidth=2, label=f"{cls_name} (AP: {sum(precision) / max(1, len(precision)):.3f})")
    
    # Set plot properties
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left')
    
    return fig


def calculate_f1_score(precision, recall):
    """
    Calculate F1 score given precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def plot_metrics_over_thresholds(all_detections, all_ground_truths, categories, 
                                iou_thresholds=None, conf_threshold=0.5):
    """
    Plot metrics (precision, recall, F1, mAP) over different IoU thresholds.
    
    Args:
        all_detections: List of detections for each image
        all_ground_truths: List of ground truths for each image
        categories: Category information
        iou_thresholds: List of IoU thresholds to evaluate
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Matplotlib figure
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    # Filter detections by confidence threshold
    filtered_detections = []
    for detections in all_detections:
        if isinstance(detections, dict):
            mask = detections['scores'] >= conf_threshold
            filtered_detections.append({
                'boxes': detections['boxes'][mask],
                'scores': detections['scores'][mask],
                'classes': detections['classes'][mask]
            })
        else:
            boxes, scores, classes = detections
            mask = scores >= conf_threshold
            filtered_detections.append([boxes[mask], scores[mask], classes[mask]])
    
    # Calculate metrics at each threshold
    map_values = []
    avg_precision = []
    avg_recall = []
    avg_f1 = []
    
    for iou_threshold in iou_thresholds:
        mean_ap, ap_per_class, _ = calculate_map(
            filtered_detections, all_ground_truths, categories, iou_threshold
        )
        
        map_values.append(mean_ap)
        
        # Calculate overall precision, recall, and F1
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for img_idx in range(len(filtered_detections)):
            detections = filtered_detections[img_idx]
            ground_truths = all_ground_truths[img_idx]
            
            # Get detection components
            if isinstance(detections, dict):
                det_boxes = detections['boxes']
                det_classes = detections['classes']
            else:
                det_boxes, _, det_classes = detections
            
            # Get ground truth components
            if isinstance(ground_truths, dict):
                gt_boxes = ground_truths['boxes']
                gt_classes = ground_truths['classes']
            else:
                gt_boxes, gt_classes = ground_truths
            
            # Calculate tp, fp, fn
            tp, fp, fn = calculate_tpfpfn(
                det_boxes, det_classes, gt_boxes, gt_classes, iou_threshold
            )
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Calculate precision, recall, F1
        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = 0.0
            
        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = 0.0
            
        f1 = calculate_f1_score(precision, recall)
        
        avg_precision.append(precision)
        avg_recall.append(recall)
        avg_f1.append(f1)
    
    # Plot metrics
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    ax.plot(iou_thresholds, map_values, 'o-', linewidth=2, label='mAP')
    ax.plot(iou_thresholds, avg_precision, 's-', linewidth=2, label='Precision')
    ax.plot(iou_thresholds, avg_recall, '^-', linewidth=2, label='Recall')
    ax.plot(iou_thresholds, avg_f1, 'D-', linewidth=2, label='F1 Score')
    
    # Set plot properties
    ax.set_xlabel('IoU Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Detection Metrics vs. IoU Threshold (Conf. Threshold = {conf_threshold})')
    ax.grid(True)
    ax.set_xlim([min(iou_thresholds), max(iou_thresholds)])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left')
    
    return fig


def calculate_tpfpfn(det_boxes, det_classes, gt_boxes, gt_classes, iou_threshold):
    """
    Calculate true positives, false positives, and false negatives.
    
    Args:
        det_boxes: Detection boxes
        det_classes: Detection classes
        gt_boxes: Ground truth boxes
        gt_classes: Ground truth classes
        iou_threshold: IoU threshold for a positive detection
        
    Returns:
        true_positives, false_positives, false_negatives
    """
    # Convert to numpy arrays if needed
    det_boxes = np.array(det_boxes)
    det_classes = np.array(det_classes)
    gt_boxes = np.array(gt_boxes)
    gt_classes = np.array(gt_classes)
    
    # Initialize counts
    true_positives = 0
    false_positives = 0
    
    # Array to keep track of which ground truths have been detected
    if len(gt_boxes) > 0:
        gt_detected = np.zeros(len(gt_boxes), dtype=bool)
    else:
        gt_detected = np.array([])
    
    # Process each detection
    for i in range(len(det_boxes)):
        det_box = det_boxes[i]
        det_class = det_classes[i]
        
        # Find the best ground truth match
        best_iou = 0
        best_gt_idx = -1
        
        for j in range(len(gt_boxes)):
            if not gt_detected[j] and det_class == gt_classes[j]:
                iou = calculate_iou(det_box, gt_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        # Check if this detection is a true positive
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            true_positives += 1
            gt_detected[best_gt_idx] = True
        else:
            false_positives += 1
    
    # Calculate false negatives
    false_negatives = len(gt_boxes) - true_positives
    
    return true_positives, false_positives, false_negatives


def calculate_confusion_matrix(all_detections, all_ground_truths, num_classes, conf_threshold=0.5, iou_threshold=0.5):
    """
    Calculate the confusion matrix for object detection.
    
    Args:
        all_detections: List of detections for each image
        all_ground_truths: List of ground truths for each image
        num_classes: Number of classes
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for a true positive
        
    Returns:
        Confusion matrix
    """
    # Initialize confusion matrix
    # Add one row/column for background class (false positives)
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    
    # Process each image
    for img_idx in range(len(all_detections)):
        detections = all_detections[img_idx]
        ground_truths = all_ground_truths[img_idx]
        
        # Get detection components
        if isinstance(detections, dict):
            det_boxes = detections['boxes']
            det_scores = detections['scores']
            det_classes = detections['classes']
        else:
            det_boxes, det_scores, det_classes = detections
        
        # Filter by confidence threshold
        confidence_mask = det_scores >= conf_threshold
        det_boxes = det_boxes[confidence_mask]
        det_classes = det_classes[confidence_mask]
        
        # Get ground truth components
        if isinstance(ground_truths, dict):
            gt_boxes = ground_truths['boxes']
            gt_classes = ground_truths['classes']
        else:
            gt_boxes, gt_classes = ground_truths
        
        # Track which ground truths have been matched
        if len(gt_boxes) > 0:
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        else:
            gt_matched = np.array([])
        
        # Process each detection
        for i in range(len(det_boxes)):
            det_box = det_boxes[i]
            det_class = int(det_classes[i])
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for j in range(len(gt_boxes)):
                if not gt_matched[j]:  # Only consider unmatched ground truths
                    iou = calculate_iou(det_box, gt_boxes[j])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            # If we found a match with sufficient IoU
            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                gt_class = int(gt_classes[best_gt_idx])
                confusion_matrix[gt_class, det_class] += 1
                gt_matched[best_gt_idx] = True
            else:
                # False positive (background class)
                confusion_matrix[num_classes, det_class] += 1
        
        # Add false negatives (ground truths that weren't matched)
        for j in range(len(gt_boxes)):
            if not gt_matched[j]:
                gt_class = int(gt_classes[j])
                # Detected as background
                confusion_matrix[gt_class, num_classes] += 1
    
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, categories, figsize=(10, 8)):
    """
    Plot a confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix to plot
        categories: Category information
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Get class names
    if isinstance(categories, list):
        # List of dicts with 'id' and 'name' keys
        class_names = [cat['name'] for cat in categories]
    elif isinstance(categories, dict):
        # Dict mapping ID to name
        class_names = [categories.get(i, f"Class {i}") for i in range(len(categories))]
    else:
        class_names = [f"Class {i}" for i in range(confusion_matrix.shape[0] - 1)]
    
    # Add background class
    class_names.append("Background")
    
    # Create figure
    fig, ax = plt.figure(figsize=figsize), plt.gca()
    
    # Plot confusion matrix
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add labels and ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, confusion_matrix[i, j],
                          ha="center", va="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
    
    # Set titles
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
    plt.tight_layout()
    
    return fig


def calculate_detection_metrics(all_detections, all_ground_truths, categories, 
                              conf_threshold=0.5, iou_threshold=0.5):
    """
    Calculate comprehensive detection metrics.
    
    Args:
        all_detections: List of detections for each image
        all_ground_truths: List of ground truths for each image
        categories: Category information
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for a true positive
        
    Returns:
        Dictionary of metrics
    """
    # Filter detections by confidence threshold
    filtered_detections = []
    for detections in all_detections:
        if isinstance(detections, dict):
            mask = detections['scores'] >= conf_threshold
            filtered_detections.append({
                'boxes': detections['boxes'][mask],
                'scores': detections['scores'][mask],
                'classes': detections['classes'][mask]
            })
        else:
            boxes, scores, classes = detections
            mask = scores >= conf_threshold
            filtered_detections.append([boxes[mask], scores[mask], classes[mask]])
    
    # Calculate mAP and AP per class
    mean_ap, ap_per_class, precision_recall_curves = calculate_map(
        filtered_detections, all_ground_truths, categories, iou_threshold
    )
    
    # Calculate overall precision, recall, and F1
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for img_idx in range(len(filtered_detections)):
        detections = filtered_detections[img_idx]
        ground_truths = all_ground_truths[img_idx]
        
        # Get detection components
        if isinstance(detections, dict):
            det_boxes = detections['boxes']
            det_classes = detections['classes']
        else:
            det_boxes, _, det_classes = detections
        
        # Get ground truth components
        if isinstance(ground_truths, dict):
            gt_boxes = ground_truths['boxes']
            gt_classes = ground_truths['classes']
        else:
            gt_boxes, gt_classes = ground_truths
        
        # Calculate tp, fp, fn
        tp, fp, fn = calculate_tpfpfn(
            det_boxes, det_classes, gt_boxes, gt_classes, iou_threshold
        )
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate precision, recall, F1
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
    else:
        precision = 0.0
        
    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
    else:
        recall = 0.0
        
    f1 = calculate_f1_score(precision, recall)
    
    # Calculate confusion matrix
    num_classes = len(categories) if isinstance(categories, list) else len(categories.keys())
    confusion_mat = calculate_confusion_matrix(
        filtered_detections, all_ground_truths, num_classes, conf_threshold, iou_threshold
    )
    
    # Compile all metrics
    metrics = {
        'mAP': mean_ap,
        'AP_per_class': ap_per_class,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'confusion_matrix': confusion_mat,
        'precision_recall_curves': precision_recall_curves
    }
    
    return metrics


def print_metrics_summary(metrics, categories):
    """
    Print a summary of detection metrics.
    
    Args:
        metrics: Metrics dictionary from calculate_detection_metrics
        categories: Category information
    """
    print("\n===== Detection Metrics Summary =====")
    print(f"mAP@0.5: {metrics['mAP']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    print("\nAP per class:")
    for cls, ap in metrics['AP_per_class'].items():
        # Get class name
        if isinstance(categories, list):
            # List of dicts with 'id' and 'name' keys
            cls_name = next((cat['name'] for cat in categories if cat['id'] == cls), f"Class {cls}")
        elif isinstance(categories, dict):
            # Dict mapping ID to name
            cls_name = categories.get(cls, f"Class {cls}")
        else:
            cls_name = f"Class {cls}"
        
        print(f"  {cls_name}: {ap:.4f}")
    
    print("=====================================\n")
