This repository implements a deep learning-based object detection system for identifying defects and monitoring the nozzle-workpiece interaction in 3D printing processes. By automatically detecting issues during the printing process, this system aims to improve print quality and reduce material waste.
Features

Real-time detection of 3D printing defects and nozzle-workpiece interaction
Support for multiple 3D printer configurations and materials
Detection model trained on a diverse dataset of 3D printing scenarios
Comprehensive evaluation metrics and visualizations

Dataset
The model is trained on the "3D printing pictures" dataset from Roboflow, which contains 458 annotated images of 3D printing processes with COCO format annotations:

Dataset Version: v6 (2022-11-06)
Number of Images: 458
Annotation Format: COCO
Classes: Nozzle-workpiece interactions and defects
Image Size: 416x416 (Fit within)

The dataset is available for download at Roboflow Universe.
Model Architecture
The system uses a state-of-the-art object detection architecture to identify defects and nozzle-workpiece interactions:

Backbone: ResNet50
Detection Framework: RetinaNet-inspired architecture
Input Size: 416x416
Output: Bounding box coordinates, objectness score, and class probabilities
