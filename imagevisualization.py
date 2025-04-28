import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os
import glob
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

class MultiPeregrineVisualizer:
    """
    A class to visualize and compare defects across multiple Peregrine HDF5 files.
    """
    
    def __init__(self, hdf5_files, output_dir="./defect_visualizations"):
        """
        Initialize the multi-build defect visualizer.
        
        Args:
            hdf5_files: List of paths to the Peregrine HDF5 files
            output_dir: Directory to save visualizations
        """
        if isinstance(hdf5_files, str):
            # If a string is provided, treat it as a pattern to glob
            self.hdf5_files = sorted(glob.glob(hdf5_files))
        else:
            # Otherwise, assume it's a list of file paths
            self.hdf5_files = hdf5_files
        
        self.output_dir = output_dir
        self.build_names = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract build names and validate files
        self.validate_files()
        
        # Anomaly class mappings - based on the documentation
        self.anomaly_classes = {
            0: "Powder",          # Not a defect, normal powder
            1: "Printed",         # Not a defect, normal print
            2: "Recoater_Hopping",
            3: "Recoater_Streaking",
            4: "Incomplete_Spreading",
            5: "Swelling",
            6: "Debris",
            7: "Super_Elevation",
            8: "Spatter",
            9: "Misprint",
            10: "Over_Melting",
            11: "Under_Melting"
        }
        
        # List defect classes (excluding normal conditions)
        self.defect_class_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        
        # Define colors for each defect type - using distinctive colors
        self.defect_colors = {
            2: 'red',               # Recoater Hopping
            3: 'orange',            # Recoater Streaking
            4: 'yellow',            # Incomplete Spreading
            5: 'lime',              # Swelling
            6: 'cyan',              # Debris
            7: 'dodgerblue',        # Super Elevation
            8: 'blue',              # Spatter
            9: 'magenta',           # Misprint
            10: 'purple',           # Over Melting
            11: 'brown'             # Under Melting
        }
    
    def validate_files(self):
        """
        Validate HDF5 files and extract build names.
        """
        valid_files = []
        build_names = []
        
        for i, file_path in enumerate(self.hdf5_files):
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'slices/camera_data/visible/0' in f:
                        valid_files.append(file_path)
                        
                        # Try to get build name
                        if 'core/build_name' in f.attrs:
                            build_name = f.attrs['core/build_name']
                            if isinstance(build_name, bytes):
                                build_name = build_name.decode('utf-8')
                        else:
                            # Use filename if build name not available
                            build_name = f"Build_{Path(file_path).stem}"
                        
                        build_names.append(build_name)
                    else:
                        print(f"Warning: File {file_path} does not contain expected image data.")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
        
        if not valid_files:
            raise ValueError("No valid HDF5 files found.")
        
        self.hdf5_files = valid_files
        self.build_names = build_names
        
        print(f"Found {len(valid_files)} valid HDF5 files:")
        for i, (file_path, build_name) in enumerate(zip(valid_files, build_names)):
            print(f"  {i+1}. {build_name}: {file_path}")
    
    def get_dataset_summary(self):
        """
        Display summary information about all datasets.
        
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for i, (file_path, build_name) in enumerate(zip(self.hdf5_files, self.build_names)):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Get basic metrics
                    total_layers = f['slices/camera_data/visible/0'].shape[0]
                    image_height, image_width = f['slices/camera_data/visible/0'].shape[1:3]
                    
                    # Get process parameters if available
                    parameter_sets = set()
                    if 'parts/process_parameters/parameter_set' in f:
                        for param_set in f['parts/process_parameters/parameter_set']:
                            if isinstance(param_set, bytes):
                                param_set = param_set.decode('utf-8')
                            parameter_sets.add(param_set)
                    
                    # Count number of parts and samples
                    num_parts = 0
                    if 'parts/process_parameters/parameter_set' in f:
                        num_parts = len(f['parts/process_parameters/parameter_set'])
                    
                    num_samples = 0
                    if 'samples/test_results/ultimate_tensile_strength' in f:
                        uts_data = f['samples/test_results/ultimate_tensile_strength'][...]
                        num_samples = np.sum(uts_data > 0)  # Count non-zero values
                    
                    # Count defects
                    defect_counts = {}
                    total_defect_pixels = 0
                    
                    for class_id in self.defect_class_ids:
                        defect_counts[class_id] = 0
                        if f'slices/segmentation_results/{class_id}' in f:
                            # Sample a subset of layers for efficiency
                            sample_step = max(1, total_layers // 100)  # Sample at most 100 layers
                            for layer in range(0, total_layers, sample_step):
                                defect_mask = f[f'slices/segmentation_results/{class_id}'][layer, ...]
                                if np.any(defect_mask):
                                    defect_counts[class_id] += 1
                                total_defect_pixels += np.sum(defect_mask)
                    
                    # Get material info
                    material_name = "Unknown"
                    if 'material/name' in f.attrs:
                        material_name = f.attrs['material/name']
                        if isinstance(material_name, bytes):
                            material_name = material_name.decode('utf-8')
                    
                    # Compile summary
                    summary_data.append({
                        'Build Name': build_name,
                        'File Path': file_path,
                        'Total Layers': total_layers,
                        'Image Dimensions': f"{image_width}x{image_height}",
                        'Material': material_name,
                        'Parameter Sets': ', '.join(parameter_sets) if parameter_sets else "Unknown",
                        'Number of Parts': num_parts,
                        'Number of Samples': num_samples,
                        'Total Defect Count': sum(defect_counts.values()),
                        'Total Defect Pixels': total_defect_pixels,
                        'Defect Types Present': sum(1 for c in defect_counts.values() if c > 0)
                    })
                    
                    # Add individual defect counts
                    for class_id in self.defect_class_ids:
                        defect_name = self.anomaly_classes[class_id].replace('_', ' ')
                        summary_data[-1][defect_name] = defect_counts[class_id]
                    
            except Exception as e:
                print(f"Error analyzing {build_name}: {str(e)}")
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Display summary
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nDataset Summary:")
        print(summary_df)
        
        # Save to CSV
        summary_csv = os.path.join(self.output_dir, "dataset_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary saved to {summary_csv}")
        
        return summary_df

    def visualize_defect_comparison(self):
        """
        Create visualizations comparing defect distributions across builds.
        """
        # First collect defect statistics
        defect_stats = []
        
        for i, (file_path, build_name) in enumerate(zip(self.hdf5_files, self.build_names)):
            try:
                with h5py.File(file_path, 'r') as f:
                    total_layers = f['slices/camera_data/visible/0'].shape[0]
                    
                    # Count defects in sampled layers
                    sample_step = max(1, total_layers // 100)  # Sample at most 100 layers
                    
                    for layer in range(0, total_layers, sample_step):
                        layer_stats = {'Build': build_name, 'Layer': layer}
                        
                        for class_id in self.defect_class_ids:
                            if f'slices/segmentation_results/{class_id}' in f:
                                defect_mask = f[f'slices/segmentation_results/{class_id}'][layer, ...]
                                defect_area = np.sum(defect_mask > 0)
                                layer_stats[self.anomaly_classes[class_id]] = defect_area
                            else:
                                layer_stats[self.anomaly_classes[class_id]] = 0
                        
                        defect_stats.append(layer_stats)
            
            except Exception as e:
                print(f"Error processing {build_name}: {str(e)}")
        
        # Convert to DataFrame
        defect_df = pd.DataFrame(defect_stats)
        
        # Create comparison visualizations
        
        # 1. Bar chart comparing total defect prevalence across builds
        plt.figure(figsize=(14, 8))
        
        # Reshape data for plotting
        plot_data = []
        for build in defect_df['Build'].unique():
            build_data = defect_df[defect_df['Build'] == build]
            
            for defect_type in [self.anomaly_classes[id] for id in self.defect_class_ids]:
                if defect_type in build_data:
                    # Calculate percentage of layers with this defect
                    total_layers = len(build_data)
                    layers_with_defect = np.sum(build_data[defect_type] > 0)
                    percentage = (layers_with_defect / total_layers) * 100
                    
                    plot_data.append({
                        'Build': build,
                        'Defect Type': defect_type.replace('_', ' '),
                        'Percentage of Layers': percentage
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        ax = sns.barplot(x='Defect Type', y='Percentage of Layers', hue='Build', data=plot_df)
        plt.title('Defect Prevalence Comparison Across Builds', fontsize=16)
        plt.xlabel('Defect Type', fontsize=12)
        plt.ylabel('Percentage of Layers with Defect', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Build', loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'defect_prevalence_comparison.png'), dpi=150)
        plt.close()
        
        # 2. Heatmap of defect prevalence across builds
        defect_pivot = pd.pivot_table(
            plot_df, 
            values='Percentage of Layers', 
            index='Build', 
            columns='Defect Type'
        )
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(defect_pivot, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Percentage of Layers'})
        plt.title('Defect Prevalence Heatmap Across Builds', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'defect_heatmap.png'), dpi=150)
        plt.close()
        
        # 3. Create a correlation matrix between defects
        defect_types = [self.anomaly_classes[id] for id in self.defect_class_ids]
        corr_df = defect_df[defect_types].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   mask=mask, fmt='.2f')
        plt.title('Correlation Between Different Defect Types', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'defect_correlation.png'), dpi=150)
        plt.close()
        
        print("Defect comparison visualizations created successfully.")

    def find_similar_layers(self, num_layers=5, min_defect_area=100):
        """
        Find layers with similar defect patterns across different builds.
        
        Args:
            num_layers: Number of similar layer sets to find
            min_defect_area: Minimum area for a defect to be considered
            
        Returns:
            List of similar layer sets
        """
        # First, extract defect signatures for each build
        build_signatures = []
        
        for i, (file_path, build_name) in enumerate(zip(self.hdf5_files, self.build_names)):
            try:
                with h5py.File(file_path, 'r') as f:
                    total_layers = f['slices/camera_data/visible/0'].shape[0]
                    
                    # Create a defect signature for each layer
                    signatures = []
                    
                    for layer in tqdm(range(total_layers), 
                                     desc=f"Processing {build_name}", 
                                     leave=False):
                        # Create a binary vector indicating which defects are present
                        signature = np.zeros(len(self.defect_class_ids))
                        
                        for j, class_id in enumerate(self.defect_class_ids):
                            if f'slices/segmentation_results/{class_id}' in f:
                                defect_mask = f[f'slices/segmentation_results/{class_id}'][layer, ...]
                                if np.sum(defect_mask) > min_defect_area:
                                    signature[j] = 1
                        
                        signatures.append({
                            'build_idx': i,
                            'build_name': build_name,
                            'layer': layer,
                            'signature': signature,
                            'defect_count': np.sum(signature)
                        })
                    
                    build_signatures.append(signatures)
            
            except Exception as e:
                print(f"Error processing {build_name}: {str(e)}")
        
        # Find similar layers across builds
        similar_sets = []
        
        # Only proceed if we have at least 2 builds
        if len(build_signatures) < 2:
            print("Need at least 2 builds to find similar layers.")
            return similar_sets
        
        # Filter to layers with at least one defect
        defect_layers = []
        for build_idx, signatures in enumerate(build_signatures):
            defect_layers.extend([s for s in signatures if s['defect_count'] > 0])
        
        # Sort by number of defects (most interesting first)
        defect_layers.sort(key=lambda x: x['defect_count'], reverse=True)
        
        # Find similar layers
        processed_layers = set()
        
        for base_layer in tqdm(defect_layers, desc="Finding similar layers"):
            # Skip if already processed
            key = (base_layer['build_idx'], base_layer['layer'])
            if key in processed_layers:
                continue
            
            # Find similar layers in other builds
            similar_layers = [base_layer]
            
            for build_idx, signatures in enumerate(build_signatures):
                if build_idx == base_layer['build_idx']:
                    continue  # Skip same build
                
                # Find most similar layer in this build
                best_match = None
                best_similarity = -1
                
                for layer_info in signatures:
                    if (build_idx, layer_info['layer']) in processed_layers:
                        continue  # Skip processed layers
                    
                    # Calculate similarity (dot product of signatures)
                    similarity = np.dot(base_layer['signature'], layer_info['signature'])
                    
                    # Only consider layers with at least one matching defect
                    if similarity > 0 and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = layer_info
                
                if best_match:
                    similar_layers.append(best_match)
            
            # Only add if we found similar layers in other builds
            if len(similar_layers) > 1:
                similar_sets.append(similar_layers)
                
                # Mark as processed
                for layer in similar_layers:
                    processed_layers.add((layer['build_idx'], layer['layer']))
            
            # Stop once we have enough sets
            if len(similar_sets) >= num_layers:
                break
        
        # Visualize similar layers
        self.visualize_similar_layers(similar_sets)
        
        return similar_sets

    def visualize_similar_layers(self, similar_sets):
        """
        Create visualizations of similar layers across builds.
        
        Args:
            similar_sets: List of sets of similar layers
        """
        if not similar_sets:
            print("No similar layers found.")
            return
        
        # Create a directory for similar layer visualizations
        similar_dir = os.path.join(self.output_dir, "similar_layers")
        os.makedirs(similar_dir, exist_ok=True)
        
        # Visualize each set of similar layers
        for i, layer_set in enumerate(similar_sets):
            # Create a figure with a row for each build
            n_builds = len(layer_set)
            fig = plt.figure(figsize=(15, 5 * n_builds))
            gs = gridspec.GridSpec(n_builds, 3, width_ratios=[1, 1, 1])
            
            # Track which defects are present for the legend
            all_defects_present = set()
            
            for j, layer_info in enumerate(layer_set):
                build_idx = layer_info['build_idx']
                layer = layer_info['layer']
                build_name = layer_info['build_name']
                
                # Load image and masks
                with h5py.File(self.hdf5_files[build_idx], 'r') as f:
                    # Get the image
                    image = f['slices/camera_data/visible/0'][layer, ...]
                    
                    # Normalize image
                    img_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
                    
                    # Original image
                    ax1 = plt.subplot(gs[j, 0])
                    ax1.imshow(image, cmap='gray')
                    ax1.set_title(f"{build_name} - Layer {layer}")
                    ax1.axis('off')
                    
                    # Create defect mask
                    all_defects = np.zeros_like(image, dtype=np.uint8)
                    defects_present = []
                    
                    for class_id in self.defect_class_ids:
                        if f'slices/segmentation_results/{class_id}' in f:
                            defect_mask = f[f'slices/segmentation_results/{class_id}'][layer, ...]
                            if np.any(defect_mask):
                                all_defects[defect_mask > 0] = class_id
                                defects_present.append(class_id)
                                all_defects_present.add(class_id)
                    
                    # Defect mask
                    ax2 = plt.subplot(gs[j, 1])
                    
                    # Create a custom colormap for defects
                    colors = ['black']  # Background
                    for k in range(1, max(self.defect_class_ids) + 1):
                        if k in self.defect_colors:
                            colors.append(self.defect_colors[k])
                        else:
                            colors.append('gray')
                    
                    defect_cmap = ListedColormap(colors)
                    ax2.imshow(all_defects, cmap=defect_cmap, vmin=0, 
                             vmax=max(self.defect_class_ids))
                    ax2.set_title(f"Defect Mask")
                    ax2.axis('off')
                    
                    # Overlay
                    ax3 = plt.subplot(gs[j, 2])
                    
                    # Create RGB image
                    rgb_img = np.stack([img_norm, img_norm, img_norm], axis=-1)
                    
                    # Add colored overlay for each defect
                    for class_id in defects_present:
                        defect_mask = f[f'slices/segmentation_results/{class_id}'][layer, ...]
                        
                        # Get color for this defect
                        color = self.defect_colors.get(class_id, 'white')
                        from matplotlib.colors import to_rgb
                        r, g, b = to_rgb(color)
                        
                        # Create colored defect overlay with alpha blending
                        alpha = 0.7  # Transparency
                        mask_3d = np.stack([defect_mask, defect_mask, defect_mask], axis=-1)
                        color_overlay = np.zeros_like(rgb_img)
                        color_overlay[mask_3d > 0] = [r, g, b]
                        
                        # Blend with original image
                        rgb_img = np.where(mask_3d > 0, 
                                          (1-alpha)*rgb_img + alpha*color_overlay, 
                                          rgb_img)
                    
                    ax3.imshow(rgb_img)
                    ax3.set_title(f"Defect Overlay")
                    ax3.axis('off')
            
            # Add a shared legend for all defects
            legend_elements = []
            for class_id in sorted(all_defects_present):
                legend_elements.append(
                    Patch(facecolor=self.defect_colors[class_id], 
                          label=self.anomaly_classes[class_id].replace('_', ' '))
                )
            
            if legend_elements:
                fig.legend(handles=legend_elements, loc='lower center', 
                          bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(legend_elements)))
            
            # Add a title for the entire set
            defect_names = [self.anomaly_classes[int(id)].replace('_', ' ') 
                           for id in layer_set[0]['signature'] 
                           if id > 0]
            
            plt.suptitle(f"Similar Defect Pattern Set #{i+1}\nDefects: {', '.join(defect_names)}", 
                       fontsize=16, y=0.98)
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(similar_dir, f"similar_set_{i+1}.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualized {len(similar_sets)} sets of similar layers to {similar_dir}")

    def create_defect_gallery(self, defect_class_id=None):
        """
        Create a gallery of defect examples from all builds.
        
        Args:
            defect_class_id: Specific defect class to visualize, or None for all
        """
        # Create directory for galleries
        gallery_dir = os.path.join(self.output_dir, "defect_galleries")
        os.makedirs(gallery_dir, exist_ok=True)
        
        # If no specific defect requested, create galleries for all defects
        if defect_class_id is None:
            for class_id in self.defect_class_ids:
                self.create_defect_gallery(class_id)
            return
        
        # Find examples of this defect type across all builds
        examples = []
        
        for build_idx, (file_path, build_name) in enumerate(zip(self.hdf5_files, self.build_names)):
            try:
                with h5py.File(file_path, 'r') as f:
                    total_layers = f['slices/camera_data/visible/0'].shape[0]
                    
                    # Sample layers
                    sample_step = max(1, total_layers // 50)  # Sample at most 50 layers
                    
                    for layer in range(0, total_layers, sample_step):
                        if f'slices/segmentation_results/{defect_class_id}' in f:
                            defect_mask = f[f'slices/segmentation_results/{defect_class_id}'][layer, ...]
                            defect_area = np.sum(defect_mask)
                            
                            if defect_area > 200:  # Only consider significant defects
                                # Rank examples by area
                                examples.append({
                                    'build_idx': build_idx,
                                    'build_name': build_name,
                                    'layer': layer,
                                    'area': defect_area
                                })
            
            except Exception as e:
                print(f"Error processing {build_name}: {str(e)}")
        
        # Sort by defect area (largest first)
        examples.sort(key=lambda x: x['area'], reverse=True)
        
        # Keep only the top examples
        max_examples = min(20, len(examples))
        examples = examples[:max_examples]
        
        if not examples:
            print(f"No examples found for defect type: {self.anomaly_classes[defect_class_id]}")
            return
        
        # Create gallery
        defect_name = self.anomaly_classes[defect_class_id].replace('_', ' ')
        
        # Determine grid dimensions
        n_cols = min(5, max_examples)
        n_rows = (max_examples + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
        
        for i, example in enumerate(examples):
            build_idx = example['build_idx']
            layer = example['layer']
            build_name = example['build_name']
            
            # Load image and mask
            with h5py.File(self.hdf5_files[build_idx], 'r') as f:
                # Get the image
                image = f['slices/camera_data/visible/0'][layer, ...]
                
                # Get the defect mask
                defect_mask = f[f'slices/segmentation_results/{defect_class_id}'][layer, ...]
                
                # Normalize image
                img_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
                
                # Create RGB image with overlay
                rgb_img = np.stack([img_norm, img_norm, img_norm], axis=-1)
                
                # Add colored overlay
                color = self.defect_colors.get(defect_class_id, 'red')
                from matplotlib.colors import to_rgb
                r, g, b = to_rgb(color)
                
                alpha = 0.7  # Transparency
                mask_3d = np.stack([defect_mask, defect_mask, defect_mask], axis=-1)
                color_overlay = np.zeros_like(rgb_img)
                color_overlay[mask_3d > 0] = [r, g, b]
                
                # Blend with original image
                rgb_img = np.where(mask_3d > 0, 
                                   (1-alpha)*rgb_img + alpha*color_overlay, 
                                   rgb_img)
                
                # Plot
                ax = fig.add_subplot(n_rows, n_cols, i+1)
                ax.imshow(rgb_img)
                ax.set_title(f"{build_name}\nLayer {layer}")
                ax.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f"{defect_name} Examples Across Builds", fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.92)
        
        # Save
        output_path = os.path.join(gallery_dir, f"{self.anomaly_classes[defect_class_id]}_gallery.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Created gallery for {defect_name} with {len(examples)} examples")

    def visualize_process_parameters(self):
        """
        Visualize the relationship between process parameters and defects.
        """
        # Collect process parameters and defect statistics
        data = []
        
        for build_idx, (file_path, build_name) in enumerate(zip(self.hdf5_files, self.build_names)):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Check if process parameters exist
                    if 'parts/process_parameters/laser_beam_power' not in f:
                        print(f"No process parameters found in {build_name}")
                        continue
                    
                    # Get part IDs
                    if 'slices/part_ids' in f:
                        # Get unique part IDs (excluding 0, which is background)
                        part_ids = np.unique(f['slices/part_ids'][...])
                        part_ids = part_ids[part_ids > 0]
                        
                        # Get process parameters for each part
                        for part_id in part_ids:
                            # Get parameters (subtract 1 because part IDs start at 1)
                            idx = int(part_id) - 1
                            
                            if idx < len(f['parts/process_parameters/laser_beam_power']):
                                power = f['parts/process_parameters/laser_beam_power'][idx]
                                speed = f['parts/process_parameters/laser_beam_speed'][idx]
                                
                                # Try to get other parameters if available
                                hatch = f['parts/process_parameters/hatch_spacing'][idx] if 'parts/process_parameters/hatch_spacing' in f else np.nan
                                spot_size = f['parts/process_parameters/laser_spot_size'][idx] if 'parts/process_parameters/laser_spot_size' in f else np.nan
                                
                                # Calculate energy density (if possible)
                                if power > 0 and speed > 0 and hatch > 0:
                                    energy_density = power / (speed * hatch)
                                else:
                                    energy_density = np.nan
                                
                                # Get part name if available
                                part_name = f"Part {part_id}"
                                if 'parts/process_parameters/parameter_set' in f and idx < len(f['parts/process_parameters/parameter_set']):
                                    param_set = f['parts/process_parameters/parameter_set'][idx]
                                    if isinstance(param_set, bytes):
                                        param_set = param_set.decode('utf-8')
                                    part_name = param_set
                                
                                # Count defects for this part
                                defect_counts = {class_id: 0 for class_id in self.defect_class_ids}
                                total_defect_pixels = 0
                                
                                # Sample layers for efficiency
                                total_layers = f['slices/camera_data/visible/0'].shape[0]
                                sample_step = max(1, total_layers // 50)  # Sample up to 50 layers
                                
                                for layer in range(0, total_layers, sample_step):
                                    # Get part mask for this layer
                                    part_mask = (f['slices/part_ids'][layer, ...] == part_id)
                                    
                                    # Count defects within this part
                                    for class_id in self.defect_class_ids:
                                        if f'slices/segmentation_results/{class_id}' in f:
                                            defect_mask = f[f'slices/segmentation_results/{class_id}'][layer, ...]
                                            
                                            # Count defects only within this part
                                            defect_in_part = np.logical_and(defect_mask, part_mask)
                                            defect_pixels = np.sum(defect_in_part)
                                            
                                            if defect_pixels > 0:
                                                defect_counts[class_id] += 1
                                                total_defect_pixels += defect_pixels
                                
                                # Add to dataset
                                part_data = {
                                    'Build': build_name,
                                    'Part ID': part_id,
                                    'Part Name': part_name,
                                    'Laser Power (W)': power,
                                    'Scan Speed (mm/s)': speed,
                                    'Hatch Spacing (mm)': hatch,
                                    'Spot Size (mm)': spot_size,
                                    'Energy Density (J/mm²)': energy_density,
                                    'Total Defect Count': sum(defect_counts.values()),
                                    'Total Defect Pixels': total_defect_pixels
                                }
                                
                                # Add individual defect counts
                                for class_id in self.defect_class_ids:
                                    defect_name = self.anomaly_classes[class_id].replace('_', ' ')
                                    part_data[defect_name] = defect_counts[class_id]
                                
                                data.append(part_data)
            
            except Exception as e:
                print(f"Error processing {build_name} for process parameters: {str(e)}")
        
        # Create DataFrame
        if not data:
            print("No process parameter data found.")
            return
        
        param_df = pd.DataFrame(data)
        
        # Save to CSV
        param_csv = os.path.join(self.output_dir, "process_parameters.csv")
        param_df.to_csv(param_csv, index=False)
        print(f"Process parameter data saved to {param_csv}")
        
        # Create visualizations
        
        # 1. Scatter plot of energy density vs defect count
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=param_df, 
            x='Energy Density (J/mm²)', 
            y='Total Defect Count',
            hue='Build',
            size='Total Defect Pixels',
            sizes=(50, 300),
            alpha=0.7
        )
        plt.title('Energy Density vs. Total Defect Count', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'energy_density_vs_defects.png'), dpi=150)
        plt.close()
        
        # 2. Parameter space plot (Power vs. Speed) with defect count
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            param_df['Laser Power (W)'],
            param_df['Scan Speed (mm/s)'],
            c=param_df['Total Defect Count'],
            s=param_df['Total Defect Pixels'] / 100 + 50,  # Scale size
            alpha=0.7,
            cmap='YlOrRd'
        )
        
        # Add part labels
        for i, row in param_df.iterrows():
            plt.annotate(
                row['Part Name'],
                (row['Laser Power (W)'], row['Scan Speed (mm/s)']),
                fontsize=8,
                alpha=0.7,
                ha='center',
                va='bottom'
            )
        
        plt.colorbar(scatter, label='Total Defect Count')
        plt.xlabel('Laser Power (W)', fontsize=12)
        plt.ylabel('Scan Speed (mm/s)', fontsize=12)
        plt.title('Process Parameter Space with Defect Counts', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'process_parameter_space.png'), dpi=150)
        plt.close()
        
        # 3. Heatmap of specific defect types vs process parameters
        # Group by parameter ranges
        if len(param_df) > 10:  # Only if we have enough data points
            # Create parameter bins
            param_df['Power Bin'] = pd.cut(param_df['Laser Power (W)'], 5)
            param_df['Speed Bin'] = pd.cut(param_df['Scan Speed (mm/s)'], 5)
            
            # For each defect type, create a heatmap
            for class_id in self.defect_class_ids:
                defect_name = self.anomaly_classes[class_id].replace('_', ' ')
                
                if defect_name in param_df.columns:
                    # Skip if no defects of this type
                    if param_df[defect_name].sum() == 0:
                        continue
                    
                    # Create pivot table
                    pivot = pd.pivot_table(
                        param_df, 
                        values=defect_name, 
                        index='Power Bin', 
                        columns='Speed Bin',
                        aggfunc='mean',
                        fill_value=0
                    )
                    
                    # Create heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.1f')
                    plt.title(f'{defect_name} Occurrence vs. Process Parameters', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'{self.anomaly_classes[class_id]}_parameter_heatmap.png'), dpi=150)
                    plt.close()
        
        # 4. Correlation of process parameters with defect types
        # Calculate correlations
        corr_cols = ['Laser Power (W)', 'Scan Speed (mm/s)', 'Hatch Spacing (mm)', 'Energy Density (J/mm²)']
        defect_cols = [self.anomaly_classes[id].replace('_', ' ') for id in self.defect_class_ids]
        
        # Filter columns that exist in the dataframe
        corr_cols = [col for col in corr_cols if col in param_df.columns]
        defect_cols = [col for col in defect_cols if col in param_df.columns]
        
        if corr_cols and defect_cols:
            # Calculate correlation matrix
            corr_df = param_df[corr_cols + defect_cols].corr()
            
            # Extract parameter-defect correlations
            param_defect_corr = corr_df.loc[corr_cols, defect_cols]
            
            # Plot heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(param_defect_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
            plt.title('Correlation of Process Parameters with Defect Types', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'parameter_defect_correlation.png'), dpi=150)
            plt.close()
    
    def visualize_mechanical_properties(self):
        """
        Visualize the relationship between defects and mechanical properties.
        """
        # Collect mechanical property data and defect statistics
        data = []
        
        for build_idx, (file_path, build_name) in enumerate(zip(self.hdf5_files, self.build_names)):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Check if mechanical property data exists
                    if 'samples/test_results/ultimate_tensile_strength' not in f:
                        print(f"No mechanical property data found in {build_name}")
                        continue
                    
                    # Get sample IDs (excluding 0, which is background)
                    if 'slices/sample_ids' in f:
                        sample_ids = np.unique(f['slices/sample_ids'][...])
                        sample_ids = sample_ids[sample_ids > 0]
                        
                        # Process each sample
                        for sample_id in sample_ids:
                            # Get mechanical properties (subtract 1 because IDs start at 1)
                            idx = int(sample_id) - 1
                            
                            # Skip if out of bounds
                            if idx >= len(f['samples/test_results/ultimate_tensile_strength']):
                                continue
                            
                            # Get tensile strength
                            uts = f['samples/test_results/ultimate_tensile_strength'][idx]
                            
                            # Skip if no data (value = 0)
                            if uts == 0:
                                continue
                            
                            # Try to get other properties if available
                            properties = {
                                'Ultimate Tensile Strength (MPa)': uts,
                                'Yield Strength (MPa)': f['samples/test_results/yield_strength'][idx] if 'samples/test_results/yield_strength' in f else np.nan,
                                'Total Elongation (%)': f['samples/test_results/total_elongation'][idx] if 'samples/test_results/total_elongation' in f else np.nan,
                                'Uniform Elongation (%)': f['samples/test_results/uniform_elongation'][idx] if 'samples/test_results/uniform_elongation' in f else np.nan
                            }
                            
                            # Count defects for this sample
                            defect_counts = {class_id: 0 for class_id in self.defect_class_ids}
                            defect_areas = {class_id: 0 for class_id in self.defect_class_ids}
                            
                            # Sample layers for efficiency
                            total_layers = f['slices/camera_data/visible/0'].shape[0]
                            sample_step = max(1, total_layers // 50)  # Sample up to 50 layers
                            
                            for layer in range(0, total_layers, sample_step):
                                # Get sample mask for this layer
                                sample_mask = (f['slices/sample_ids'][layer, ...] == sample_id)
                                
                                # Skip if no sample pixels in this layer
                                if not np.any(sample_mask):
                                    continue
                                
                                # Count defects within this sample
                                for class_id in self.defect_class_ids:
                                    if f'slices/segmentation_results/{class_id}' in f:
                                        defect_mask = f[f'slices/segmentation_results/{class_id}'][layer, ...]
                                        
                                        # Count defects only within this sample
                                        defect_in_sample = np.logical_and(defect_mask, sample_mask)
                                        defect_pixels = np.sum(defect_in_sample)
                                        
                                        if defect_pixels > 0:
                                            defect_counts[class_id] += 1
                                            defect_areas[class_id] += defect_pixels
                            
                            # Add to dataset
                            sample_data = {
                                'Build': build_name,
                                'Sample ID': sample_id,
                                'Total Defect Count': sum(defect_counts.values()),
                                'Total Defect Area': sum(defect_areas.values())
                            }
                            
                            # Add mechanical properties
                            sample_data.update(properties)
                            
                            # Add individual defect counts and areas
                            for class_id in self.defect_class_ids:
                                defect_name = self.anomaly_classes[class_id].replace('_', ' ')
                                sample_data[f"{defect_name} Count"] = defect_counts[class_id]
                                sample_data[f"{defect_name} Area"] = defect_areas[class_id]
                            
                            data.append(sample_data)
            
            except Exception as e:
                print(f"Error processing {build_name} for mechanical properties: {str(e)}")
        
        # Create DataFrame
        if not data:
            print("No mechanical property data found.")
            return
        
        mech_df = pd.DataFrame(data)
        
        # Save to CSV
        mech_csv = os.path.join(self.output_dir, "mechanical_properties.csv")
        mech_df.to_csv(mech_csv, index=False)
        print(f"Mechanical property data saved to {mech_csv}")
        
        # Create visualizations
        
        # 1. Scatter plot of defect count vs. tensile strength
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=mech_df, 
            x='Total Defect Count',
            y='Ultimate Tensile Strength (MPa)',
            hue='Build',
            size='Total Defect Area',
            sizes=(50, 300),
            alpha=0.7
        )
        plt.title('Defect Count vs. Ultimate Tensile Strength', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'defect_count_vs_uts.png'), dpi=150)
        plt.close()
        
        # 2. Correlation between mechanical properties and defect types
        # Get defect columns
        defect_cols = []
        for class_id in self.defect_class_ids:
            defect_name = self.anomaly_classes[class_id].replace('_', ' ')
            if f"{defect_name} Count" in mech_df.columns:
                defect_cols.append(f"{defect_name} Count")
        
        # Get mechanical property columns
        mech_cols = [col for col in ['Ultimate Tensile Strength (MPa)', 'Yield Strength (MPa)', 
                                    'Total Elongation (%)', 'Uniform Elongation (%)']
                    if col in mech_df.columns]
        
        if defect_cols and mech_cols:
            # Calculate correlation matrix
            corr_df = mech_df[defect_cols + mech_cols].corr()
            
            # Extract defect-property correlations
            defect_mech_corr = corr_df.loc[defect_cols, mech_cols]
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(defect_mech_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
            plt.title('Correlation of Defect Types with Mechanical Properties', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'defect_mechanical_correlation.png'), dpi=150)
            plt.close()
        
        # 3. Regression plots for most correlated defect types
        for defect_col in defect_cols:
            for mech_col in mech_cols:
                # Skip if no data
                if mech_df[defect_col].sum() == 0:
                    continue
                
                plt.figure(figsize=(10, 6))
                sns.regplot(
                    data=mech_df,
                    x=defect_col,
                    y=mech_col,
                    scatter_kws={"alpha": 0.6},
                    line_kws={"color": "red"}
                )
                plt.title(f'{defect_col} vs. {mech_col}', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Calculate correlation for the title
                corr = mech_df[[defect_col, mech_col]].corr().iloc[0, 1]
                plt.suptitle(f'Correlation: {corr:.3f}', fontsize=12, y=0.05)
                
                # Create valid filename
                safe_defect = defect_col.replace(' ', '_').replace('(', '').replace(')', '')
                safe_mech = mech_col.replace(' ', '_').replace('(', '').replace(')', '')
                plt.savefig(os.path.join(self.output_dir, f'{safe_defect}_vs_{safe_mech}.png'), dpi=150)
                plt.close()
    
    def run_comprehensive_analysis(self):
        """
        Run a comprehensive analysis across all builds.
        """
        print("\n===== Starting Multi-Build Peregrine Dataset Analysis =====")
        
        # 1. Get dataset summary
        print("\n1. Analyzing dataset summary...")
        self.get_dataset_summary()
        
        # 2. Compare defect distributions
        print("\n2. Comparing defect distributions across builds...")
        self.visualize_defect_comparison()
        
        # 3. Create defect galleries for each type
        print("\n3. Creating defect galleries...")
        self.create_defect_gallery()
        
        # 4. Find similar layers across builds
        print("\n4. Finding similar defect patterns across builds...")
        self.find_similar_layers(num_layers=5)
        
        # 5. Analyze process parameters
        print("\n5. Analyzing relationship between process parameters and defects...")
        self.visualize_process_parameters()
        
        # 6. Analyze mechanical properties
        print("\n6. Analyzing relationship between defects and mechanical properties...")
        self.visualize_mechanical_properties()
        
        print("\n===== Multi-Build Analysis Complete =====")
        print(f"Results saved to {self.output_dir}")


if __name__ == "__main__":
    # Listing of HDF5 files for all 5 builds
    hdf5_files = [
        "C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-08-23 TCR Phase 1 Build 1.hdf5",
        "C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-08-23 TCR Phase 1 Build 2.hdf5",
        "C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-08-23 TCR Phase 1 Build 3.hdf5",
        "C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-08-23 TCR Phase 1 Build 4.hdf5",
        "C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-08-23 TCR Phase 1 Build 5.hdf5"
    ]
    
    # Initialize the visualizer
    visualizer = MultiPeregrineVisualizer(hdf5_files, output_dir="./peregrine_analysis")
    
    # Run comprehensive analysis
    visualizer.run_comprehensive_analysis()
    
    # Alternatively, you can run individual analyses:
    # visualizer.get_dataset_summary()
    # visualizer.visualize_defect_comparison()
    # visualizer.create_defect_gallery()
    # visualizer.find_similar_layers()
    # visualizer.visualize_process_parameters()
    # visualizer.visualize_mechanical_properties()