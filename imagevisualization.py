import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.collections as collections
import numpy as np
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from matplotlib.colors import ListedColormap

# Needed to enable plotting if using the Spyder IDE
try:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('matplotlib qt')
except:
    pass

# Create output directory for saved visualizations
output_dir = "./peregrine_visualizations"
os.makedirs(output_dir, exist_ok=True)

# List of all HDF5 files
hdf5_files = [
    r"C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-04-16 TCR Phase 1 Build 2.hdf5",
    r"C:\Users\Gospel\Documents\Peregrine Dataset v2023-11\2021-08-23 TCR Phase 1 Build 5.hdf5"
]

# Defect class information
anomaly_classes = {
    0: "Powder",
    1: "Printed",
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

# Define colors for each defect type
defect_colors = {
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

# Defect class IDs (excluding normal conditions)
defect_class_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Function to create a summary for all datasets
def create_dataset_summary():
    """Create a summary of all datasets"""
    print("\nGenerating dataset summary...")
    
    summary_data = []
    
    for file_path in hdf5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                # Get build name
                if 'core/build_name' in f.attrs:
                    build_name = f.attrs['core/build_name']
                    if isinstance(build_name, bytes):
                        build_name = build_name.decode('utf-8')
                else:
                    build_name = f"Build_{Path(file_path).stem}"
                
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
                
                for class_id in defect_class_ids:
                    defect_counts[class_id] = 0
                    if f'slices/segmentation_results/{class_id}' in f:
                        # Sample a subset of layers for efficiency
                        sample_step = max(1, total_layers // 20)  # Sample 20 layers
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
                for class_id in defect_class_ids:
                    defect_name = anomaly_classes[class_id].replace('_', ' ')
                    summary_data[-1][defect_name] = defect_counts[class_id]
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Display summary
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nDataset Summary:")
    print(summary_df)
    
    # Save to CSV
    summary_csv = os.path.join(output_dir, "dataset_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")
    
    return summary_df

# Function to visualize a layer with defect overlay
def visualize_layer_with_defects(file_path, layer_number, save=True):
    """Visualize a layer with defect overlays"""
    build_name = Path(file_path).stem
    print(f"\nVisualizing layer {layer_number} from {build_name}...")
    
    with h5py.File(file_path, 'r') as f:
        total_layers = f['slices/camera_data/visible/0'].shape[0]
        
        if layer_number >= total_layers:
            print(f"Error: Layer {layer_number} exceeds maximum layer {total_layers-1}")
            return None
        
        # Get the image data
        image = f['slices/camera_data/visible/0'][layer_number, ...]
        
        # Check which defects are present in this layer
        defects_present = []
        for class_id in defect_class_ids:
            if f'slices/segmentation_results/{class_id}' in f:
                defect_mask = f[f'slices/segmentation_results/{class_id}'][layer_number, ...]
                if np.any(defect_mask):
                    defects_present.append(class_id)
        
        # Create figure with three panels
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Create a combined defect mask
        all_defects = np.zeros_like(image, dtype=np.uint8)
        for class_id in defects_present:
            defect_mask = f[f'slices/segmentation_results/{class_id}'][layer_number, ...]
            all_defects[defect_mask > 0] = class_id
        
        # Custom colormap for defects
        colors = ['black']  # For 0 (no defect)
        for i in range(1, max(defect_class_ids) + 1):
            if i in defect_colors:
                colors.append(defect_colors[i])
            else:
                colors.append('gray')  # Default color
        
        defect_cmap = ListedColormap(colors)
        
        # Defect mask
        axes[1].imshow(all_defects, cmap=defect_cmap, vmin=0, vmax=max(defect_class_ids))
        axes[1].set_title("Defect Mask")
        axes[1].axis('off')
        
        
        # Normalize the original image
        img_norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        
        rgb_img = np.stack([img_norm, img_norm, img_norm], axis=-1)
        
        
        for class_id in defects_present:
            defect_mask = f[f'slices/segmentation_results/{class_id}'][layer_number, ...]
            
            
            color = defect_colors.get(class_id, 'white')
            r, g, b = mcolors.to_rgb(color)
            
            
            alpha = 0.7  # Transparency
            mask_3d = np.stack([defect_mask, defect_mask, defect_mask], axis=-1)
            color_overlay = np.zeros_like(rgb_img)
            color_overlay[mask_3d > 0] = [r, g, b]
            
            # Blend with original image
            rgb_img = np.where(mask_3d > 0, (1-alpha)*rgb_img + alpha*color_overlay, rgb_img)
        
        # Show overlay
        axes[2].imshow(rgb_img)
        axes[2].set_title("Defect Overlay")
        axes[2].axis('off')
        
        # Add a legend
        legend_elements = []
        for class_id in defects_present:
            from matplotlib.patches import Patch
            legend_elements.append(
                Patch(facecolor=defect_colors[class_id], 
                      label=anomaly_classes[class_id].replace('_', ' '))
            )
        
        if legend_elements:
            axes[2].legend(handles=legend_elements, loc='upper right')
        
        # Add title with layer info
        defect_names = [anomaly_classes[cid].replace('_', ' ') for cid in defects_present]
        plt.suptitle(f"{build_name} - Layer {layer_number}" + 
                    (f"\nDefects: {', '.join(defect_names)}" if defect_names else "\nNo defects detected"), 
                    fontsize=16)
        
        plt.tight_layout()
        
        if save:
            # Create build-specific directory
            build_dir = os.path.join(output_dir, build_name)
            os.makedirs(build_dir, exist_ok=True)
            
            plt.savefig(os.path.join(build_dir, f"layer_{layer_number:04d}.png"), 
                        dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig

# Function to visualize scan paths
def visualize_scan_path(file_path, layer_number, save=True):
    """Visualize the laser scan path for a layer"""
    build_name = Path(file_path).stem
    print(f"\nVisualizing scan path for layer {layer_number} from {build_name}...")
    
    with h5py.File(file_path, 'r') as f:
        # Check if scan path data exists
        if f'scans/%i' % (layer_number) not in f:
            print(f"No scan path data found for layer {layer_number}")
            return None
        
        # Extract scan path data
        scan_data = f[f'scans/%i' % (layer_number)][...]
        
        if len(scan_data) == 0:
            print(f"Empty scan path data for layer {layer_number}")
            return None
        
        # Extract x, y coordinates and timing
        x_start, x_end = scan_data[:, 0], scan_data[:, 1]
        y_start, y_end = scan_data[:, 2], scan_data[:, 3]
        times = scan_data[:, 4]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Normalize time for coloring
        norm_times = (times - np.min(times)) / (np.max(times) - np.min(times))
        
        # Create line segments
        points = np.array([np.column_stack([x_start, y_start]), 
                         np.column_stack([x_end, y_end])])
        points = points.transpose(1, 0, 2)
        
        # Create a line collection
        colormap = cm.get_cmap('jet')
        colors = [colormap(t) for t in norm_times]
        lc = collections.LineCollection(points, colors=colors, linewidths=1)
        
        # Plot
        ax.add_collection(lc)
        ax.set_xlim(min(np.min(x_start), np.min(x_end)), max(np.max(x_start), np.max(x_end)))
        ax.set_ylim(min(np.min(y_start), np.min(y_end)), max(np.max(y_start), np.max(y_end)))
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_array([])
        plt.colorbar(sm, label="Relative Time")
        
        plt.title(f"{build_name} - Layer {layer_number} Scan Path")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save:
            # Create build-specific directory
            build_dir = os.path.join(output_dir, build_name)
            os.makedirs(build_dir, exist_ok=True)
            
            plt.savefig(os.path.join(build_dir, f"scan_path_layer_{layer_number:04d}.png"), 
                        dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig

# Function to visualize temporal data
def visualize_temporal_data(file_path, save=True):
    """Visualize temporal sensor data"""
    build_name = Path(file_path).stem
    print(f"\nVisualizing temporal data from {build_name}...")
    
    with h5py.File(file_path, 'r') as f:
        # Select interesting temporal data series
        temporal_keys = [
            'temporal/build_chamber_position',
            'temporal/build_plate_temperature',
            'temporal/top_chamber_temperature',
            'temporal/module_oxygen',
            'temporal/layer_times'
        ]
        
        # Check which keys exist
        available_keys = [key for key in temporal_keys if key in f]
        
        if not available_keys:
            print("No temporal data found")
            return None
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        
        for i, key in enumerate(available_keys):
            data = f[key][...]
            units = f[key].attrs.get('units', '')
            
            plt.subplot(len(available_keys), 1, i+1)
            plt.plot(data, marker='.', linestyle='-', markersize=1)
            plt.title(key.split('/')[-1].replace('_', ' ').title())
            plt.ylabel(units)
            
            # Only add x-label for the bottom plot
            if i == len(available_keys) - 1:
                plt.xlabel("Layer Number")
        
        plt.tight_layout()
        
        if save:
            # Create build-specific directory
            build_dir = os.path.join(output_dir, build_name)
            os.makedirs(build_dir, exist_ok=True)
            
            plt.savefig(os.path.join(build_dir, "temporal_data.png"), 
                        dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig

# Function to find interesting layers with defects
def find_interesting_layers(file_path, num_layers=5, min_defect_area=200):
    """Find the most interesting layers with defects"""
    build_name = Path(file_path).stem
    print(f"\nFinding interesting layers in {build_name}...")
    
    with h5py.File(file_path, 'r') as f:
        total_layers = f['slices/camera_data/visible/0'].shape[0]
        
        # Calculate "interestingness" score for each layer
        layer_scores = np.zeros(total_layers)
        
        sample_step = max(1, total_layers // 100)  # Sample up to 100 layers
        
        for layer in tqdm(range(0, total_layers, sample_step), desc="Scanning layers"):
            # Count different defect types and their areas
            defect_count = 0
            total_defect_area = 0
            
            for class_id in defect_class_ids:
                if f'slices/segmentation_results/{class_id}' in f:
                    defect_mask = f[f'slices/segmentation_results/{class_id}'][layer, ...]
                    defect_area = np.sum(defect_mask > 0)
                    
                    if defect_area > min_defect_area:
                        defect_count += 1
                        total_defect_area += defect_area
            
            # Score based on number of defect types and total area
            layer_scores[layer] = defect_count * np.log1p(total_defect_area)
        
        # Interpolate scores for unsampled layers
        sampled_indices = np.arange(0, total_layers, sample_step)
        all_indices = np.arange(total_layers)
        
        if len(sampled_indices) > 1:  # Need at least 2 points for interpolation
            from scipy.interpolate import interp1d
            f_interp = interp1d(sampled_indices, layer_scores[sampled_indices], 
                              kind='linear', bounds_error=False, fill_value=0)
            layer_scores = f_interp(all_indices)
        
        # Select top scoring layers
        interesting_layers = np.argsort(layer_scores)[-num_layers:][::-1]
        
        print(f"\nTop {num_layers} most interesting layers in {build_name}:")
        for i, layer in enumerate(interesting_layers):
            print(f"{i+1}. Layer {layer} (score: {layer_scores[layer]:.2f})")
        
        return interesting_layers

# Function to compare defect types across builds
def compare_defect_distributions():
    """Compare defect distributions across builds"""
    print("\nComparing defect distributions across builds...")
    
    # First collect defect statistics
    defect_stats = []
    
    for file_path in hdf5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                # Get build name
                if 'core/build_name' in f.attrs:
                    build_name = f.attrs['core/build_name']
                    if isinstance(build_name, bytes):
                        build_name = build_name.decode('utf-8')
                else:
                    build_name = f"Build_{Path(file_path).stem}"
                
                total_layers = f['slices/camera_data/visible/0'].shape[0]
                
                # Count defects in sampled layers
                sample_step = max(1, total_layers // 50)  # Sample up to 50 layers
                
                for layer in tqdm(range(0, total_layers, sample_step), 
                                 desc=f"Processing {build_name}", leave=False):
                    layer_stats = {'Build': build_name, 'Layer': layer}
                    
                    for class_id in defect_class_ids:
                        if f'slices/segmentation_results/{class_id}' in f:
                            defect_mask = f[f'slices/segmentation_results/{class_id}'][layer, ...]
                            defect_area = np.sum(defect_mask > 0)
                            layer_stats[anomaly_classes[class_id]] = defect_area
                        else:
                            layer_stats[anomaly_classes[class_id]] = 0
                    
                    defect_stats.append(layer_stats)
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Convert to DataFrame
    defect_df = pd.DataFrame(defect_stats)
    
    # Create visualization
    
    # Reshape data for plotting
    plot_data = []
    for build in defect_df['Build'].unique():
        build_data = defect_df[defect_df['Build'] == build]
        
        for defect_type in [anomaly_classes[id] for id in defect_class_ids]:
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
    
    # Bar chart comparing defect prevalence
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(x='Defect Type', y='Percentage of Layers', hue='Build', data=plot_df)
    plt.title('Defect Prevalence Comparison Across Builds', fontsize=16)
    plt.xlabel('Defect Type', fontsize=12)
    plt.ylabel('Percentage of Layers with Defect', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Build', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'defect_prevalence_comparison.png'), dpi=150)
    plt.show()
    
    # Heatmap of defect prevalence
    defect_pivot = pd.pivot_table(
        plot_df, 
        values='Percentage of Layers', 
        index='Build', 
        columns='Defect Type'
    )
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(defect_pivot, annot=True, cmap='YlOrRd', fmt='.1f', 
               cbar_kws={'label': 'Percentage of Layers'})
    plt.title('Defect Prevalence Heatmap Across Builds', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'defect_heatmap.png'), dpi=150)
    plt.show()
    
    return defect_df

# Main function
def main():
    """Run the main visualization workflow"""
    print("Peregrine Dataset Visualizer")
    print("===========================")
    
    # Create a summary of all datasets
    summary_df = create_dataset_summary()
    
    # Compare defect distributions across builds
    defect_df = compare_defect_distributions()
    
    # Process each file
    for file_path in hdf5_files:
        try:
            print(f"\nProcessing {file_path}...")
            
            # Find interesting layers with defects
            interesting_layers = find_interesting_layers(file_path, num_layers=5)
            
            # Visualize each interesting layer
            for layer in interesting_layers:
                visualize_layer_with_defects(file_path, layer)
                visualize_scan_path(file_path, layer)
            
            # Visualize temporal data
            visualize_temporal_data(file_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"\nVisualization complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
