import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from glob import glob
from pathlib import Path
import random
from collections import defaultdict


# Color map for different classes
CLASS_COLORS = defaultdict(lambda: 'red')  # default to red
PRESET_COLORS = ['red', 'lime', 'blue', 'orange', 'magenta', 'cyan', 'yellow', 'purple', 'brown', 'deeppink']


def assign_colors(class_list):
    for i, cls in enumerate(sorted(set(class_list))):
        CLASS_COLORS[cls] = PRESET_COLORS[i % len(PRESET_COLORS)]


def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        objects.append({"label": label, "bbox": (xmin, ymin, xmax, ymax)})
    return objects


def visualize_dataset(split_dir, title, num_samples=30, save=False, output_dir="./outputs"):
    image_files = sorted(glob(os.path.join(split_dir, "*.jpg")))
    xml_files = sorted(glob(os.path.join(split_dir, "*.xml")))

    if len(image_files) == 0:
        print(f"No images found in {split_dir}")
        return

    samples = random.sample(list(zip(image_files, xml_files)), min(num_samples, len(image_files)))

   
    all_classes = []
    for _, xml in samples:
        anns = parse_voc_annotation(xml)
        all_classes.extend([a['label'] for a in anns])
    assign_colors(all_classes)

    cols = 5
    rows = (len(samples) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axs = axs.flatten()

    for i, (img_path, xml_path) in enumerate(samples):
        img = Image.open(img_path).convert("RGB")
        annotations = parse_voc_annotation(xml_path)

        axs[i].imshow(img)
        axs[i].set_title(Path(img_path).name, fontsize=9)

        for ann in annotations:
            xmin, ymin, xmax, ymax = ann["bbox"]
            label = ann["label"]
            color = CLASS_COLORS[label]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor=color, facecolor='none')
            axs[i].add_patch(rect)
            axs[i].text(xmin, ymin - 5, label, color='white',
                        bbox=dict(facecolor=color, alpha=0.7), fontsize=7)

        axs[i].axis('off')

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.suptitle(f"{title} - Annotated Samples ({len(samples)} images)", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{title.lower()}_samples.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved {title} visualization to {out_path}")
        plt.close()
    else:
        plt.show()


def visualize_all_sets(dataset_base_path, save=False):
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(dataset_base_path, split)
        if os.path.exists(split_path):
            visualize_dataset(split_path, split.capitalize(), save=save)
        else:
            print(f"{split_path} not found.")



if __name__ == "__main__":
    dataset_base_path = "/content/data_base_path"
    visualize_all_sets(dataset_base_path, save=False)  
