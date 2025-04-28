import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from pycocotools.coco import COCO


def get_category_name(coco_data, category_id):
    """Get category name from category ID."""
    for category in coco_data["categories"]:
        if category["id"] == category_id:
            return category["name"]
    return f"Unknown ({category_id})"

def plot_bbox(ax, bbox, color, x0=0, y0=0, crop_coords=None, image_shape=None):
    x, y, w, h = bbox
    x_disp, y_disp = x - x0, y - y0
    if crop_coords and image_shape:
        x1, y1 = crop_coords[1], crop_coords[3]
        if x + w < x0 or x > x1 or y + h < y0 or y > y1:
            return
        x_disp = max(0, x_disp)
        y_disp = max(0, y_disp)
        w = min(w, x1 - x) if x + w > x1 else w
        h = min(h, y1 - y) if y + h > y1 else h
    rect = patches.Rectangle((x_disp, y_disp), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

def plot_segmentation(ax, segmentation, color, x0=0, y0=0):
    for seg in segmentation:
        coords = np.array(seg).reshape(-1, 2)
        coords[:, 0] -= x0
        coords[:, 1] -= y0
        poly = MplPolygon(
            coords,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=2,
            alpha=1.0
        )
        ax.add_patch(poly)

def visualize_annotations(
    image,
    annotation_ids,
    coco: COCO,
    crop_coords=None,
    show_masks=True,
    show_class_names=False,
    show_bboxes=False
):
    """
    Visualize COCO annotations on an image using matplotlib polygons for correct holes/edges.

    Args:
        image: np.ndarray, the image array (H, W, C)
        annotation_ids: list of annotation IDs to visualize
        coco: pycocotools.coco.COCO object
        crop_coords: Optional (x_start, x_end, y_start, y_end) for cropping
        show_masks: Show segmentation masks
        show_class_names: Show class names
        show_bboxes: Show bounding boxes
    """

    # Prepare figure
    fig, ax = plt.subplots(1, figsize=(15, 15))

    # Handle cropping
    if crop_coords is not None and len(crop_coords) == 4:
        x0, x1, y0, y1 = crop_coords
        x0, x1 = max(0, x0), min(image.shape[1], x1)
        y0, y1 = max(0, y0), min(image.shape[0], y1)
        img_disp = image[y0:y1, x0:x1]
        ax.imshow(img_disp)
        ax.set_title(f"Cropped view ({x0}:{x1}, {y0}:{y1})")
    else:
        img_disp = image
        x0, y0 = 0, 0
        ax.imshow(img_disp)
        ax.set_title("Full image view")

    # Color map for categories
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    n_cats = len(cats)
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_cats)))
    catid_to_color = {cat['id']: colors[i % len(colors)] for i, cat in enumerate(cats)}

    anns = coco.loadAnns(annotation_ids)
    ann_cat_ids = set(ann['category_id'] for ann in anns)
    use_single_color = len(ann_cat_ids) == 1
    if use_single_color:
        # All objects are the same class: assign a unique color for each annotation (per label)
        single_color_list = plt.cm.tab20(np.linspace(0, 1, max(20, len(anns))))
        annid_to_color = {ann['id']: single_color_list[i % len(single_color_list)] for i, ann in enumerate(anns)}
    else:
        # Multiple classes: assign color per class
        annid_to_color = {ann['id']: catid_to_color[ann['category_id']] for ann in anns}

    for ann in anns:
        color = annid_to_color[ann['id']]
        cat_id = ann['category_id']
        # Bounding box
        if show_bboxes and 'bbox' in ann:
            plot_bbox(ax, ann['bbox'], color, x0, y0, crop_coords, image.shape)
        # Class name
        if show_class_names:
            x, y = ann['bbox'][:2]
            x_disp, y_disp = x - x0, y - y0
            cat_name = coco.loadCats([cat_id])[0]['name']
            ax.text(x_disp, y_disp-5, cat_name, color=color, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        # Segmentation mask (polygon, with holes support)
        if show_masks and 'segmentation' in ann and isinstance(ann['segmentation'], list):
            plot_segmentation(ax, ann['segmentation'], color, x0, y0)

    # Legend
    if not use_single_color:
        legend_patches = [
            mpatches.Patch(color=catid_to_color[cat['id']], label=cat['name'])
            for cat in cats
        ]
        ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.01, 0.5), title="Categories")
    plt.tight_layout()
    plt.show()