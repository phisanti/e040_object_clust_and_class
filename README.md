# Object Classification for ASTC Microscopy Images

## Overview

This repository contains a framework for training the object classifiers for the ASCT. Object annotations were made in a semi-supervised way using iLastik framework. The annotatioins are stored with the COCO annotation standard fromat. Given the simplicity of hte objects, teh neural network architecture is kept simple using Residual Blocks.
## Dataset Structure

The project expects data in COCO annotation format with the following structure:

- Microscopy images in TIFF format
- COCO annotation JSON file with categories: "cell", "clump", "noise", "off-focus", and "joint cell"
- Each object has a bounding box and segmentation mask

### Example Annotation Structure

```json
{
  "images": [{"id": 1, "file_name": "image.tiff", "width": 2400, "height": 2400}, ...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]]
    },
    ...
  ],
  "categories": [
    {"id": 1, "name": "cell"},
    {"id": 2, "name": "clump"},
    {"id": 3, "name": "noise"},
    {"id": 4, "name": "off-focus"},
    {"id": 5, "name": "joint cell"}
  ]
}
```

