---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: bcd
  language: python
  name: python3
---
# Image Segmentation for Mammography

Image segmentation
Segmentation is the process by which an image is partitioned into separate regions for the purposes of to detect and extract the region of interest (ROI), breast tissue containing one or more masses or calcifications. However, ROI detection and segmentation can be affected by pectoral muscles, text, and other artifacts which may cause over-segmentation, and lead to misclassification of breast tissue abnormalities. Below, four randomly selected mammogram images from the CBIS-DDSM dataset illustrate the challenge.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../../..")))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from bcd.utils.image import convert_uint8
from myst_nb import glue

img1 = "data/image/2_exp/train/benign/2a44122c-f831-4220-95a8-408bcafcf2ce.png"
img2 = "data/image/2_exp/train/benign/3f72309d-7cd9-4e30-ae81-073adb541bcd.png"
img3 = "data/image/2_exp/train/benign/97556037-b959-4395-830b-380dcac2d58e.png"
img4 = "data/image/2_exp/train/malignant/6cdf46d8-596b-47ab-a428-c8769733c93c.png"

img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(img4, cv2.IMREAD_GRAYSCALE)

img1 = convert_uint8(img1)
img2 = convert_uint8(img2)
img3 = convert_uint8(img3)
img4 = convert_uint8(img4)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,6))
_ = axes[0].imshow(img1, cmap='gray', aspect='auto')
_ = axes[1].imshow(img2, cmap='gray',aspect='auto')
_ = axes[2].imshow(img3, cmap='gray',aspect='auto')
_ = axes[3].imshow(img4, cmap='gray',aspect='auto')

labels = np.array(["(a)", "(b)", "(c)", "(d)"])
for i in range(4):
    _ = axes[i].set_xlabel(labels[i])
    _ = axes[i].set_xticks([])
    _ = axes[i].set_yticks([])


plt.tight_layout()

glue("artifacts", fig)
```

```{glue:figure} artifacts
---
align: center
name: artifacts_fig
---
Artifacts in Mammogram Images
```

As shown in {numref}`artifacts_fig`, edges, text, annotations, and other anomalies can have pixel characteristics similar to those of the ROI. Consequently, artifacts and other anomalies must be removed before ROI detection and segmentation can occur.

This section highlights the classical segmentation methods most commonly used to remove artifacts in biomedical imaging in general, and in mammography specifically. We'll be focusing on three classifications of image segmentation: threshold-based segmentation, edge-based segmentation, and region-based segmentation.

- Threshold-Based Segmentation (TBS) partitions regions of an image by applying a mask formed by applying one or more pixel intensity thresholds to the image. Such methods include Otsu thresholding, morphological thresholding, and global and adaptive thresholding.
- Edge-Based Segmentation (EBS) uses contours and gradients in the image to detect the edges surrounding one or more regions of interest. Canny Edge Detection, Active Contour, and Sobel segmentation are among the most common EBS methods.
- Region-Based Segmentation (RBS) separates the image into regions with similar characteristics. The Watershed algorithm and certain morphological methods are prominent in this category.

We’ll review the most prominent threshold-Based, edge-Based, and region-Based methods, evaluate their performance, and characterize their relative strengths, and shortcomings.
