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
# Threshold-Based Segmentation

Our first step in breast segmentation is to separate the foreground of the mammogram containing the breast ROI from the background. From a grayscale mammogram, we can use threshold-based segmentation to create a binary image or mask, that, when applied to the original image, separates the foreground and the ROI from the background.

In binary masking, each pixel in the binary mask has one of two values: ‘0’ representing the background, and ‘1’, corresponding to the image foreground.  Whether the pixel value in the binary mask is assigned a 0 or 1 depends on whether the associated image pixel value meets a designated or automatically selected threshold. Generally speaking, image pixel values less than or equal to the threshold are 0 in the corresponding binary mask, all other pixel values are set to 1.

Two commonly used threshold-based segmentation techniques are global threshold segmentation and adaptive local threshold segmentation.

## Global Threshold Segmentation

Global threshold segmentation applies a single designated or automatically selected threshold value to the entire image.  Specifically, let $I$ be the input image with height $H$ and width $W$ pixels, and that $I_{x,y}$ represents the gray value of row $x$ and column $y$ of $I$, such that $0 \le x < H, 0 \le y < W$. Then, $B_{x,y}$ represents the binary value in row $x$ and column $y$ of binary mask $B$, and is given by:

```{math}
:label: global_threshold
B_{x,y} = \begin{cases}
1 & if I_{x,y} > T \\
0 & if I_{x,y} \le T \\
\end{cases}
```

where $T$ is a global threshold.

To illustrate, let’s create some binary masks for our test images using a manually set threshold $T=30$.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

import os
import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../../..")))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from bcd.utils.image import convert_uint8
from skimage.filters import threshold_mean
from bcd.preprocess.image.analysis.threshold import ThresholdMean, ThresholdLi, ThresholdYen, ThresholdMinimum, ThresholdISOData, ThresholdTriangle, ThresholdOTSU, ThresholdAdaptiveMean, TryAllThresholds, ThresholdManual

img1 = "data/image/1_dev/converted/train/benign/347c2455-cb62-40f8-a173-9e4eb9a21902.png"
img2 = "data/image/1_dev/converted/train/benign/4ed91643-1e06-4b2c-8efb-bc60dd9e0313.png"
img3 = "data/image/1_dev/converted/train/malignant/7dcc12fd-88f0-4048-a6ab-5dd0bd836f08.png"
img4 = "data/image/1_dev/converted/train/malignant/596ef5db-9610-4f13-9c1a-4c411b1d957c.png"

img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(img4, cv2.IMREAD_GRAYSCALE)

images = (img1,img2,img3,img4)

analysis = ThresholdManual(threshold=30)
fig = analysis.analyze(images=images)

glue("threshold_manual", fig)
```

```{glue:figure} threshold_manual
---
align: center
name: threshold_manual_fig
---
Threshold-Based Segmentation with $T=30$
```
