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
# Thresholding

```{code-cell} ipython3
:tags: [remove-input, remove-output]

import os
import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../../..")))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from myst_nb import glue

from bcd.preprocess.image.threshold import (ThresholdAnalyzer,
    ThresholdLi, ThresholdISOData, ThresholdTriangle, ThresholdOTSU, ThresholdAdaptiveMean, ThresholdAdaptiveGaussian, ThresholdManual, ThresholdYen, ThresholdTriangleAnalyzer
)
from bcd.utils.visual import plot_images

img1 = "data/image/1_dev/converted/train/benign/347c2455-cb62-40f8-a173-9e4eb9a21902.png"
img2 = "data/image/1_dev/converted/train/benign/4ed91643-1e06-4b2c-8efb-bc60dd9e0313.png"
img3 = "data/image/1_dev/converted/train/malignant/7dcc12fd-88f0-4048-a6ab-5dd0bd836f08.png"
img4 = "data/image/1_dev/converted/train/malignant/596ef5db-9610-4f13-9c1a-4c411b1d957c.png"

img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(img4, cv2.IMREAD_GRAYSCALE)

images = (img1,img2,img3,img4)
```

In digital mammography preprocessing, thresholding separates structures, features, and regions of interest (ROIs) from the image background based on pixel intensity values and a designated cut-off value or threshold.  From a grayscale image, a binary image is created where the pixel intensity values in the grayscale image are reduced to just two pixel intensity values in the binary image: 0 corresponding to all pixels in the grayscale image that are less than or equal to the threshold, and 1 (or 255 for 8-bit grayscale binary images) corresponding to all pixels in the grayscale image that are greater than the threshold.

## Thresholding in Digital Mammography

Segmentation based on pixel intensity thresholds is justified in digital mammography where pixel intensity is the parameter that most directly relates to the spatial characteristics of the structures within a mammogram. For this reason, threshold-based algorithm segmentation is a crucial early step in almost any digital mammography image analysis and preprocessing workflow.

## Critical Importance of Threshold

The selection of the threshold critically determines the quality of the thresholding result.


```{code-cell} ipython3
:tags: [remove-input, remove-output]

def plot_binary_images(image: np.ndarray, thresholds: list):
    thresholds = [32,64,128]
    images = [img3]
    for t in thresholds:
        _, img = cv2.threshold(img3, thresh=t, maxval=np.max(img3), type=cv2.THRESH_BINARY)
        images.append(img)
    titles = [f"Binary Image T={t}" for t in thresholds]
    titles = ["Original Image"] + titles
    title = "Binary Thresholding at Various Threshold Values"
    return plot_images(images=images, titles=titles, title=title)

thresholds = [32,64,128]
fig = plot_binary_images(image=img3, thresholds=thresholds)
glue("various_thresholds", fig)
```

```{glue:figure} various_thresholds
---
align: center
name: various_thresholds_fig
---
Impact of Threshold Value on Binarization
```

{numref}`various_thresholds_fig` illustrates the impact of various threshold values on the binarization result.  As the threshold value increases from $T=32$ to $T=128$, we notice a few things:

- **Border Smoothness**:  Borders tend to become increasingly rough as threshold value increases.
- **Information Loss**.  Threshold value and information loss are positively correlated. High thresholds can shrink objects to the background as indicated in {numref}`various_thresholds_fig`  (d).
- **Artifact Removal**: On the other hand, some information loss is acceptable if that information is an artifact. Higher thresholds tend to be associated with greater artifact removal.

And there's the trade-off. {numref}`various_thresholds_fig` clearly illustrates the importance of selecting an appropriate threshold, what principled techniques exist for selecting thresholds that balance information capture and artifact removal?

Well, manual techniques can be tedious, and time-consuming, and don’t reflect the inherent variability in digital mammography. No single threshold value will perform consistently across all images.  And though no universally superior *automated* thresholding method exists, several techniques have been proposed, each with distinct performance characteristics, strengths, and weaknesses, that have broad applicability across a range of image analysis and processing domains. We’ll examine those next.

## Automated Thresholding Algorithms

Sezgin and Sankur {cite}sankurSurveyImageThresholding2004 cast the space of automated thresholding techniques as follows: [^thresholds]

- **Histogram shape-based** methods that analyze, for instance, the peaks, valleys, and curvatures of smoothed histograms.
- **Clustering-based methods** cluster the gray-level samples into background and foreground. Alternatively, the image is modeled as a mixture of two Gaussians.
- **Entropy-based methods** use the entropy of the foreground and background regions, the cross-entropy between the original and binarized image, etc.
- **Object attribute-based** methods that analyze the similarity between the gray-level and the binarized images, such as fuzzy shape similarity, edge coincidence, etc.
- **The spatial methods** use higher-order probability distribution and/or correlation between pixels
- **Local methods adapt** the threshold value on each pixel to the local image characteristics.

[^thresholds]: For the taxonomist, Sezgin’s framework is not mutually exclusive and collectively exhaustive (MECE). For instance, Otsu’s Method {cite}otsuThresholdSelectionMethod1979 can be categorized as both a histogram shape-based method and a clustering-based method.

For this effort, eight automated thresholding techniques ({numref}`aut-thresh-tbl`) were selected based on the intrinsic properties of the CBIS-DDSM dataset.

```{table} Automated Threshold Methods
:name: auto-thresh-tbl

| # | Type                    | Method                               | Author(s)                                            | Publication                                                  |
|---|-------------------------|--------------------------------------|------------------------------------------------------|--------------------------------------------------------------|
| 1 | Histogram-Based         | Triangle Method                      | Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,    | Automatic Measurement of Sister Chromatid Exchange Frequency |
| 2 | Cluster-Based           | ISOData Method                       | Ridler, TW & Calvard, S (1978)                       | Picture thresholding using an iterative selection method     |
| 3 | Histogram/Cluster-Based | Otsu's Method                        | Nobuyuki Otsu (1979)                                 | A threshold selection method from gray-level histograms      |
| 4 | Entropy-Based           | Li's Minimum Cross-Entropy Method    | Li C.H. and Lee C.K. (1993)                          | Minimum Cross Entropy Thresholding                           |
| 5 | Spatial-Based           | Yen's Multilevel Thresholding Method | Jui-Cheng Yen, Fu-Juay Chang, and Shyang Chang (1995) | A new criterion for automatic multilevel thresholding        |
| 7 | Local                   | Adaptive Gaussian Method             | Bradley, D., G. Roth 2007                            | Adapting Thresholding Using the Integral Image               |
| 8 |                         | Adaptive Mean Method                 | Bradley, D., G. Roth 2007                            | Adapting Thresholding Using the Integral Image

```
