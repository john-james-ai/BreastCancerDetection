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
Various types of artifacts were found on breast mammogram images, as shown below.

```{code-cell} ipython3
import os
import numpy as np
import cv2
from bcd.preprocess.image.threshold import ThresholdAnalyzer, ThresholdManual, ThresholdOTSU, ThresholdTriangle, ThresholdAdaptiveMean,  ThresholdAdaptiveGaussian, ThresholdISOData, ThresholdLi, ThresholdTriangleAnalyzer, ThresholdYen, ThresholdSurveyor
```

```{code-cell} ipython3
img1 = "data/image/1_dev/converted/train/benign/347c2455-cb62-40f8-a173-9e4eb9a21902.png"
img2 = "data/image/1_dev/converted/train/benign/4ed91643-1e06-4b2c-8efb-bc60dd9e0313.png"
img3 = "data/image/1_dev/converted/train/malignant/7dcc12fd-88f0-4048-a6ab-5dd0bd836f08.png"
img4 = "data/image/1_dev/converted/train/malignant/596ef5db-9610-4f13-9c1a-4c411b1d957c.png"
```

```{code-cell} ipython3
img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(img4, cv2.IMREAD_GRAYSCALE)
```

```{code-cell} ipython3
images = (img1,img2,img3,img4)
```

## Manual Thresholding

```{code-cell} ipython3
analyzer = ThresholdAnalyzer()
threshold = ThresholdManual(threshold=10)
_ = analyzer.analyze(images, threshold)
```

## ISOData Thresholding

```{code-cell} ipython3
analyzer = ThresholdAnalyzer(show_histograms=False)
threshold = ThresholdISOData()
_ = analyzer.analyze(images, threshold)
```

## Minimum Entropy Thresholding

```{code-cell} ipython3
analyzer = ThresholdAnalyzer()
threshold = ThresholdLi()
_ = analyzer.analyze(images, threshold)
```

## OTSU's Method

```{code-cell} ipython3
analyzer = ThresholdAnalyzer(show_histograms=False)
threshold = ThresholdOTSU()
_ = analyzer.analyze(images, threshold)
```

## Triangle Method

```{code-cell} ipython3
analyzer = ThresholdTriangleAnalyzer(show_masked_images=False)
threshold = ThresholdTriangle()
_ = analyzer.analyze(images, threshold)
```

## Adaptive Mean Thresholding

```{code-cell} ipython3
analyzer = ThresholdAnalyzer(show_masks=False)
threshold = ThresholdAdaptiveMean()
_ = analyzer.analyze(images, threshold)
```

```{code-cell} ipython3
analyzer = ThresholdAnalyzer(show_masks=False)
threshold = ThresholdAdaptiveMean(blocksize=21)
_ = analyzer.analyze(images, threshold)
```

## Adaptive Gaussian Thresholding

```{code-cell} ipython3
analyzer = ThresholdAnalyzer(show_masks=False)
threshold = ThresholdAdaptiveGaussian(blocksize=5)
_ = analyzer.analyze(images, threshold)
```

```{code-cell} ipython3
analyzer = ThresholdAnalyzer(show_masks=False)
threshold = ThresholdAdaptiveGaussian(blocksize=21)
_ = analyzer.analyze(images, threshold)
```

```{code-cell} ipython3
analyzer = ThresholdAnalyzer(show_masks=False)
threshold = ThresholdYen()
_ = analyzer.analyze(images, threshold)
```

```{code-cell} ipython3
surveyor = ThresholdSurveyor()
_ = surveyor.tryall(image=img1)
```
