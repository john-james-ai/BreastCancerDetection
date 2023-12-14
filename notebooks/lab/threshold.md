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
:tags: [remove-input, remove-output]

import os
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
```

```{code-cell} ipython3
images = (img1,img2,img3,img4)
```

```{code-cell} ipython3
analysis = ThresholdManual()
analysis.analyze(images=images)
```

```{code-cell} ipython3
analysis = ThresholdManual(threshold=30)
analysis.analyze(images=images)
```

```{code-cell} ipython3
analysis = ThresholdMean()
analysis.analyze(images=images)
```

```{code-cell} ipython3
analysis = ThresholdLi()
analysis.analyze(images=images)
```

```{code-cell} ipython3
analysis = ThresholdYen()
analysis.analyze(images=images)
```

```{code-cell} ipython3
analysis = ThresholdTriangle()
analysis.analyze(images=images)
```

```{code-cell} ipython3
analysis = ThresholdOTSU()
analysis.analyze(images=images)
```

```{code-cell} ipython3
analysis = ThresholdAdaptiveMean()
analysis.analyze(images=images, blockSize=11, C=2)
```

```{code-cell} ipython3
analysis = ThresholdAdaptiveMean()
analysis.analyze(images=images, blockSize=21, C=2)
```

```{code-cell} ipython3
analysis = ThresholdISOData()
analysis.analyze(images=images)
```

```{code-cell} ipython3
analysis = ThresholdMinimum()
analysis.analyze(images=images)
```

```{code-cell} ipython3
analysis = TryAllThresholds()
analysis.try_all(image=img4)
```
