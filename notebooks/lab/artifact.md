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

# Artifact Removal
Various types of artifacts were found on breast mammogram images, as shown below.

```{code-cell} ipython3
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from bcd.utils.image import grayscale
```

```{code-cell} ipython3
img1 = "data/image/2_exp/train/benign/2a44122c-f831-4220-95a8-408bcafcf2ce.png"
img2 = "data/image/2_exp/train/benign/3f72309d-7cd9-4e30-ae81-073adb541bcd.png"
img3 = "data/image/2_exp/train/benign/97556037-b959-4395-830b-380dcac2d58e.png"
img4 = "data/image/2_exp/train/malignant/6cdf46d8-596b-47ab-a428-c8769733c93c.png"
```

```{code-cell} ipython3
img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(img4, cv2.IMREAD_GRAYSCALE)
```

```{code-cell} ipython3
img1 = grayscale(img1)
img2 = grayscale(img2)
img3 = grayscale(img3)
img4 = grayscale(img4)
```

```{code-cell} ipython3
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,6))
_ = axes[0].imshow(img1, cmap='gray', aspect='auto')
_ = axes[1].imshow(img2, cmap='gray',aspect='auto')
_ = axes[2].imshow(img3, cmap='gray',aspect='auto')
_ = axes[3].imshow(img4, cmap='gray',aspect='auto')
plt.tight_layout()
plt.show()
```

## Binary Masking

```{code-cell} ipython3
r, bm1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
r, bm2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
r, bm3 = cv2.threshold(img3, 127, 255, cv2.THRESH_BINARY)
r, bm4 = cv2.threshold(img4, 127, 255, cv2.THRESH_BINARY)
```

```{code-cell} ipython3
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12,6))
_ = axes[0].imshow(bm1, cmap='gray', aspect='auto')
_ = axes[1].imshow(bm2, cmap='gray',aspect='auto')
_ = axes[2].imshow(bm3, cmap='gray',aspect='auto')
_ = axes[3].imshow(bm4, cmap='gray',aspect='auto')
plt.tight_layout()

```
