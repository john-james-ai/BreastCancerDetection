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

# Frequency Domain

```{code-cell} ipython3
:tags: [hide-cell, remove-output]
import cv2
import matplotlib.pyplot as plt
import numpy as np
from bcd.utils.image import convert_uint8

FP_WHITE = "../../figures/frequency_white.jpg"
FP_HLINE = "../../figures/frequency_hline.png"
FP_VLINE = "../../figures/frequency_vline.png"

def get_image(fp: str, size: int = 200):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (size,size))
    #return convert_uint8(img=img, asfloat=True, invert=True)

def transform_image(img):
    # Compute fourier transformation
    img_fft = np.fft.fft2(img)
    # Shift the zero-frequency to the center
    img_shifted = np.fft.fftshift(img_fft)
    # Compute amplitude
    img_amp = np.abs(img_shifted)
    img_amp = np.clip(img_amp, a_min=0, a_max=50000)
    return img_amp

def plot_images(img1, img2, size=(12,6)):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=size)
    _ = axes[0].imshow(img1, cmap='gray')
    _ = axes[1].imshow(img2, cmap='gray')

    for i in range(2):
        _ = axes[i].set_xticks([])
        _ = axes[i].set_yticks([])
    plt.tight_layout()
    return fig

```

## White

```{code-cell} ipython3
# obtain the original images
img = get_image(fp=FP_WHITE)
img_fft = transform_image(img)
fig = plot_images(img1=img,img2=img_fft)
```

## Horizontal Lines

```{code-cell} ipython3
# obtain the original images
img = get_image(fp=FP_HLINE)
img_fft = transform_image(img)
fig = plot_images(img1=img,img2=img_fft)
```

## Vertical Lines

```{code-cell} ipython3
# obtain the original images
img = get_image(fp=FP_VLINE)
img_fft = transform_image(img)
fig = plot_images(img1=img,img2=img_fft)
```

## Diagonal Lines

```{code-cell} ipython3
# obtain the original images
img = get_image(fp=FP_DLINE)
img_fft = transform_image(img)
fig = plot_images(img1=img,img2=img_fft)
```

## Mammography

```{code-cell} ipython3
# obtain the original images
img = get_image(fp=FP_MMG)
img_fft = transform_image(img)
fig = plot_images(img1=img,img2=img_fft)
```
