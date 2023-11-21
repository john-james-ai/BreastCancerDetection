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

# Image Generator

```{code-cell} ipython3
:tags: [remove-input]
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import random_noise
```

```{code-cell} ipython3
:tags: [remove-input]
FP_ORIG = "jbook/figures/mammogram.png"
CMAP = 'gray'
```

```{code-cell} ipython3
:tags: [remove-input]
def convert_uint8(img: np.ndarray) -> np.ndarray:
    """Converts floating point array in [0,1] to unit8 in [9,255]"""
    return np.array(255*img, dtype='uint8')
```

## Gaussian Noise

```{code-cell} ipython3
:tags: [remove-input]
orig = cv2.imread(FP_ORIG)
gaussian_1 = random_noise(orig, mode='gaussian', mean=0,var=0.01)
gaussian_2 = random_noise(orig, mode='gaussian', mean=0,var=0.1)
gaussian_3 = random_noise(orig, mode='gaussian', mean=0,var=0.2)

gaussian_1 = convert_uint8(gaussian_1)
gaussian_2 = convert_uint8(gaussian_2)
gaussian_3 = convert_uint8(gaussian_3)
```

```{code-cell} ipython3
:tags: [remove-input]
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10,8.5))
for ax in axes.flat:
    _ = ax.set(xticks=[], yticks=[])

_ = axes[0,0].imshow(orig, cmap=CMAP)
_ = axes[0,0].set_xlabel('Original Image', fontsize=10)

_ = axes[0,1].imshow(gaussian_1, cmap=CMAP)
_ = axes[0,1].set_xlabel(f'Gaussian Noise with mean = 0, variance = {0.01}', fontsize=10)

_ = axes[1,0].imshow(gaussian_2, cmap=CMAP)
_ = axes[1,0].set_xlabel(f'Gaussian Noise with mean = 0, variance = {0.1}', fontsize=10)

_ = axes[1,1].imshow(gaussian_3, cmap=CMAP)
_ = axes[1,1].set_xlabel(f'Gaussian Noise with mean = 0, variance = {0.2}', fontsize=10)

_ = plt.suptitle("Mammogram with Gaussian Noise", fontsize=12)
plt.tight_layout()
plt.show()
```

## Quantization Noise

```{code-cell} ipython3
:tags: [remove-input]
orig = Image.open(FP_ORIG)
quant = orig.quantize(colors=2)
```

```{code-cell} ipython3
:tags: [remove-input]
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,4))
for ax in axes.flat:
    _ = ax.set(xticks=[], yticks=[])

_ = axes[0].imshow(orig, cmap=CMAP)
_ = axes[0].set_xlabel('Original Image', fontsize=10)

_ = axes[1].imshow(quant, cmap=CMAP)
_ = axes[1].set_xlabel('Quantization colors = 2', fontsize=10)

_ = plt.suptitle("Mammogram with Quantization Noise", fontsize=12)
plt.tight_layout()
plt.show()
```

## Speckle Noise

```{code-cell} ipython3
:tags: [remove-input]
orig = cv2.imread(FP_ORIG)
speckle_1 = random_noise(orig, mode='speckle', mean=2,var=0.1)
speckle_2 = random_noise(orig, mode='speckle', mean=4,var=0.25)
speckle_3 = random_noise(orig, mode='speckle', mean=8,var=0.5)

speckle_1 = convert_uint8(speckle_1)
speckle_2 = convert_uint8(speckle_2)
speckle_3 = convert_uint8(speckle_3)
```

```{code-cell} ipython3
:tags: [remove-input]
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10,8.5))
for ax in axes.flat:
    _ = ax.set(xticks=[], yticks=[])

_ = axes[0,0].imshow(orig, cmap=CMAP)
_ = axes[0,0].set_xlabel('Original Image', fontsize=10)

_ = axes[0,1].imshow(speckle_1, cmap=CMAP)
_ = axes[0,1].set_xlabel(f'Speckle Noise with mean = 2, variance = {0.1}', fontsize=10)

_ = axes[1,0].imshow(speckle_2, cmap=CMAP)
_ = axes[1,0].set_xlabel(f'Speckle Noise with mean = 4, variance = {0.25}', fontsize=10)

_ = axes[1,1].imshow(speckle_3, cmap=CMAP)
_ = axes[1,1].set_xlabel(f'Speckle Noise with mean = 8, variance = {0.5}', fontsize=10)

_ = plt.suptitle("Mammogram with Speckle Noise", fontsize=12)
plt.tight_layout()
plt.show()
```

## Salt & Pepper Noise

```{code-cell} ipython3
:tags: [remove-input]
orig = cv2.imread(FP_ORIG)
snp_1 = random_noise(orig, mode='s&p', amount=0.05)
snp_2 = random_noise(orig, mode='s&p', amount=0.1)
snp_3 = random_noise(orig, mode='s&p', amount=0.2)

snp_1 = convert_uint8(snp_1)
snp_2 = convert_uint8(snp_2)
snp_3 = convert_uint8(snp_3)
```

```{code-cell} ipython3
:tags: [remove-input]
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10,8.5))
for ax in axes.flat:
    _ = ax.set(xticks=[], yticks=[])

_ = axes[0,0].imshow(orig, cmap=CMAP)
_ = axes[0,0].set_xlabel('Original Image', fontsize=10)

_ = axes[0,1].imshow(snp_1, cmap=CMAP)
_ = axes[0,1].set_xlabel('Salt & Pepper Noise = 5%', fontsize=10)

_ = axes[1,0].imshow(snp_2, cmap=CMAP)
_ = axes[1,0].set_xlabel('Salt & Pepper Noise = 10%', fontsize=10)

_ = axes[1,1].imshow(snp_3, cmap=CMAP)
_ = axes[1,1].set_xlabel('Salt & Pepper Noise = 20%', fontsize=10)

_ = plt.suptitle("Mammogram with Salt & Pepper Noise", fontsize=12)
plt.tight_layout()
plt.show()
```

## Poisson Noise

```{code-cell} ipython3
:tags: [remove-input]
orig = cv2.imread(FP_ORIG)
poisson = random_noise(orig, mode='poisson')
poisson = convert_uint8(poisson)
```

```{code-cell} ipython3
:tags: [remove-input]
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,4))
for ax in axes.flat:
    _ = ax.set(xticks=[], yticks=[])

_ = axes[0].imshow(orig, cmap=CMAP)
_ = axes[0].set_xlabel('Original Image', fontsize=10)

_ = axes[1].imshow(snp_1, cmap=CMAP)
_ = axes[1].set_xlabel('Poisson Noise', fontsize=10)

_ = plt.suptitle("Mammogram with Poisson Noise", fontsize=12)
plt.tight_layout()
plt.show()
```
