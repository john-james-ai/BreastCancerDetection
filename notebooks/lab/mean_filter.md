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

# Mean Filter

```{code-cell} ipython3
:tags: [hide-cell, remove-output]

FP_ORIG = "jbook/figures/mammogram.png"
CMAP = 'gray'

# Obtain the source image
orig = cv2.imread(FP_ORIG)

# Add random Gaussian noise with zero mean and variance of 0.1
img_gaussian = random_noise(orig, mode='gaussian', mean=0,var=0.1)
img_gaussian = convert_uint8(img_gaussian)

# Apply the 3x3 mean filter kernel
img_filtered = cv2.blur(img_gaussian, (3,3))

# Subtract the filtered
img_noise = img_gaussian - img_filtered

# Compute histograms
img_gaussian_hist = cv2.calcHist([img_gaussian], [0], None, [256], [0,256])
img_filtered_hist = cv2.calcHist([img_filtered], [0], None, [256], [0,256])
img_noise_hist = cv2.calcHist([img_noise], [0], None, [256], [0,256])

# Create Figure object
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,6), dpi=200)

# Show images
_ = ax[0,0].imshow(img_gaussian, cmap=CMAP)
_ = ax[0,0].set_xlabel('(a) Original Image', fontsize=10)

_ = ax[0,1].imshow(img_filtered, cmap=CMAP)
_ = ax[0,1].set_xlabel('(b) Mean Filtered Image', fontsize=10)

_ = ax[0,2].imshow(img_noise, cmap=CMAP)
_ = ax[0,2].set_xlabel('(c) Image Noise', fontsize=10)

# Show histograms
_ = ax[1,0].plot(img_gaussian_hist)
_ = ax[1,0].set_xlabel("(d) Original Image Histogram", fontsize=10)

_ = ax[1,1].plot(img_filtered_hist)
_ = ax[1,1].set_xlabel("(e) Mean Filtered Image Histogram", fontsize=10)

_ = ax[1,2].plot(img_noise_hist)
_ = ax[1,2].set_xlabel("(f) Image Noise Histogram", fontsize=10)

plt.tight_layout()
glue("mean_filter_fig", fig)
```

```{glue:figure} mean_filter_fig
---
align: center
name: mean_filter_figure
---
Applying a 3×3 mean filter makes the image smoother, which is evident upon close examination of the features in the region of interest. The histograms illuminate the distribution of the signal vis-a-vis the noise. As (f) illustrates, most of the noise was in the brighter regions of the image.
```