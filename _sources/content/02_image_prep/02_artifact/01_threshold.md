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
:tags: [remove-input]

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

For this effort, eight automated thresholding techniques {numref}`auto-thresh-tbl` were selected based on the intrinsic properties of the CBIS-DDSM dataset.

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

### Triangle Method

The Triangle method was proposed in 1977 as a method for automatically detecting and counting sister chromatid exchanges in human chromosomes {cite}`zackAutomaticMeasurementSister1977`.

```{figure} ../../../figures/triangle_zack.png
---
name: triangle
---
Triangle Thresholding Method
```

{numref}`triangle` was taken from the original paper and geometrically depicts the triangle threshold method. The threshold is selected by first normalizing the dynamic range and the counts of the intensity histogram to values in [0,1]. A line is then constructed between the histogram peak and the tip of the long tail. Point A is selected at the base of the histogram bin that has the maximum perpendicular distance from the ‘peak-to-tip’ line. The threshold is set to A $+\delta \ge 0$.

The triangle method assumes pixel intensity distributions with a maximum peak near one end of the histogram and searches for thresholds towards the other end. Hence, this method is particularly well suited for images with highly skewed pixel intensity distributions with a single dominant peak and one or more weak peaks.

### ISODATA Method

The ISODATA method is an iterative approach based upon the Iterative Self-Organizing Data Analysis algorithm (ISODATA) algorithm {cite}`ridlerPictureThresholdingUsing1978` and is commonly used in biomedical imaging.

In general, it first assigns an initial threshold $T$, usually the average of the pixel intensities in the image. The second step classifies each pixel to the closest class. In the third step, the mean values $\mu_1$ and $\mu_2$ for each class are estimated using a Gaussian distribution. Next, a new threshold is selected as the average of $\mu_1$ and $\mu_2$ {cite}`el-zaartImagesThresholdingUsing2010`.  Steps two and three are repeated until the change in threshold is less than a predesignated parameter.

An advantage of the ISOData method is that one doesn't need to know much about the properties of the image in advance.  On the other hand, the ISODATA algorithm assumes a symmetric Gaussian pixel distribution and may not perform well with images with a non-symmetric histogram. It can also be time-consuming and computationally expensive for large images.

### OTSU's Method

Otsu's Method {cite}`otsuThresholdSelectionMethod1979`  is a very popular image thresholding technique that separates images into two classes by minimizing the intra-class variance, denoted as $\sigma^2_w(t)$. Let’s build this from the bottom up.
The Otsu method initializes the threshold $t=0$, then computes the image histogram and probabilities of each intensity level $p(i)$ denoted as:

```{math}
:label: otsu_prob
p(i) = \frac{n_i}{n}
```

where  $i$ is a pixel intensity level, $n_i$ is the number of pixels with intensity level $i$, and $n$ is the total number of pixels.
Next, the image is split into two classes: $C_0$  containing all pixels with intensity in the range [0,t), and $C_1$ which contains all pixels with intensity in the range [t, L], where L is the number of bins in the histogram.
The next step is to obtain weights for $C_0, and $C_1$, denoted $w_0(t)$ and $w_1(t)$, respectively.

```{math}
:label: otsu_weights
w_0(t) = \sum_{i=0}^{t-1} p(i),
w_1(t) = \sum_{i=t}^{L-1} p(i),
```

Next, the means for $C_0$ and $C_1$, denoted $\mu_0$ and $\mu_1$, respectively:

```{math}
:label: otsu_means
\mu_0(t) = \frac{\sum_{i=0}^{t-1} ip(i)}{w_0{t}},
\mu_1(t) = \frac{\sum_{i=t}^{L-1} ip(i)}{w_1{t}},
```

Now, we compute the weighted variance of each class, denoted as $\sigma_0^2$, and $\sigma_1^2$:

```{math}
:label: otsu_cvar
\sigma_0^2(t) = \sum{i=0}^{t-1}[i-\mu_0(t)]^2 \frac{p(i)}{w_0(t)}
\sigma_1^2(t) =  \sum{i=t}^{L-1}[i-\mu_1(t)]^2 \frac{p(i)}{w_1(t)}
```

From {cite}`otsuThresholdSelectionMethod1979`  we have two options to find the threshold. The first is to minimize intra-class variance as follows:

```{math}
:label: otsu_ivar
\sigma_w^2(t) =w_0(t)\sigma^2_0(t)+w_1(t)\sigma^2_1(t).
```

The second method is to maximize inter-class variance as follows:

```{math}
:label: otsu_bvar
\sigma_b^2(t) =w_0(t)w_1(t)[\mu_0(t)-\mu_1(t)]^2
```

Hence, the general algorithm for minimizing the intra-class variance is given by:

1. Initialize threshold $t=0$
2. Calculate the histogram and intensity level probabilities $p(i)$
3. Initialize $w_0(0)$ and $w_1(0)$
4. FOR t in range (0,L-1)
   - Update values of $w_i,\mu_i,$ where $w_i$ is the weighted probability and $\mu_i$ is the mean of class $i$.
   - Calculate the intra-class variance value $\sigma_w^2(t)$
5. The final threshold is that which minimizes $\sigma_w^2(t)$

The Otsu method performs well when the image histogram is bimodal with a deep and sharp valley between two peaks {cite}`kittlerThresholdSelectionUsing1985`.  However, the method isn’t the best choice in the presence of heavy noise,  large variances in lighting, or when intra-class variance is larger than inter-class variance.  In such cases, adaptations have been proposed such as the  Kittler-Illingworth method {cite}`kittlerMinimumErrorThresholding1986`.

## Li's Minimum Cross-Entropy Method

```{code-cell} ipython3
:tags: [remove-input, remove-output]

analyzer = ThresholdAnalyzer(show_histograms=False)
threshold = ThresholdLi()
fig = analyzer.analyze(images=images, threshold=threshold)

glue("threshold_li", fig)
```

```{glue:figure} threshold_li
---
align: center
name: threshold_li_fig
---
Li's Minimum Cross-Entropy Threshold Segmentation Method. (a) through (d) are the original images, (e) through (h) are the binary masks, and (i) through (l) are the segmented images.
```

## Yen's Multilevel Thresholding Method

```{code-cell} ipython3
:tags: [remove-input, remove-output]

analyzer = ThresholdAnalyzer(show_histograms=False)
threshold = ThresholdYen()
fig = analyzer.analyze(images=images, threshold=threshold)

glue("threshold_yen", fig)
```

```{glue:figure} threshold_yen
---
align: center
name: threshold_yen_fig
---
Yen's Multilevel Threshold Segmentation Method. (a) through (d) are the original images, (e) through (h) are the binary masks, and (i) through (l) are the segmented images.
```

## Adaptive Mean Thresholding Method

```{code-cell} ipython3
:tags: [remove-input, remove-output]

analyzer = ThresholdAnalyzer(show_histograms=False)
threshold = ThresholdAdaptiveMean()
fig = analyzer.analyze(images=images, threshold=threshold)

glue("threshold_local_mean", fig)
```

```{glue:figure} threshold_local_mean
---
align: center
name: threshold_local_mean_fig
---
Adaptive Mean Threshold Segmentation Method. (a) through (d) are the original images, (e) through (h) are the binary masks, and (i) through (l) are the segmented images.
```

## Adaptive Gaussian Thresholding Method

```{code-cell} ipython3
:tags: [remove-input, remove-output]

analyzer = ThresholdAnalyzer(show_histograms=False)
threshold = ThresholdAdaptiveGaussian()
fig = analyzer.analyze(images=images, threshold=threshold)

glue("threshold_local_gaussian", fig)
```

```{glue:figure} threshold_local_gaussian
---
align: center
name: threshold_local_gaussian_fig
---
Adaptive Gaussian Threshold Segmentation Method. (a) through (d) are the original images, (e) through (h) are the binary masks, and (i) through (l) are the segmented images.
```
