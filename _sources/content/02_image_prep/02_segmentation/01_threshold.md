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

Our first step in the segmentation process is to separate the foreground of the mammogram containing the breast region of interest (ROI) from the background.  The result of this *binarization* process is a binary image or mask that can be used to isolate ROIs, label structures within the image, detect edges and contours, classify abnormalities, and perform other tasks across the image analysis workflow.

Binarization of the image can be accomplished using a range of statistical approaches and sophisticated machine-learning classification methods that group the pixels of an image into different classes. However, the simplest approach would be to set a pixel value cut-off point, or *threshold*, that separates groups of pixel intensities from each other.

Given a gray-level digital mammogram $I$, that can take $K$ possible gray levels $0,1,2,...,K-1$, we define an integer threshold $T \in {0,1,2,...,K-1}$. The process of thresholding simply compares each pixel value $I(n)$ in $I$ to the threshold $T$ to determine the corresponding pixel value $B(n)$ in an output binary image $B$ as follows:

```{math}
:label: global_threshold
B(n) = \begin{cases}
1 & \text{if } I(n) > T \\
0 & \text{if } I(n) \le T \\
\end{cases}
```

In other words, the pixel value in the binary image is (typically) set to 0 if the corresponding pixel value in the input image is less than or equal to the threshold. Otherwise, the corresponding binary image pixel value is set to 1. In practice; however, many software packages create gray-scale binary images with values of 0 and 255 for visualization purposes. In such cases, a segmented image is produced by applying a bitwise_and operation on the binary and input image.

Image binarization using pixel intensity thresholds is justified in digital mammography where pixel intensity is the parameter that most directly relates to the spatial characteristics of the structures within a mammogram. Hence, binarization using threshold-based segmentation is often a critical first step in many digital mammography image analysis and preprocessing workflows.

To illustrate, let’s create some binary masks for our test images using a manually set threshold $T=10$.

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

img1 = "data/image/1_dev/converted/train/benign/347c2455-cb62-40f8-a173-9e4eb9a21902.png"
img2 = "data/image/1_dev/converted/train/benign/4ed91643-1e06-4b2c-8efb-bc60dd9e0313.png"
img3 = "data/image/1_dev/converted/train/malignant/7dcc12fd-88f0-4048-a6ab-5dd0bd836f08.png"
img4 = "data/image/1_dev/converted/train/malignant/596ef5db-9610-4f13-9c1a-4c411b1d957c.png"

img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(img4, cv2.IMREAD_GRAYSCALE)

images = (img1,img2,img3,img4)

analyzer = ThresholdAnalyzer()
threshold = ThresholdManual(threshold=10)
fig = analyzer.analyze(images=images, threshold=threshold)

glue("threshold_manual_10", fig)
```

```{glue:figure} threshold_manual_10
---
align: center
name: threshold_manual_10_fig
---
Manual Threshold-Based Segmentation with $T=10$
```

In {numref}`threshold_manual_10_fig`, we have our four randomly selected images, the associated binary masks, and the output images. At threshold $T=10$, we have a clear separation between foreground and background; however, we have little to no artifact suppression.

Let’s examine the effect of increasing the threshold to $T=100$.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

analyzer = ThresholdAnalyzer()
threshold = ThresholdManual(threshold=100)
fig = analyzer.analyze(images=images, threshold=threshold)

glue("threshold_manual_100", fig)
```

```{glue:figure} threshold_manual_100
---
align: center
name: threshold_manual_100_fig
---
Manual Threshold-Based Segmentation with $T=100$
```

{numref}`threshold_manual_10_fig` and {numref}`threshold_manual_100_fig` illustrate the first key takeaway of this section.

`````{admonition} In threshold segmentation, the choice of threshold is crucial!
:class: tip
Different thresholds may yield dramatically different segmentation results.
`````

A threshold that is too low tends to produce over-segmentation, combining distinct objects into single structures and failing to separate artifacts from regions of interest as shown in {numref}`threshold_manual_10_fig`  (i)-(l). On the other hand, a high threshold can make objects smaller, resulting in loss of information as structures of interest are designated to the background as evidenced in {numref}`threshold_manual_100_fig`  (i)-(l).

So, what are the principled ways by which an appropriate threshold is selected? This leads to our second key takeaway.

`````{admonition} Choosing a threshold manually should be avoided, when possible!
:class: tip
Manual thresholding is inefficient, irreproducible, and a huge source of user bias.
`````

...rant in 3,…2,…

Selecting thresholds manually is tedious, time-consuming, and a huge source of user bias. It is based upon human perception of what information should be extracted from the image, leading to high intra- and inter-user variability; further compounded by the inherent variability in digital mammography. One fixed threshold will not extract similar features from different images. Manual thresholding has little to no reproducibility and it is incompatible with automatic, image-driven thresholding that is based on image-intrinsic properties and not on subjective real-time user decisions.

So, what is the alternative?

## Automated Thresholding

Image processing literature is replete with automatic thresholding algorithms of various types, each based upon a vast array of image-intrinsic properties.  Automated thresholding has several benefits vis-à-vis manual thresholding.

- **Bias-Free Thresholding**: No user bias is introduced during *thresholding*,
- **Objective**: Thresholds are objectively determined and image-specific,
- **Reproducible**: They are reproducible, in that a parameterized algorithm will always produce the same binarization result for a given image,
- **Efficient**: They are fast, computationally efficient, and easily automated in image analysis and preprocessing workflows.

Notwithstanding, automated thresholding presents certain challenges for the practitioner.

- **No Free Lunch** {cite}`wolpertNoFreeLunch1997a`: There is no universally superior automated thresholding algorithm that performs equally well for all biomedical images, modalities, contexts, and needs.
- **Algorithm Selection Bias**: Algorithm selection is a subjective user decision based upon experience, human perception, and prior expectations concerning the information to be extracted from an image. For instance, an algorithm selected by the current expert pathologist may eliminate a structure later determined to be a critical indicator.
- **Prior Knowledge**: Practitioners may lack the domain expertise required to effectively characterize the performance of the algorithms under evaluation. Algorithm selection is often based upon a priori expectation of visual features, structures, and information to be extracted during the segmentation process.

Selecting an appropriate automated thresholding algorithm from the growing space of candidate solutions is an increasingly challenging endeavor, guided by the intended use of the extracted information, the questions to be answered, the quality, format, and modality of the imaging content, and a working understanding of candidate algorithm performance characteristics.

### Automated Thresholding Algorithm Space

Sezgin and Sankur {cite}`sankurSurveyImageThresholding2004` cast the space of automated thresholding techniques as follows:

- **Histogram shape-based methods** that analyze, for instance, the peaks, valleys, and curvatures of smoothed histograms.
- **Clustering-based methods** cluster the gray-level samples into background and foreground. Alternatively, the image is modeled as a mixture of two Gaussians.
- **Entropy-based methods** use the entropy of the foreground and background regions, the cross-entropy between the original and binarized image, etc.
- **Object attribute-based methods** that analyze the similarity between the gray-level and the binarized images, such as fuzzy shape similarity, edge coincidence, etc.
- **The spatial methods** use higher-order probability distribution and/or correlation between pixels
- **Local methods** adapt the threshold value on each pixel to the local image characteristics.

For the taxonomist, Sezgin’s framework is not mutually exclusive and collectively exhaustive (MECE). For instance, Otsu’s Method {cite}`otsuThresholdSelectionMethod1979` can be categorized as both a histogram shape-based method and a clustering-based method.

### Automated Threshold Methods

Our candidate space will be comprised of the following ({numref}`auto-thresh-tbl`) histogram-based, entropy-based, spatial-based and local-based threshold methods.

```{table} Automated Threshold Methods
:name: auto-thresh-tbl


| Type                    | Method                               | Author(s)                                            | Publication                                                  |
|-------------------------|--------------------------------------|------------------------------------------------------|--------------------------------------------------------------|
| Histogram-Based         | Triangle Method                      | Zack, G. W., Rogers, W. E. and Latt, S. A., 1977,    | Automatic Measurement of Sister Chromatid Exchange Frequency |
| Cluster-Based           | ISOData Method                       | Ridler, TW & Calvard, S (1978)                       | Picture thresholding using an iterative selection method     |
| Histogram/Cluster-Based | Otsu's Method                        | Nobuyuki Otsu (1979)                                 | A threshold selection method from gray-level histograms      |
| Entropy-Based           | Li's Minimum Cross Entropy Method    | Li C.H. and Lee C.K. (1993)                          | Minimum Cross Entropy Thresholding                           |
| Spatial-Based           | Yen's Multilevel Thresholding Method | Jui-Cheng Yen, Fu-Juay Chang and Shyang Chang (1995) | A new criterion for automatic multilevel thresholding        |
| Local                   | Adaptive Gaussian Method             | Bradley, D., G. Roth 2007                            | Adapting Thresholding Using the Integral Image               |
|                         | Adaptive Mean Method                 | Bradley, D., G. Roth 2007                            | Adapting Thresholding Using the Integral Image               |                                    |

```

Next, we review how each method works, state any assumptions, characterize strengths and limitations, and visualize and assess the segmentation results for selected cases. Finally, an automated thresholding algorithm will be selected based on relative segmentation performance on our test cases.

#### Triangle Method

The Triangle method was proposed in 1977 as a method for automatically detecting and counting sister chromatid exchanges in human chromosomes {cite}`zackAutomaticMeasurementSister1977`. It is particularly well suited for images that have a pixel intensity distribution dominated by a single peak and a long tail.

```{figure} ../../../figures/triangle_zack.png
---
name: triangle
---
Triangle Thresholding Method
```

{numref}`triangle` was taken from the original paper and geometrically depicts the triangle threshold method. The threshold is selected by first normalizing the dynamic range and the counts of the intensity histogram to values in [0,1]. A line is then constructed between the histogram peak and the tip of the long tail. Point A is selected at the base of the histogram bin that has the maximum perpendicular distance from the ‘peak-to-tip’ line. The threshold is set to A $+\delta \ge 0$.

The triangle method assumes pixel intensity distributions with a maximum peak near one end of the histogram and searches for thresholds towards the other end. Hence, this method is particularly well suited for images with highly skewed pixel intensity distributions with a single dominant peak and one or more weak peaks.

This method was applied to four test images of varying breast densities, contrast, abnormalities, and diagnoses. {numref}`threshold_triangle_fig` shows the original images (a)-(d), the binary images (e)-(h), the segmentation results (i)-(l), and the triangle histograms with thresholds annotated (m)-(p).

```{code-cell} ipython3
:tags: [remove-input, remove-output]

analyzer = ThresholdTriangleAnalyzer()
threshold = ThresholdTriangle()
fig = analyzer.analyze(images=images, threshold=threshold)

glue("threshold_triangle", fig)
```

```{glue:figure} threshold_triangle
---
align: center
name: threshold_triangle_fig
---
Triangle Threshold Segmentation Method. (a) through (d) are the original images, (e) through (h) are the binary masks, (i) through (l) are the segmented images and the normalized histograms and thresholds are presented at (m) through (p)
```

Several observations can be made. First, all images had the same threshold $T=2$, despite varying levels of contrast, illumination, and breast density. Second, at $T=2$, we have little to no artifact removal as their pixel intensities are not distinguished from other foreground structures. Overall, the algorithm effectively distinguished the breast tissue from the background with no apparent loss of information.

#### ISOData Method

```{code-cell} ipython3
:tags: [remove-input, remove-output]

analyzer = ThresholdAnalyzer()
threshold = ThresholdISOData()
fig = analyzer.analyze(images=images, threshold=threshold)

glue("threshold_isodata", fig)
```

```{glue:figure} threshold_isodata
---
align: center
name: threshold_isodata_fig
---
ISOData Threshold Segmentation Method. (a) through (d) are the original images, (e) through (h) are the binary masks, and (i) through (l) are the segmented images.
```

#### OTSU's Method

```{code-cell} ipython3
:tags: [remove-input, remove-output]

analyzer = ThresholdAnalyzer()
threshold = ThresholdOTSU()
fig = analyzer.analyze(images=images, threshold=threshold)

glue("threshold_otsu", fig)
```

```{glue:figure} threshold_otsu
---
align: center
name: threshold_otsu_fig
---
OTSU's Threshold Segmentation Method. (a) through (d) are the original images, (e) through (h) are the binary masks, and (i) through (l) are the segmented images.
```

#### Li's Minimum Cross-Entropy Method

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

#### Yen's Multilevel Thresholding Method

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

#### Adaptive Mean Thresholding Method

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

#### Adaptive Gaussian Thresholding Method

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
