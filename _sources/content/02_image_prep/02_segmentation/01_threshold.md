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

This *binarization* of the image can be accomplished using a range of statistical approaches and sophisticated machine-learning classification methods that group pixels of an image into different classes. However, the simplest approach would be to set a pixel value cut-off point, or *threshold*, that separates groups of pixel intensities from each other. From a grayscale mammogram, a binary image or mask is created whereby all pixel values less than or equal to the threshold are set to 0 and all other pixel values are set to 1  [^binary].

[^binary]: WLOG some software packages equivalently set pixel values above the threshold to 255, instead of 1. Converting between these representations can be achieved via normalization of the pixel values.

Image binarization using pixel intensity thresholds is suitable in digital mammography where pixel intensity is the parameter that most directly relates to the spatial characteristics of the structures within a mammogram. Hence, binarization using threshold-based segmentation is often a critical first step in many biomedical image analysis workflows.

The taxonomy of threshold-based segmentation techniques can be roughly characterized as global threshold segmentation and adaptive local threshold segmentation.

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

from bcd.preprocess.image.analysis.threshold import (
    ThresholdLi, ThresholdISOData, ThresholdTriangle, ThresholdOTSU, ThresholdAdaptiveMean, ThresholdAdaptiveGaussian, TryAllThresholds, ThresholdManual
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

analysis = ThresholdManual(threshold=10)
fig = analysis.analyze(images=images)

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

analysis = ThresholdManual(threshold=100)
fig = analysis.analyze(images=images)

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

...rant inbound in 3,…2,…

Selecting thresholds manually is tedious, time-consuming, and a huge source of user bias. It is based upon human perception of what information should be extracted from the image, leading to high intra- and inter-user variability; further compounded by the inherent variability in digital mammography. One fixed threshold will not extract similar features from different images. Manual thresholding has little to no reproducibility and it is incompatible with automatic, image-driven thresholding that is based on image-intrinsic properties and not on subjective real-time user decisions.

So, what is the alternative?

## Automatic Thresholding

Image processing literature is replete with automatic thresholding algorithms that are based on a vast array of image-intrinsic properties. What are the benefits of automatic thresholding?

- No user bias is introduced during *thresholding*,
- Thresholds are objectively determined and image-specific,
- They are reproducible, in that an algorithm will always produce the same binarization result for a given image,
- They are fast, computationally efficient, and easily automated in image analysis and preprocessing workflows.

Still, automatic thresholding algorithms are not a universal, purely objective, and bias-free remedy to the threshold segmentation challenges in biomedical imaging and other applications.

### The Universally Superior Automatic Thresholding Algorithm

Selecting an automatic thresholding algorithm from among the space of candidate algorithms can be described in terms of Wolpert and Macreary’s {numref}`lunch`.

```{math}
:label: lunch
\sum_f P(d^y_m|f,m,a_1) = \sum_f P(d^y_m|f,m,a_2),
```

{numref}`lunch` was derived within the context of machine learning algorithms {cite}`wolpertNoFreeLunch1997a`, and classes of objective functions, and states that for any pair of algorithms, the probability distribution of results over the domain of objective functions is independent of the algorithm and identically distributed. In other words, “any two algorithms are equivalent when their performance is averaged across all possible problems.”{cite}`wolpertCoevolutionaryFreeLunches2005`

The so-called “No Free Lunch Theorem for Optimization #1” (NFL) {cite}`wolpertNoFreeLunch1997a` explicitly demonstrates, under certain conditions [^nfl], that no algorithm performs well on all classes of problems within the domain. For instance, an automatic thresholding algorithm that performs well on average for mammograms with bimodal pixel value distributions, necessarily does worse on average over the remaining classes and modalities – a problem widely documented in literature surveys {cite}`al-bayatiMammogramImagesThresholding2013` {cite}`sankurSurveyImageThresholding2004`  {cite}`niuResearchAnalysisThreshold2019`.

[^nfl]: NLF holds if and only if the distribution on objective functions is invariant under the permutation of the space of candidate algorithms{cite}`Streeter2003TwoBC`  {cite}`englishNoMoreLunch2004`. Although this condition is theoretically possible, some have argued that it doesn’t necessarily hold in practice.

So, no universally superior automatic threshold algorithm has been invented that will perform well for all images, modalities, conditions, and needs.


### Automatic Thresholding Algorithm Selection is Biased

Wait. Didn’t we just cast automatic thresholding as based upon image-intrinsic properties and free of user bias during thresholding? Yes, but this doesn’t mean that *algorithm selection* is rigidly objective and bias-free.
Just as the choice of a manual threshold in [0,256] is based upon user experience, human perception, problem characteristics, and prior expectations concerning the information to be extracted from an image, so too is the choice of an algorithm from the increasing candidate space of thresholding algorithms.
On one hand, this is a reasonable basis upon which an algorithm can be selected. Indeed, the NFL theorem further establishes that the probability distribution of results $P(f)$ is uniform , ,that the probability distribution of performance over a class of that selection of an algorithm is only justified that without knowledge of problem-specific characteristics problem specific knowledge

