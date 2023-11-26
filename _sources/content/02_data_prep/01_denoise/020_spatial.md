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
# Spatial Domain Filtering

```{code-cell}
:tags: [hide-cell]
import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../../..")))
import cv2
from matplotlib import pyplot as plt
from myst_nb import glue
import numpy as np
from skimage.util import random_noise

from bcd.preprocess.image.denoise.analyze import MeanFilterAnalyzer, GaussianFilterAnalyzer, MedianFilterAnalyzer, BilateralFilterAnalyzer, NLMeansFilterAnalyzer
```

In spatial domain filtering, the value of a pixel is based upon both itself and the values of the surrounding pixels. Specifically, the output pixel value results from an algorithm that is applied to the values of the pixels in the neighborhood of a corresponding input pixel. Spatial domain filters are classified into two types: linear filters and non-linear filters.

## Linear Filters

Linear filters are those in which the value of an output pixel is a linear combination of the values of the pixels in the input pixel’s neighborhood.  The filtering process is one of sliding a kernel, or mask of weights, along the image and performing a ‘multiple and accumulate’ operation (convolution) on the pixels covered by the mask. The effect is that the image pixel at the center of the mask is set to the weighted average of all the pixels in its *neighborhood*. The shape, and weights of the neighborhood are given by the kernel.
Examples of linear filters include the mean filter and the Wiener filter.

### Mean Filter

Most commonly used to reduce additive Gaussian noise, the mean filter is a simple, intuitive, and easy to implement, filter of the linear class. It’s based on the idea that random noise consists of “sharp” transitions in the intensity of the signal. Mean filtering simply replaces each pixel value with the average value of the intensities in its neighborhood. By doing so, the “sharp” transitions in the intensities are reduced.

The mean filter is based upon the notion of a m x n kernel or matrix, typically of size 3 x 3, which defines the shape and size of the neighborhood to be sampled when computing the mean average intensities.  {numref}`kernel` illustrates a 3 x 3 kernel.

```{figure} ../../../figures/kernel.png
---
name: kernel
---
3 x 3 Mean Filter Kernel
```

The mean filter works by convolving the kernel over the image as follows. Let w(x,y) represent a set of coordinates in a  m x n rectangular window in the image, centered at point (x,y). The mean filtering process computes the average value of the corrupted image g(x,y) in the area defined by w(x,y). The value of the restored image at any point (x,y) is:

```{math}
:label: mean_filter
^\hat{f}(x,y) = \frac{1}{mn}\displaystyle\sum_{(s,t)\in w(x,y)} g(s,t)
```

```{admonition} Kernel Coefficients
Note that the coefficients for the 3x3 kernel are 1 as opposed to 1/9. It is computationally more efficient to have coefficients valued at 1. Then, the normalization constant,  $\frac{1}{mn}$, is applied at the end.
```

The process of convolving with a 3x3 mean filter is as follows:
![MeanFilter](../../../figures/gif/02_mean_filter.gif)

{numref}`mean_gaussian_characteristics_fig` illustrates the results of a 3x3 mean filter kernel on a mammogram image degraded with Gaussian noise.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = MeanFilterAnalyzer()
analyzer.add_gaussian_noise(var=0.2)
fig = analyzer.analyze()
glue("mean_gaussian_characteristics", fig)
```

```{glue:figure} mean_gaussian_characteristics
---
align: center
name: mean_gaussian_characteristics_fig
---
Mean Filter Performance Characteristics with Gaussian Noise
```

As shown in {numref}`mean_gaussian_characteristics_fig`, applying a 3×3 mean filter makes the image smoother, which is evident upon close examination of the features in the region of interest. The histograms illuminate the distribution of the signal vis-a-vis the noise. As (f) illustrates, most of the noise was in the brighter regions of the image.

Let's examine the effects of various kernel sizes on performance.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = MeanFilterAnalyzer()
analyzer.add_gaussian_noise(var=0.2)
fig = analyzer.compare()
glue("mean_gaussian_analysis", fig)
```

```{glue:figure} mean_gaussian_analysis
---
align: center
name: mean_gaussian_analysis_fig
---
Mean Filter Performance Analysis with Gaussian Noise
```

{numref}`mean_gaussian_analysis_fig` shows the effect of increasing kernel sizes on images corrupted by Gaussian noise. A blurring effect, and loss of detail is increasingly evident with larger kernel sizes.

Due to its simplicity, and computational efficiency, the mean filter is one of the most widely used spatial domain filters in biomedical imaging. As a low-pass frequency filter, it reduces the spatial intensity derivatives in the image; thereby, reducing the amount of noise corrupting the representation. There are; however, two main challenges with the main filter:

1. The mean filter averaging is sensitive to unrepresentative pixel values, which can significantly affect the mean value of all pixels in the neighborhood.
2. Since edges tend to have 'sharp' intensity gradients, the mean filter will interpolate new values based on the averages, which has the effect of blurring the edges.

Next, we examine another low-pass, linear filter widely used in image processing, the Gaussian filter.

### Gaussian Filter

The Gaussian Filter is similar to the mean filter, in that it works by convolving a 2-D point-spread function (kernel) with an image over a sliding window. Unlike the mean filter, however, the Gaussian filter’s kernel has a distribution equal to that of the 2-D Gaussian function:

```{math}
:label: gaussian_filter
G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}
```

{numref}`gaussian_kernel` shows a 5x5 Gaussian kernel with $\sigma$ = 1. Notice, the coefficients diminish with increasing distance from the kernel’s centre. Central pixels have a greater influence on the value of the output pixel than those on the periphery.

```{figure} ../../../figures/gaussian_kernel.png
---
name: gaussian_kernel
---
5 x 5 Gaussian Kernel
```

Producing such a kernel of discrete coefficients requires an approximation of the Gaussian distribution. Theoretically, the Gaussian distribution is non-zero over its spatial extent from $-\infty$ to $+\infty$. Covering the distribution would require a kernel of infinite size. But then, its values beyond, say, $5\sigma$ are negligible. (Given that the total area of a 1-D normal Gaussian distribution is 1, the area under the curve from $5\sigma$ to $\infty$ is about $2.9 \times 10^{-7}$.) In practice, we can limit the kernel size to three standard deviations of the mean and still cover 99% of the distribution.

{numref}`gaussian_gaussian_characteristics_fig` illustrates the results of a 3x3 gaussian filter kernel on a mammogram image degraded with Gaussian noise.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = GaussianFilterAnalyzer()
analyzer.add_gaussian_noise(var=0.2)
fig = analyzer.analyze()
glue("gaussian_gaussian_characteristics", fig)
```

```{glue:figure} gaussian_gaussian_characteristics
---
align: center
name: gaussian_gaussian_characteristics_fig
---
Gaussian Filter Performance Characteristics with Gaussian Noise
```

As shown in {numref}`gaussian_gaussian_characteristics_fig`, applying a 3×3 gaussian filter makes the image smoother, which is evident upon close examination of the features in the region of interest. The histograms illuminate the distribution of the signal vis-a-vis the noise. As (f) illustrates, most of the noise was in the brighter regions of the image.

Let's examine the effects of various kernel sizes on performance.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = GaussianFilterAnalyzer()
analyzer.add_gaussian_noise(var=0.2)
fig = analyzer.compare()
glue("gaussian_gaussian_analysis", fig)
```

```{glue:figure} gaussian_gaussian_analysis
---
align: center
name: gaussian_gaussian_analysis_fig
---
Gaussian Filter Performance Analysis with Gaussian Noise
```

Close examination shows some loss of detail, but not to the extent observed using the mean filter.

The Gaussian filter has several advantages:

1. Easy to implement.
2. Its’ Fourier transform is also a Gaussian distribution, centered around zero frequency. Its low-pass effectiveness can be controlled by adjusting its standard deviation.
3. Coefficients give higher weights to pixels in the centre; thereby, reducing the blurring effect over edges.
4. Computationally efficient. Gaussian kernels are separable; therefore, large filters can be implemented using many small 1D filters.
5. Rotationally symmetric, with no directional bias.

Most fundamentally, the Gaussian filter is based on the Human Visual System (HVS). It has been found that neurons create similar filters when processing visual information.

Gaussian filters do have certain challenges:

1. Blurring removes fine detail that may have diagnostic relevance.
2. Not as effective at removing "salt and pepper" noise.
3. Blurred edges can complicate edge detection.

Still, its design, computational efficiency, and flexibility makes the Gaussian filter ubiquitous in image processing.

The next linear filter takes a statistical approach to reducing blur, and minimizing noise, while retaining image detail.

### Wiener Filter

The Wiener filter, introduced in a 1943 classified memo by Norbert Wiener, incorporates statistical characteristics of the signal and noise into the image restoration process.

The objective of the Wiener filter is to take an input image, and then create an estimate of the true image such that the expected value of the squared difference between the estimate and the true image, is minimized.

The Wiener filter asserts the principle of orthogonality, which loosely states that the difference between the estimate and the input image, is uncorrelated with the input image. This means that the estimate is the projection of the input image onto the subspace spanned by the original image that minimizes the expected value of the error. From this, the filter computes a function that optimizes the trade-off between noise reduction, and signal distortion, by optimally attenuating the frequencies where the noise is dominant and preserving the frequencies where the signal is dominant.

The Wiener filter has some nice properties. Since it is optimal in the mean-squared error sense, it can produce high-quality images where noise is minimized and fine detail is retained. Another advantage is that it adapts to changing characteristics of the signal and the noise, making it suitable for speech, image restoration, and other applications whereby the noise is unknown or varies in time and space.

However, certain disadvantages limit its applicability and performance. For instance, the Wiener filter requires a priori knowledge of the degradation function and the power spectra for noise and signal, which may be difficult or impractical to obtain. Another disadvantage is that the Wiener filter can introduce artifacts, such as blurring and ringing due to smoothing in the frequency domain.
Next, up? Non-linear filters.

## Non-Linear Filters

In the previous section, we examined filters in which the output was a linear combination of the input. For additive, independent noise, or that which follows a simple statistical pattern, linear filters will reduce noise to the extent that signal and noise can be separated in the frequency domain.  For multiplicative, noise (speckle) or signals with non-linear features (edges, lines) that must be preserved, non-linear methods will be needed.

Now, we explore four non-linear techniques widely used in biomedical imaging: median filter, adaptive median filter, non-local means filter, and bilateral filter.

In the next sections, we’ll describe each of these methods, exhibit their performance, assess their advantages and disadvantages, and highlight the differences among them.

### Median Filter

The median filter is, a non-linear denoising and smoothing filter that uses ordering to compute the filtered value. A histogram is computed on the neighborhood, defined by a 2D kernel, and the central pixel value is replaced by the median of the pixel values in the neighborhood.

In {numref}`median_gaussian_characteristics_fig`, we have the results of a 3x3 median filter on a mammogram image degraded with Gaussian noise.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = MedianFilterAnalyzer()
analyzer.add_gaussian_noise(var=0.2)
fig = analyzer.analyze()
glue("median_gaussian_characteristics", fig)
```

```{glue:figure} median_gaussian_characteristics
---
align: center
name: median_gaussian_characteristics_fig
---
Median Filter Performance Characteristics with Gaussian Noise
```

{numref}`median_gaussian_characteristics_fig` (c) shows the result of applying a median filter to an image corrupted by Gaussian noise. The noise is effectively, removed while preserving much of the fine detail. Note the shape of the histogram in `median_gaussian_characteristics_fig` (g) more closely resembles that of the original image than the histograms of the linear filters.

Where the median filter is quite distinguished is with noise that produces extreme changes in pixel intensity. In {numref}`median_snp_characteristics_fig`, we apply the median filter to an image corrupted by 'salt and pepper' noise.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = MedianFilterAnalyzer()
analyzer.add_snp_noise(amount=0.4)
fig = analyzer.analyze()
glue("median_snp_characteristics", fig)
```

```{glue:figure} median_snp_characteristics
---
align: center
name: median_snp_characteristics_fig
---
Median Filter Performance Characteristics with Salt and Pepper Noise
```

Again, the noise is largely eliminated with little blurring effect.

{numref}`median_snp_analysis_fig` displays the effect of varying median filter kernels on salt and pepper noise.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = MedianFilterAnalyzer()
analyzer.add_snp_noise(amount=0.4)
fig = analyzer.compare()
glue("median_snp_analysis", fig)
```

```{glue:figure} median_snp_analysis
---
align: center
name: median_snp_analysis_fig
---
Median Filter Performance Analysis with Salt and Pepper Noise
```

The performance of the median filter is a characteristic of two properties:

1. The median is a more robust estimate of centrality than the mean. It is less affected by outliers.
2. Since the a pixel's output is an actual pixel value from its neighborhood, the median filter doesn't create unrealistic pixel values when the kernel straddles a line or edge.

As such, the median filter is much better at preserving sharp edges than the mean or Gaussian filters.

The median filter preserves high spatial frequency detail and performs best when noise is characterized by relatively few extreme changes in pixel intensity. However, when noise (such as Gaussian noise) effects the majority of pixels in the neighborhood, the median filter can be subjectively less effective than the mean or Gaussian filters.

Though the median filter is robust and possesses many optimality properties, its performance can be limited as all pixel values in the neighborhood are treated equally, regardless of their location within the observation window. A variant of the median filter that addresses this problem is the weighted median filter. Much like the weighted mean filter (Gaussian filter), where the output is the weighted median of the values in the neighborhood.

### Bilateral Filter

Introduced in 1995 by Volker Aurich et. al. {cite}`aurichNonLinearGaussianFilters1995`, the bilateral filter is a non-linear technique that can smooth an image while preserving strong edges. It has become a standard denoising tool in interactive applications such as Adobe Photoshop$\circledR$.

The bilateral filter is a weighted average of nearby pixels, similar to the Gaussian filter. The difference is that the bilateral filter considers the difference between a pixel value and that of its neighbors. The main idea is that a pixel is influenced by pixels that are not only nearby, but also have a similar intensity. The formulation is given by:

```{math}
:label: bilateral
BF[I] = \frac{1}{W_p}\displaystyle\sum_{q \in S} G_{\sigma_s}(||p-q||)G_{\sigma_r}(|I_p-I_q|)I_q,
```

where the normalization factor $W_p$ ensures the pixel weights sum to 1.0:

```{math}
:label: bilateral_normalizer
W_p = \displaystyle\sum_{q \in S} G_{\sigma_s}(||p-q||)G_{\sigma_r}(|I_p-I_q|).
```

where:

- $I_p$ is the intensity value of center pixel $p$,
- $I_q$ is the intensity value of neighboring pixel $q$,
- $G_{\sigma_s}$ is a spatial Gaussian weighting that decreases the influence of distant pixels, and
- $G_{\sigma_r}$ is a range Gaussian weighting that decreases the influence of pixels $q$ with intensity values that differ from $I_p$.

The spatial, or domain parameter, $\sigma_s$, is related to the scale of the pixel values and is found experimentally. The range parameter, $\sigma_r$, can be adapted from the noise level {cite}`celiuNoiseEstimationSingle2006`. As $\sigma_r$ increases, the bilateral filter gradually approximates the Gaussian filter. Best results have been achieved with  $\sigma_r = 1.95 \sigma_n$, where $\sigma_n$ is the local noise level.

In {numref}`bilateral_gaussian_characteristics_fig`, a bilateral filter with $\sigma_r=25$, and $\sigma_s=25$ is applied to an image degraded with Gaussian noise.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = BilateralFilterAnalyzer()
analyzer.add_gaussian_noise(var=0.2)
fig = analyzer.analyze(sigma_range=25, sigma_domain=25)
glue("bilateral_gaussian_characteristics", fig)
```

```{glue:figure} bilateral_gaussian_characteristics
---
align: center
name: bilateral_gaussian_characteristics_fig
---
Bilateral Filter Performance Characteristics with Gaussian Noise
```

{numref}`bilateral_gaussian_characteristics_fig`(c) exhibits noise reduction at the cost of some image detail. {numref}`bilateral_gaussian_analysis_fig` shows how $\sigma_r$, and $\sigma_s$ affect performance.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = BilateralFilterAnalyzer()
analyzer.add_gaussian_noise(var=0.2)
fig = analyzer.compare()
glue("bilateral_gaussian_analysis", fig)
```

```{glue:figure} bilateral_gaussian_analysis
---
align: center
name: bilateral_gaussian_analysis_fig
---
Bilateral Filter Performance Analysis with Gaussian Noise
```

The bilateral filter has several advantages. It has a simple formulation: a pixel's value becomes the weighted average of the pixel values in its neighborhood. Consequently, its behavior can be understood intuitively and adapted to application-specific tasks. Another advantage is that the bilateral filter depends on only two parameters that specify the contrast and size of features to preserve. Therefore, it is easy to tune to application-specific requirements.

An aspect that limits the applicability of the bilateral filter is its computational complexity, $O(S^2)$, where $S$ is the number of pixels. With quadratic complexity, the computational cost increases sharply with image size.  can explode the computational cost. Yet, there is no dearth of optimizations proposed ({cite}`durandFastBilateralFiltering2002`,{cite}`eladOriginBilateralFilter2002`, {cite}`phamSeparableBilateralFiltering2005`, {cite}`parisFastApproximationBilateral2006`, {cite}`weissFastMedianBilateral2006a`), which reduce the computation cost.

Another limitation extends from one of its strengths. As mentioned, the bilateral filter considers the range of intensity values when computing the weights to apply. If a pixel value is too different from the values of neighboring pixels, it has less influence on the output. As a consequence, the bilateral filter is not the best filter for salt and pepper noise. Here, the difference between a pixel and its neighbors can span the entire range (e.g., 0-255 for 8-bit images). In such cases, the values are too different to influence the output. Several approaches have been proposed to improve its performance in such cases. An example involves mollification. Often, images containing extreme intensity gradients are mollified using a median filter first, to obtain an initial estimate. Then the bilateral filter is applied to produce a more precise final estimate.

The use of the bilateral filter has grown rapidly since its introduction and is now ubiquitous in many image-processing applications.

### Non-Local Means (NL-means) Filter

The bilateral filter above is pixel-based, in that the weights are based on the location and intensity similarity between individual pixels. The problem is that comparing only grey levels in a single pixel is not very robust when these values are noisy {cite}`buadesNonLocalAlgorithmImage2005`.

The NL-means algorithm {cite}`buadesNonLocalAlgorithmImage2005` extends neighborhood filtering algorithms, such as the bilateral filter, by comparing the similarity not of the individual pixels, but of the neighborhoods centered at the pixels. Somewhat informally, each pixel $i$, in a neighborhood $\mathcal{N}_i$ of some fixed size $s$, is the weighted sum of the similarities between $\mathcal{N}_i$ and every other neighborhood $\mathcal{N}_j$ for $j\in I$.

More precisely, let $v=\{v(i) | i \in I\}$ be a noisy image, $i$, be a pixel in that image, and $\mathcal{N}_i$ be a (square) neighborhood of size $s$ centered on pixel $i$, then the estimated value NL(v)(i), for pixel $i$, is:

```{math}
:label: nl_means
NL(v)(i) = \displaystyle\sum_{j \in I} w(i,j)v(i,j),
```

where the set of weights ${w(i,j)}_j$ depend on the similarity between the neighborhoods of pixel $i$, $\mathcal{N}_i$ and pixel $j$, $\mathcal{N}_j$,  and satisfy conditions $0 \le w(i,j) \le 1$ and $\displaystyle\sum_j w(i,j) = 1$.

We describe the similarity between the neighborhoods $\mathcal{N}_i$ and $\mathcal{N}_j$ in terms of the intensity gray level vectors $v(\mathcal{N}_i)$ and $v(\mathcal{N}_j)$, where $\mathcal{N}_k$ denotes a square neighborhood of fixed size and centered at pixel $k$.

The similarity is measured as a decreasing function of the weighted Euclidean distance, $||v(\mathcal{N}_i) - v(\mathcal{N}_j)||^2_{2,\alpha}$, where $\alpha > 0$ is the standard deviation of the Gaussian kernel.
Hence, the weights are defined as:

```{math}
:label: nl_means_weights
w(i,j) = \frac{1}{Z(i)}e^{-\frac{||v(\mathcal{N}_i) - v(\mathcal{N}_j)||^2_{2,\alpha}}{h^2}},
```

where $Z(i)$ is the normalizing constant:

```{math}
:label: nl_means_nc
Z(i) = \sum_j e^{-\frac{||v(\mathcal{N}_i)-v(\mathcal{N}_j)||^2_{2,\alpha}}{h^2}},
```

and the parameter $h$ controls the decay of the exponential function and therefore the decay of the weights as a function of the Euclidean distances {cite}`buadesNonLocalAlgorithmImage2005`.

The original NL-means algorithm compares the neighborhood of each pixel $i$, with the neighborhoods of every other pixel $j$ $\forall j \in I$! This has a computational complexity that is quadratic in the number of pixels in the image. In practice, the search for similar neighborhoods is often restricted to a search window centered on the pixel itself, instead of the whole image.

In {numref}`nlmeans_gaussian_characteristics_fig`, a non-local means filter with $h=40$, kernel size = 7, and search window size = 21 is applied to an image degraded with Gaussian noise.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = NLMeansFilterAnalyzer()
analyzer.add_gaussian_noise(var=0.2)
fig = analyzer.analyze(h=40)
glue("nlmeans_gaussian_characteristics", fig)
```

```{glue:figure} nlmeans_gaussian_characteristics
---
align: center
name: nlmeans_gaussian_characteristics_fig
---
Non-Local Means Filter Performance Characteristics with Gaussian Noise
```

{numref}`nlmeans_gaussian_characteristics_fig`(c) shows some residual noise in the tissues which, upon close examination, obscures some fine detail. The histogram in {numref}`nlmeans_gaussian_characteristics_fig`(g) has a shape that more closely resembles those of the linear filters, than those of the other non-linear filters.

```{code-cell}
:tags: [hide-cell, remove-output]

analyzer = NLMeansFilterAnalyzer()
analyzer.add_gaussian_noise(var=0.2)
fig = analyzer.compare()
glue("nlmeans_gaussian_analysis", fig)
```

```{glue:figure} nlmeans_gaussian_analysis
---
align: center
name: nlmeans_gaussian_analysis_fig
---
Non-Local Means Filter Performance Analysis with Gaussian Noise
```

In {numref}`nlmeans_gaussian_analysis_fig`, we see the effects of various Gaussian kernels on a noisy image. The amount of residual noise is a decreasing function of $h$; however, the blurring effect is shown to increase with larger Gaussian kernels. At $h=80$, almost all of the noise is removed; although, the amount of blur is objectionable.

This wraps up our discussion of spatial domain filtering. In the next section, we will explore filtering in the frequency domain.
