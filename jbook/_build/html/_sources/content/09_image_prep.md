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
# Image Preprocessing

Optimizing mammography for deep learning will comprise a series of image preprocessing steps. Denoising, artifact removal, pectoral muscle removal, and image enhancement are among the most essential steps in medical image preprocessing, and determining the optimal methods for these steps will be to a degree, an exercise in experimentation.  In this section, we conduct experiments that will determine the image preprocessing methods that will ultimately be applied to each image before model training.

This section will be organized as follows:

1. **Setup**: Initialize the repositories that will contain the images, preprocessing tasks, and image quality evaluations and extract a multivariate stratified sample of the images for experimentation.
2. **Denoise**: Conduct our first experiments with denoising methods


Import modules

```{code-cell}
import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../..")))
from bcd.config import Config
from bcd.container import BCDContainer
from bcd.etl.load import Loader
from bcd.preprocess.image.experiment.denoise import DenoiseExperiment
from bcd.preprocess.image.method.denoise import (
    BilateralFilter,
    GaussianFilter,
    MeanFilter,
    MedianFilter,
)
```

```{code-cell}
:tags: [remove-cell]

Config.set_log_level('INFO')
Config.set_mode('exp')
```

Wire our dependencies.

```{code-cell}
container = BCDContainer()
container.init_resources()
container.wire(
    packages=[
        "bcd.dal.repo", "bcd.preprocess.image.experiment", "bcd.dal.io", "bcd.etl"
    ]
)
```

```{code-cell}
:tags: [parameters]

# Section parameters
SETUP_COMPLETE = False
DENOISE_COMPLETE = False
BATCHSIZE = 16
```

## Setup

### Initialize Repositories

Experiment repositories are reset.

```{code-cell}
if not SETUP_COMPLETE:
    uow = container.dal.uow()
    uow.reset()
```

### Load Data

We will load 5% of the data, stratified by abnormality type, image view, BI-RADS assessment and cancer diagnosis.

```{code-cell}
if not SETUP_COMPLETE:
    loader = Loader(frac=0.05, groupby=['abnormality_type', 'image_view', 'assessment', 'cancer'])
    loader.run()
```

## Denoise

Noise in mammography is random variations in image brightness, color, or contrast that may have been produced during the image capture process. These fluctuations are largely categorized as salt and pepper noise, speckle noise, Gaussian noise, and Poisson noise. Salt and pepper noise, also known as spike noise, impulsive noise or flat-tail distributed noise will appear as black and white dots on the image.  Speckle noise is mainly found in radar images whereby the return signal from an object causes random fluctuations within the image. Gaussian noise is additive in nature and follows a Gaussian distribution. Finally, Poisson noise or shot noise appears when there is statistical variation in image brightness, primarily due to characteristics of the capturing device, such as the number of photons used in low-dose X-ray mammography.


### Image Denoising Problem Statement

Mathematically, the problem of image denoising can be modeled as:

$$
y = x + n
$$

where y is the observed noisy image, x is the unknown clean image and n represents additive white Gaussian noise (AWGN) with a standard deviation $\sigma_n$. Since there is no single unique solution x to the denoising model above, our problem is ill-posed. Yet, the literature is replete with techniques for estimating $\hat{x}$.  A survey of such techniques is well beyond the scope of this effort; yet we'll introduce a few of the most commonly used methods. As mammography is inherently noisy and usually contains low-contrast regions, our denoising challenges; therefore are to:

- ensure flat areas are smooth,
- protect edges from blurring,
- preserve textures in the mammography, and
- generate no new artifacts

An additional challenge that we'll introduce relates to the computational efficiency of the denoising technique. Given the volume of images to preprocess, our focus will be the methods that have linear time complexity or better.

### Image Denoising Methods

Image denoising is roughly classified into two categories: spatial domain methods and transform domain methods. Spatial domain methods remove noise by replacing image pixels with values derived from the pixel values in neighboring pixels. These are further classified into spatial domain filtering and variational denoising. The vast range of spatial domain filters is further classified into linear and non-linear filters. The original linear filters, MeanFilter and GaussianFilter are simple, computationally efficient, and intuitive. Non-linear filters such as the MedianFilter and Non-Local Means Filter reduce noise while preserving edges and textures from the original image.

###### MeanFilter

The MeanFilter, commonly used to reduce Gaussian noise, simply replaces each pixel value in an image with the mean value of its neighbors, including itself. A kernel specifies the shape and size of the neighborhood to
be sampled when computing the mean and must be a positive and odd integer. Typical kernel sizes of 3,5, or 7 are used and represent square kernels of 3,5, or 7 pixels in the horizontal and vertical directions. The larger the kernel, the greater the blurring or smoothing effect in the image.

The filter works by convolving the kernel over the image, estimating the local average intensities at each output pixel position. While, simple, and easy to implement, the MeanFilter has two drawbacks, namely:

- A single outlier pixel value can significantly affect the mean value of all the pixels in its neighborhood, and
- Edges are blurred, which can be problematic if sharp edges are required in the output.

#### GuassianFilter

Like the MeanFilter, the GaussianFilter is a 2-D convolution operator that is used to remove noise. By contrast; however, the GaussianFilter uses a different kernel that represents the shape of an isotropic (i.e. circularly symmetric) Gaussian distribution with the following form:

$$
G(x,y,\sigma) = \frac{1}{2\pi\sigma^2}e^-{\frac{x^2+y^2}{2\sigma^2}}
$$

where $x$ is the distance from the origin in the horizontal axis, $y$ is the distance from the origin in the vertical axis, and $\sigma$ is the standard deviation of the distribution, which is assumed to have a zero mean. The standard deviation, or scale of the Gaussian determines the amount of smoothing.

The distribution is illustrated as follows:

![2D Gaussian Distribution](/home/john/projects/bcd/jbook/figures/gaussian.png)

The GaussianFilter works by convolving this 2-D distribution as a point-spread function, giving more weight to the central pixels, having the highest Gaussian value, and lesser weights to the neighbors as the distances from the original pixel increase.  Since the image is stored as a collection of discrete pixels, we need to compute a discrete approximation of the Gaussian function before we can perform the convolution. We'll explore an approach to the problem of discretizing the Gaussian function, but first, let's review some of the properties of the Gaussian kernel.

#### Normalization

The $\frac{1}{2\pi\sigma^2}$ term in the Gaussian distribution is a normalization constant that ensures that its integral over its full domain is unity for every $\sigma$ and that the grey level of the image remains the same when we blur the image with this kernel. This means that increasing the $\sigma$ of the kernel substantially reduces the amplitude of the impulse response of the Gaussian filter. This property is known as average grey level invariance.

#### Cascading Property

The shape of the Gaussian kernel is scale-invariant. When we convolve two Gaussian kernels we get a new wider Gaussian with a variance $\sigma^2$, which is the sum of the variances of the constituting Gaussians. In this way, the Gaussian is known to be a self-similar function. As such, we can concatenate Gaussians to create a larger blurring Gaussian analogous to a cascade of waterfalls spanning the same height as the total waterfall.

#### Separability

An N-dimensional Gaussian kernel can be described in terms of a regular product of N one-dimensional kernels. For instance:

$$
g_{2D}(x, y, \sigma^2_1+\sigma^2_2=g_{1D}(x,\sigma_1^2)\otimes
g_{1D}(x,\sigma_2^2)
$$

Since higher dimensional Gaussian kernels can be described in terms of separate 1D kernels, they are called separable. The property of separability is elemental to discrete Gaussian kernel estimation.

#### Discrete Gaussian Kernel

Now, we move to the task of estimating the discrete Gaussian kernel. One popular approach is to convolve the original image with the discrete Gaussian kernel $T(n,t)$ {cite}`lindebergScalespaceDiscreteSignals1990`

$$
L(x,t)=\displaystyle\sum_{n=-\infty}^\infty f(x-n)T(n,t)
$$

where

$$
T(n,t)=e^{-t}I_n(t)
$$

and $I_n(t)$ denotes the modified Bessel function, a generalization of the sine function.  This discrete counterpart to the continuous Gaussian kernel is also the solution to the discrete diffusion equation in discrete space, continuous time, a characteristic we will revisit. 

In theory, the Gaussian function is for $x\in (-\infty,\infty)$ and is non-zero for every point on the image, requiring an infinitely wide kernel. In practice, however, values at a distance beyond three standard deviations
from the mean are small enough to be considered effectively zero.  Hence, this filter can be truncated in the spatial domain as follows:

$$
L(x,t)=\displaystyle\sum_{n=-M}^M f(x-n)T(n,t)
$$

where $M=C\sigma+1$, where $C$ is often chosen to be somewhere between 3 and 6.

Alternatively, the discrete Gaussian can be implemented using a frequency-domain approach, leveraging the closed-form expression for its discrete-time Fourier transform:

$$
\hat{T}(\theta,t)=\displaystyle\sum_{n=-\infty}^\infty T(n,t)e^{-i\theta n}=e^{t(\text{cos }\theta-1)}.
$$

$$


$$

### Denoising Methods

These experiments will focus on linear (MeanFilter, GaussianFilter) and non-linear (MedianFilter) spatial domain filters for noise reduction. The literature is replete with 

#### MeanFilter

The MeanFilter simply replaces each pixel value in an image with the mean value of its neighbors, including itself. A kernel specifies the shape and size of the neighborhood to be sampled when computing the mean and must be a positive and odd integer. Typical kernel sizes of 3,5, or 7 are used and represent square kernels of 3,5, or 7 pixels in the horizontal and vertical directions. The larger the kernel, the greater the blurring or smoothing effect in the image.
MeanFilter is simple, intuitive, and easy to implement; however, it has two drawbacks, namely:

- A single outlier pixel value can significantly affect the mean value of all the pixels in its neighborhood, and
- Edges are blurred, which can be problematic if sharp edges are required in the output.

#### GaussianFilter

Like the MeanFilter, the GaussianFilter is a 2-D convolution operator that is used to remove noise. By contrast, the GaussianFilter uses a different kernel that represents the shape of an isotropic (i.e. circularly symmetric) Gaussian distribution with the following form:

$$
G(x,y) = \frac{1}{2\pi\sigma^2}e^{-{\frac{x^2+y^2}{2\sigma^2}}}
$$

---

The distribution is shown in {ref}`gaussian`

```{code-cell}

```

### Mean Filter

```{code-cell}
params = {"kernel": [3, 5, 7]}
task = DenoiseExperiment(method=MeanFilter, params=params, batchsize=BATCHSIZE)
task.run()
```

## Median Filter

```{code-cell}
params = {"kernel": [3, 5, 7]}
task = DenoiseExperiment(method=MedianFilter, params=params, batchsize=BATCHSIZE)
task.run()
```

## Gaussian Filter

```{code-cell}
params = {"kernel": [3, 5, 7]}
task = DenoiseExperiment(method=GaussianFilter, params=params, batchsize=BATCHSIZE)
task.run()
```

## Bilateral Filter

```{raw-cell}
params = {"sigma_color_factor": [1], "sigma_space_factor": [1]}
task = DenoiseExperiment(method=BilateralFilter, params=params, batchsize=BATCHSIZE)
task.run()
```
