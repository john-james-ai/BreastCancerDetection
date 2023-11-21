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

Precise and accurate diagnosis of breast cancer rests upon the discriminatory power of mathematical models designed to detect and classify structural abnormalities in breast tissue from biomedical imaging. Advances in artificial intelligence and computer vision, fueled by an explosion in AI task-specific computational power, have given rise to dense image recognition models capable of distinguishing increasingly complex patterns and structures in biomedical images. Still, the diagnostic performance and clinical applicability of such models rests upon the availability of large datasets containing high-quality, high-resolution images that are clear, sharp, and free of noise and artifacts.

Exploratory analysis of the CBIS-DDSM mammograph illuminated several issues that compromise the discriminatory power of image detection and recognition models.

- Various artifacts (large texts and annotations) are present within the mammography that resemble the pixel intensities of the regions of interest (ROIs), which can interfere with the ROI extraction process and/or lead to false diagnosis.
- Noise of various types in the images is an obstacle to effective feature extraction, image detection, recognition, and classification.
- Poor brightness and contrast levels in some mammograms may increase the influence of noise, and/or conceal important and subtle features.
- Malignant tumors are characterized by irregular shapes and ambiguous or blurred edges that complicate the ROI segmentation task.
- Dense breast tissue with pixel intensities similar to that of cancerous tissue, may conceal subtle structures of diagnostic importance.
- Deep learning models for computer vision require large datasets. The CBIS-DDSM has just over 3500 full mammogram images, a relatively small dataset for model training.

Addressing these challenges is fundamentally important to model detection, recognition, and classification performance.

## Image Preprocessing Overview

In this regard, a five-stage image preprocessing approach ({numref}`image_prep`) has been devised to reduce noise in the images, eliminate artifacts, and produce a collection of images for maximally effective computer vision model training and classification.

```{figure} ../figures/ImagePrep.png
---
name: image_prep
---
Image Preprocessing Approach

```

We begin with an evaluation of various denoising methods commonly applied to mammography. Once a denoising method and its (hyper) parameters are selected, we move to the artifact removal stage. Image binarizing and thresholding methods are evaluated, then morphological transformations are applied to the binarized images to remove artifacts. Next, the pectoral muscle is removed using various techniques such as Canny Edge Detection, Hough Lines Transformation, and Largest Contour Detection algorithms. To make malignant lesions more conspicuous during model training, we enhance image brightness and contrast with Gamma Correction and Contrast Limited Adaptive Histogram Equalization (CLAHE). Additive White Gaussian Noise (AWGN) is also added to improve the generalized performance of the neural network and mitigate model overfitting. Finally, we extract the ROIs using automated pixel intensity thresholding to create a binary mask which is applied to the enhanced images.

Let's dive in.

## Denoise

What is noise? Somewhat imprecisely, we might say that noise is any variation in brightness information not part of the original image. Yet, all biomedical images are imperfect representations of the underlying structure that is being imaged. Limited resolution (defined by the optics), uneven illumination or background, out-of-focus light, artifacts, and, of course, image noise, contribute to this imperfection. For denoising, we distinguish noise from other imperfections, with the following definition:

> Noise is the discrepancy between the true amount of light $s_i$ being measured at pixel $i$, and the corresponding measured pixel value $x_i$.

With that, we can state the denoising problem as follows:

> Image denoising aims to provide a function $d(x) \approx s$ that takes a noisy image $x$ as input and returns an approximation of the true clean image  $s$ as output.

Here, we've implicitly decomposed an image, $f$ into a desired or uncorrupted signal $s$ and a noise component $n$. How are $s$ and $n$ related? Are they independent? Can we eliminate $n$ with denoising? Next, we review how noise is modeled in screen-film mammography.

### Noise Models

Broadly speaking, noise can be modeled as additive, multiplicative, or impulse.

#### Additive Noise Model

The additive noise model describes the abrupt, and undesired signal that gets added to an image, and is given by:

```{math}
:label: additive_noise_model
f(x,y)=s(x,y)+n(x,y)
```

where:

- $x$ and $y$ are the coordinates of the pixel to which the noise is applied;
- $f(x,y)$ is the observed noisy image;
- $s(x,y)$ is the noise-free image signal which has been corrupted by a noise process;
- $n(x,y)$ is the signal-independent, random noise that is added to the original noise-free image.

Additive noise, the most common decomposition, can arise from numerous sources, including radiation scatter from the surface before the image is sensed; electrical noise in image sensors, random charge fluctuations induced by thermal noise, environmental electromagnetic interference, and noise associated with the acquisition system. However, dominant source of additive noise in X-ray mammography is the digitization process {cite}`bovikHandbookImageVideo2000`.

#### Multiplicative Noise Model

Multiplicative noise ({eq}`multiplicative_noise_model`), by contrast, refers to the unwanted random signal that gets *multiplied* into an image during signal capture, transmission, or other processing.

```{math}
:label: multiplicative_noise_model
f(x,y)=s(x,y)\times n(x,y)
```

where:

- $x$ and $y$ are the coordinates of the pixel to which the noise is applied;
- $f(x,y)$ is the noisy image;
- $s(x,y)$ is the noise-free image;
- $n(x,y)$ refers to signal-dependent, random noise that is multiplied into $s(x,y)$ during image capture, transmission, storage or other processing.

Whereas additive noise is signal independent, multiplicative noise is based on the value of the image pixel; whereby, the amount of noise multiplied into an image pixel is proportional to its intensity.

Note that we can transform a multiplicative noise model to an additive noise model by taking the logarithm of both sides of the multiplication model. The additive model becomes multiplicative by exponentiation.

For instance, {eq}`multiplicative_noise_model` becomes:

```{math}
:label: log_mult_model
log f = log(sn) = log (s) + log (n)
```

Similarly, {eq}`additive_noise_model` becomes:

```{math}
:label: exp_add_model
e^f = e^{s_n} = e^f e^n.
```

A common strategy for addressing multiplicative noise is to convert the noise in the image to additive noise, then apply an appropriate spatial domain filter.

#### Impulse Noise Model

Lastly, we have the impulse noise model, which represents a separate mathematical model, neither additive nor multiplicative. Impulse noise arises in the image as a consequence of signal transmission errors. Mathematically, we can describe impulse noise by the following equation.

```{math}
:label: impulse
{f(x,y)} = \begin{cases}
    s(x,y)\text{ with probability 1-P} \\
    s(x,y) + n(x,y) \text{ with probability P}
\end{cases}
```

where:

- 0 \le P \le 1
- $x$ and $y$ are the coordinates of the pixel to which the noise is applied;
- $f(x,y)$ is the noisy image;
- $s(x,y)$ is the noise-free image;
- $n(x,y)$ refers random noise

Higher values of $P$ correspond to greater levels of corruption.

A simplification of {eq}`impulse` for the case in which $n(x,y)$ replaces $s(x,y)$

```{math}
:label: impulse_2
{f(x,y)} = \begin{cases}
    s(x,y)\text{ with probability 1-P} \\
    n(x,y) \text{ with probability P}
\end{cases}
```

Impulse noise can be static or dynamic. In the case of static impulse noise, its pixel values get modified by only two values: either the low or high value in the range of pixel values. For instance, in an 8-bit grayscale image with values ranging from 0 (black) to 255 (white), dark pixels corrupted by impulse noise would be replaced by 255, and white pixels would be replaced by 0. In the case of dynamic impulse noise, pixels are modified independently, with random values between 0 and 255.

Next, we review the types of noise we may encounter.

### Types of Noise in Screen-Film Mammography

The types of noise most inherent in screen-film mammography are summarized in {numref}`noise_types`.

```{table} Noise Types in Screen-Film Mammography
:name: noise_types

| Noise                         | Model          | Signal Dependence | Source                            |
|-------------------------------|----------------|-------------------|-----------------------------------|
| Gaussian Noise                | Additive | Independent       | Signal Acquisition / Transmission |
| Quantization Noise            | Additive | Dependent         | Digitization                      |
| Speckle Noise                 | Multiplicative | Dependent         | Signal Detection                  |
| Salt & Pepper Noise           | Impulse   | Independent       | Signal Transmission               |
| Poisson Photon Counting Noise | Neither / Both       | Dependent         | Signal Detection                  |


```

### Gaussian Noise

> "Whenever a large sample of chaotic elements are taken in hand and marshalled in the order of their magnitude, an unsuspected and most beautiful form of regularity proves to have been latent all along." (Sir Francis Galton, 1889)

One of the most significant discoveries in probability theory is the **central limit theorem**, which simply states that the sum of a large number of small independent random variables, under fairly mild conditions, is normally distributed.  This holds even when the underlying random variables are not normally distributed and when the random variables do not have the same distribution.

Gaussian noise arises during data acquisition and is caused by many small, independent random contributions of factors such as:

- thermal vibration of atoms,
- heat variations in image sensors
- random variations in electronic signal, e.g. electronic circuit noise

It is additive in nature, independent at each pixel, independent of signal, and has a probability density {eq}`gaussian_pdf` equal to that of the Gaussian distribution.

```{math}
:label: gaussian_pdf
P(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
```

where $\mu$ is the mean gray value, and $\sigma$ is the standard deviation.

```{figure} ../figures/mammogram_gaussian.png
---
name: mmg_gaussian
---
Mammogram Gaussian Noise

```

In {numref}`mmg_gaussian` (b) we see the effect of a small amount of noise ($\sigma^2$ = 0.01). In {numref}`mmg_gaussian` (c), the noise has been increased by a factor of 10 ($\sigma^2 = 0.1$). Notice the overall 'fuzziness'. Increasing the noise by another factor of ten ({numref}`mmg_gaussian` (d)), the noise is much more objectionable.

### Quantization Noise

Quantization noise arises out of the process of converting a continuous analog image to a discrete digital representation. This Analog to Digital Conversion (ADC) consists of two steps: sampling and quantization. Sampling is the process of digitizing the coordinate values, $x$, and $y$. It defines the spatial resolution or number of pixels of the digitized image. Quantization is the process of digitizing the amplitude or intensity values. This process defines the number of gray levels that each pixel can take.

Quantization noise is an unavoidable aspect of ADC. An analog signal is continuous, with (almost) infinite accuracy, while the digital signal's accuracy depends upon the quantization resolution or number of bits in the ADC. A common assumption is that quantization noise is additive, uniformly distributed, and signal-independent unless the number of quantization levels are small. In which case, the noise is correlated on a pixel-by-pixel basis, and is not uniformly distributed {cite}`bovikHandbookImageVideo2000`.

Let $\triangle$ be the step size, then quantization noise, $q$, is modeled as being mean-centered and uniform between $\frac{-\triangle}{2}$ and $\frac{\triangle}{2}$. The variance is $\frac{\triangle^2}{12}$.

```{figure} ../figures/mammogram_quantize.png
---
name: mmg_quantize
---
Mammogram Quantization Noise

```

The image in {numref}`mmg_quantize` has been quantized to only one bit. Note that fine graduations in intensities are lost. Texturing in the image is lost to large areas of constant gray level. The effect of quantizing to too few bits creates an appearance known as "scalloped".

### Speckle Noise

Speckle noise is signal-dependent, non-Gaussian, multiplicative, and spatial dependent which makes it one of the more complex image noise models. When X-ray

### Poisson Noise

Image sensors measure scene irradiance by counting the number of discrete photons incident on the sensor over a given time interval. Since the detection of individual photons can be treated as independent events that follow a random temporal distribution, photon counting can be modeled as a Poisson process. The number of photons $N$ measured by a given sensor element over some time interval $t$ can be described by the standard Poisson distribution:

```{math}
:label: poisson_pdf
Pr(N=k) = \frac{e^{-\lambda t}(\lambda t)^k}{k!}
```

where $\lambda$ is the expected number of photons per unit time interval. The uncertainty described by this distribution is known as photon noise or Poisson noise.

Since the photon count follows a Poisson distribution, it has the property that the variance, $Var[N]$ is equal to the expectation, $E[N]$. This shows that photon noise is signal-dependent and that the standard deviation grows with the square root of the signal.

```{figure} ../figures/mammogram_poisson.png
---
name: mmg_poisson
---
Mammogram Poisson Noise

```

The image in {numref}`mmg_poisson` shows the effect of poisson noise. Careful examination reveals that white areas are slightly more noisy than the dark areas.

## Filters

Image filtering techniques have broad applicability in biomedical image analysis and processing. In fact, most biomedical image analysis involves the application of image filtering at stages prerequisite to analysis. Fields such as signal processing, statistics, information theory, and computer vision have produced a considerable and growing body of research devoted to the design, development, and testing of filtering methods to improve the signal-to-noise ratio (SNR) in audio, video, and imaging. While a systematic review of the denoising landscape is well beyond the scope of this effort, we will introduce the most commonly used filtering techniques used in denoising biomedical images, with a focus on applications in mammography.

For this effort, we focus on five classes of filters commonly applied to the task of biomedical image noise reduction:

1. **Mean Filter**: Also known as the average filter,
2. **Median Filter**: A non-linear noise reduction technique {cite}`arceNonlinearSignalProcessing2005`,
3. **Gaussian Filter**: A linear noise reduction method based upon a Gaussian kernel {cite}`klapperResponseApproximationGaussian1959`,
4. **Adaptive Median Filter**: Median filter with variable window size {cite}`hwangAdaptiveMedianFilters1995`, and
5. **Non-Local Means Filter**: Filtering based upon non-local averaging of all pixels in the image {cite}`buadesNonLocalAlgorithmImage2005`.

### Mean Filter

Most commonly used to reduce additive Gaussian noise, the mean filter is a simple, intuitive, and easy to implement, low-pass filter of the linear class. Low-pass filters, also known as smoothing or blurring filters, are The most basic of filtering operations is called "low-pass"
Mean filtering simply replaces each pixel value with the average value of the intensities in its neighborhood. Usually thought of as a *convolutional filter*, mean filtering is based around the notion of a kernel, which represents the shape and size of the neighborhood to be sampled when computing the average intensities. Typically a 3x3 kernel is used; however, larger kernels (5x5, 7x7) can be used if greater smoothing is required.
A kernel specifies the shape and size of the neighborhood to be sampled when computing the mean. Typically     Gaussian