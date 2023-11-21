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
# Denoise

What is noise? Somewhat imprecisely, we might say that noise is any variation in brightness information not part of the original image. Yet, all biomedical images are imperfect representations of the underlying structure that is being imaged. Limited resolution (defined by the optics), uneven illumination or background, out-of-focus light, artifacts, and, of course, image noise, contribute to this imperfection. For denoising, we distinguish noise from other imperfections, with the following definition:

> Noise is the discrepancy between the true amount of light $s_i$ being measured at pixel $i$, and the corresponding measured pixel value $x_i$.

In the following subsections, we expand on this definition with a review of the most common noise models in digital mammograph. Then, we describe the predominate types of noises encountered in digital mammography.  Next, we introduce the most frequently applied denoising methods in low-dose X-ray mammography. Finally, we conduct an evaluation to determine the denoiser that produces

## Noise Models

Broadly speaking, noise can be modeled as additive, multiplicative, or impulse.

### Additive Noise Model

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

### Multiplicative Noise Model

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

### Impulse Noise Model

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

## Types of Noise in Digital Mammography

The types of noise most inherent in digital mammography are summarized in {numref}`noise_types`.

```{table}
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

```{figure}
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

```{figure}
---
name: mmg_quantize
---
Mammogram Quantization Noise

```

The image in {numref}`mmg_quantize` has been quantized to only one bit. Note that fine graduations in intensities are lost. Texturing in the image is lost to large areas of constant gray level. The effect of quantizing to too few bits creates an appearance known as "scalloped".

### Speckle Noise

Speckle noise is signal-dependent, non-Gaussian, multiplicative, and spatial-dependent which makes it one of the more complex image noise models. When an X-ray strikes a surface, it is reflected because of random microscopic variations in the roughness of the surface within one pixel.

```{figure}
---
name: mmg_speckle
---
Mammogram Speckle Noise

```

{numref}`mmg_speckle` illustrates several distributions of speckle degradation.

### Salt and Pepper Noise

Salt and pepper noise arises during Analog to Digital Conversion (ADC) and image transmission due to bit errors. An image degraded by salt and pepper noise has dark pixels in light areas and light pixels in dark backgrounds, giving the image a “salt and pepper” appearance.

Salt and pepper noise is an example of impulse noise and is modeled as follows. Let

- $s(x,y)$ be the original 8-bit image, with minimum and maximum pixels of 0, and 255 respectively,
- $f(x,y)$ be the image after it has been altered by salt and pepper noise, and
- $\alpha$ is the probability that a pixel is affected by salt and pepper noise, typically less than 0.1.

A simple model is as follows:

```{math}
:label: snp
Pr(f=s) = 1 - \alpha
```

```{math}
:label: snp_salt
Pr(f=\text{max}) = \frac{\alpha}{2}
```

```{math}
:label: snp_pepper
Pr(f=\text{min}) = \frac{\alpha}{2}
```

For instance, {numref}`mmg_snp` shows an 8-bit image with $\alpha=0.3$. Approximately 70%  ($1-\alpha$) of the image is unaltered, and 30% ($\alpha$) of the pixels have been changed to black or white.

```{figure}
---
name: mmg_snp
---
Mammogram Salt and Pepper Noise
```

### Poisson Noise

Image sensors measure scene irradiance by counting the number of discrete photons incident on the sensor over a given time interval. Since the detection of individual photons can be treated as independent events that follow a random temporal distribution, photon counting can be modeled as a Poisson process. The number of photons $N$ measured by a given sensor element over some time interval $t$ can be described by the standard Poisson distribution:

```{math}
:label: poisson_pdf
Pr(N=k) = \frac{e^{-\lambda t}(\lambda t)^k}{k!}
```

where $\lambda$ is the expected number of photons per unit time interval. The uncertainty described by this distribution is known as photon noise or Poisson noise.

Since the photon count follows a Poisson distribution, it has the property that the variance, $Var[N]$ is equal to the expectation, $E[N]$. This shows that photon noise is signal-dependent and that the standard deviation grows with the square root of the signal.

```{figure}
---
name: mmg_poisson
---
Mammogram Poisson Noise

```

The image in {numref}`mmg_poisson` shows the effect of poisson noise. Careful examination reveals that white areas are slightly more noisy than the dark areas.

## Denoiser Methods

Having defined an image $f(x,y)$ as a function of an original image $s(x,y)$ and a noise component $n(x,y)$ ({eq}`additive_noise_model`, and {eq}`multiplicative_noise_model`), we can define the denoising task as follows:

> Image denoising aims to provide a function $d(x) \approx s$ that takes a noisy image $f$ as input and returns an approximation of the true clean image $\hat{s}$ as output.

Our objective is to create an approximation of the original image, $\hat{s}$ in which:

- flat areas are smooth,
- edges are protected without blurring
- textures are preserved, and
- no new artifacts are generated.

Over the past two decades, a considerable body of research has been devoted to the design, development, and testing of denoising methods for biomedical imaging. While a systematic review of the denoising landscape is well beyond the scope of this effort, we will introduce the most commonly used techniques used in denoising biomedical images, with a focus on applications in mammography.

For this effort, we focus on five classes of filters commonly applied to the task of biomedical image noise reduction:

1. **Mean Filter**: Also known as the average filter,
2. **Median Filter**: A non-linear noise reduction technique {cite}`arceNonlinearSignalProcessing2005`,
3. **Gaussian Filter**: A linear noise reduction method based upon a Gaussian kernel {cite}`klapperResponseApproximationGaussian1959`,
4. **Adaptive Median Filter**: Median filter with variable window size {cite}`hwangAdaptiveMedianFilters1995`, and
5. **Non-Local Means Filter**: Filtering based upon non-local averaging of all pixels in the image {cite}`buadesNonLocalAlgorithmImage2005`.

### Mean Filter

Most commonly used to reduce additive Gaussian noise, the mean filter is a simple, intuitive, and easy to implement, filter of the linear class. Mean filtering simply replaces each pixel value with the average value of the intensities in its neighborhood. Usually thought of as a *convolutional filter*, mean filtering is based around the notion of a kernel, which represents the shape and size of the neighborhood to be sampled when computing the average intensities. Typically a 3x3 kernel is used; however, larger kernels (5x5, 7x7) can be used if greater smoothing is required.
