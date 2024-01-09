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
(eda612)=
# Noise Types

The types of noise most inherent in digital mammography are summarized below.

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

## Gaussian Noise

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

```{figure} ../../figures/mammogram_gaussian.png
---
name: mammogram_gaussian
---
Mammogram Gaussian Noise

```

In {numref}`mammogram_gaussian` (b) we see the effect of a small amount of noise ($\sigma^2$ = 0.01). In {numref}`mammogram_gaussian` (c), the noise has been increased by a factor of 10 ($\sigma^2 = 0.1$). Notice the overall 'fuzziness'. Increasing the noise by another factor of ten ({numref}`mammogram_gaussian` (d)), the noise is much more objectionable.

## Quantization Noise

Quantization noise arises out of the process of converting a continuous analog image to a discrete digital representation. This Analog to Digital Conversion (ADC) consists of two steps: sampling and quantization. Sampling is the process of digitizing the coordinate values, $x$, and $y$. It defines the spatial resolution or number of pixels of the digitized image. Quantization is the process of digitizing the amplitude or intensity values. This process defines the number of gray levels that each pixel can take.

Quantization noise is an unavoidable aspect of ADC. An analog signal is continuous, with (almost) infinite accuracy, while the digital signal's accuracy depends upon the quantization resolution or number of bits in the ADC. A common assumption is that quantization noise is additive, uniformly distributed, and signal-independent unless the number of quantization levels are small. In which case, the noise is correlated on a pixel-by-pixel basis, and is not uniformly distributed {cite}`bovikHandbookImageVideo2000`.

Let $\triangle$ be the step size, then quantization noise, $q$, is modeled as being mean-centered and uniform between $\frac{-\triangle}{2}$ and $\frac{\triangle}{2}$. The variance is $\frac{\triangle^2}{12}$.

```{figure} ../../figures/mammogram_quantize.png
---
name: mammogram_quantize
---
Mammogram Quantization Noise

```

The image in {numref}`mammogram_quantize` has been quantized to only one bit. Note that fine graduations in intensities are lost. Texturing in the image is lost to large areas of constant gray level. The effect of quantizing to too few bits creates an appearance known as "scalloped".

## Speckle Noise

Speckle noise is signal-dependent, non-Gaussian, multiplicative, and spatial-dependent which makes it one of the more complex image noise models. When an X-ray strikes a surface, it is reflected because of random microscopic variations in the roughness of the surface within one pixel.

```{figure} ../../figures/mammogram_speckle.png
---
name: mammogram_speckle
---
Mammogram Speckle Noise

```

{numref}`mammogram_speckle` illustrates several distributions of speckle degradation.

## Salt and Pepper Noise

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

For instance, {numref}`mammogram_snp` shows an 8-bit image with $\alpha=0.3$. Approximately 70%  ($1-\alpha$) of the image is unaltered, and 30% ($\alpha$) of the pixels have been changed to black or white.

```{figure} ../../figures/mammogram_snp.png
---
name: mammogram_snp
---
Mammogram Salt and Pepper Noise
```

## Poisson Noise

Image sensors measure scene irradiance by counting the number of discrete photons incident on the sensor over a given time interval. Since the detection of individual photons can be treated as independent events that follow a random temporal distribution, photon counting can be modeled as a Poisson process. The number of photons $N$ measured by a given sensor element over some time interval $t$ can be described by the standard Poisson distribution:

```{math}
:label: poisson_pdf
Pr(N=k) = \frac{e^{-\lambda t}(\lambda t)^k}{k!}
```

where $\lambda$ is the expected number of photons per unit time interval. The uncertainty described by this distribution is known as photon noise or Poisson noise.

Since the photon count follows a Poisson distribution, it has the property that the variance, $Var[N]$ is equal to the expectation, $E[N]$. This shows that photon noise is signal-dependent and that the standard deviation grows with the square root of the signal.

```{figure} ../../figures/mammogram_poisson.png
---
name: mammogram_poisson
---
Mammogram Poisson Noise

```

The image in {numref}`mammogram_poisson` shows the effect of poisson noise. Careful examination reveals that white areas are slightly more noisy than the dark areas.
