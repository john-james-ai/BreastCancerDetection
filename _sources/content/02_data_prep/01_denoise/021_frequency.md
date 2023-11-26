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
# Frequency Domain Filtering

Normally, we think of an image as a rectangular array of pixels, each pixel representing an intensity at a position in the spatial domain. However, some operations are complicated, or impossible to perform in the spatial domain, and; therefore, a different representation is required.

Representing an image in the frequency domain makes certain denoising and smoothing operations on the periodic structure of the image possible. The frequency domain is a space in which an image is decomposed into the sum of complex sinusoidal waves; each represented by:

```{math}
:label: wave
y = A\text{sin}\Bigg(\frac{2\pi x}{\lambda} + \phi \Bigg)
```

where A is the amplitude of the wave, $2\pi$ is the angle measured in radians, equivalent to $360^{\circ}$, $\lambda$ is the wavelength, and $\phi$ is the phase, the amount the wave has shifted along the $x$ axis.

Each image value at a position, $F$, in the frequency domain, represents the amount that the intensity values in the spatial domain image vary over a specific distance relative to $F$ {cite}`GlossaryFrequencyDomain`.

We convert an image from the spatial domain to a spectrum in the frequency domain via the *Discrete Fourier transformation* (DFT) {cite}`fourierAnalyticalTheoryHeat2007`.  The DFT of an image $f$ of size $M \times N$ is an image $F$ of the same size and is defined as:

```{math}
:label: dft
F(u,v) = \displaystyle\sum_{m=0}^{M-1}\displaystyle\sum_{n=0}^{N-1} f(m,n)e^{-j2\pi(\frac{um}{M}+\frac{vn}{N})}
```

We can convert an image back into the spatial domain using the Inverse Discrete Fourier Transformation, given by:

```{math}
:label: dft_inv
F(x,y) = \frac{1}{MN}\displaystyle\sum_{m=0}^{M-1}\displaystyle\sum_{n=0}^{N-1} F(u,v)e^{+j2\pi(\frac{um}{M}+\frac{vn}{N})}
```

To get some intuition into the frequency domain representation, let’s plot a few FT images. In general, we plot the magnitude images and **not** the phase images [^phase].

[^phase] The case reports of people who have studied phase images shortly thereafter succumbing to hallucinogenics or ending up in a Tibetan monastery {cite}`IntroductionFourierTransform`  have not been corroborated. Still, better safe….

```{code-cell} ipython3
:tags: [hide-cell, remove-output]
import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../../..")))
import cv2
import matplotlib.pyplot as plt
import numpy as np
from bcd.utils.image import convert_uint8
from myst_nb import glue

FP_WHITE = "jbook/figures/frequency_white.jpg"
FP_MMG = "jbook/figures/mammogram.png"

def create_image(wavelength = 200, angle = 0):
    x = np.arange(-500,501,1)
    X,Y = np.meshgrid(x,x)
    return np.sin(2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / wavelength)

def get_image(fp: str, size: int = 200):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (size,size))

def transform_image(img, log: bool = False, clip: bool = False):
    # Shift the input
    img = np.fft.ifftshift(img)
    # Compute fourier transformation
    img = np.fft.fft2(img)
    # Shift the zero-frequency to the center
    img = np.fft.fftshift(img)
    # Compute amplitude
    img = np.abs(img)
    # Convert to log to scale the image
    if log:
        img = np.log(img)
    # Clip the image for photos.
    if clip:
        img = np.clip(img, a_min=0, a_max=50000)

    return img


img_white = get_image(fp=FP_WHITE)
img_hline = create_image()
img_vline = create_image(angle=2 * np.pi / 4, wavelength=100)
img_mmg = get_image(fp=FP_MMG)

img_white_fft = transform_image(img_white)
img_hline_fft = transform_image(img_hline)
img_vline_fft = transform_image(img_vline)
img_mmg_fft = transform_image(img_mmg, clip=True)

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,6))
_ = axes[0,0].imshow(img_white, cmap='gray')
_ = axes[0,1].imshow(img_hline, cmap='gray')
_ = axes[0,2].imshow(img_vline, cmap='gray')
_ = axes[0,3].imshow(img_mmg, cmap='gray')

_ = axes[1,0].imshow(img_white_fft, cmap='gray')
_ = axes[1,1].imshow(img_hline_fft, cmap='gray')
_ = axes[1,1].set_xlim([480,520])
_ = axes[1,1].set_ylim([520,480])
_ = axes[1,2].imshow(img_vline_fft, cmap='gray')
_ = axes[1,2].set_xlim([520,480])
_ = axes[1,2].set_ylim([480,520])
_ = axes[1,3].imshow(img_mmg_fft, cmap='gray')

labels = np.array([["(a)", "(b)", "(c)", "(d)"], ["(e)", "(f)", "(g)", "(h)"]])
for i in range(2):
    for j in range(4):
        _ = axes[i,j].set_xlabel(labels[i,j])
        _ = axes[i,j].set_xticks([])
        _ = axes[i,j].set_yticks([])

plt.tight_layout()

glue("fft", fig)
```

```{glue:figure} fft
---
align: center
name: fft_fig
---
Discrete Fourier Transformation
```

In {numref}`fft_fig` we see a constant white image on the left, a vertical sinusoidal grating with a wavelength of 200 pixels (5 cycles),  a horizontal sinusoidal grating with a wavelength of 100 pixels (10 cycles), and a  mammogram on the top row, along with their frequency spectrum images on the second row.

The DFT of the white image (e) contains a single dot at the center (origin) of the frequency coordinate system. The color of the dot represents the average intensity in the image and its location indicates the amplitude of the zero-frequency wave, also called the direct current (DC) of the constant white image.

Moving to the right (f), we have the DFT for the vertical sinusoid. Notice The DFT has two dots placed symmetrically along the $u-axis$ about the center of the DFT image.  The distance each dot occupies from the origin indicates the frequency with which the intensities are changing.

In the next image (g), we have twice the number of cycles (1/2 wavelength), rotated $\frac{\pi}{2}$ radians. Notice in this case, the dots are oriented along the $v-axis$, the direction of the rate of change, and are twice the distance from the origin.

{numref}`fft_fig` (h) shows the DFT for a mammogram image. Notice, the single blob about the origin. Most 8-bit gray-scale images tend to have an average value of about 128, and lots of low-frequency information close to the origin.

The above examples illuminate the important properties of the frequency spectrum of an image:

- The center pixel of the spectrum image is the average color or gray-level intensity of the image.
- The frequency amplitude spectrum is symmetric about the center DC pixel. Hence, the amplitudes of a given frequency F, are contained in a ring of radius F about the center DC pixel.
- Lower frequencies will present as pairs of dots symmetrically placed a short distance from the origin; whereas, higher frequencies will render pairs of dots symmetrically placed at farther distances from the origin.
- The frequency spectrum contains only frequency amplitude information, no positional data.

As a consequence of the above properties, smoothing, and denoising operations are a relatively simple matter of applying a circular ring mask around the center of the frequency spectrum. The diameter of the mask controls the cut-off frequency. Low-pass smoothing and denoising filters attenuate the high-frequencies outside the mask perimeter and *pass* the low-frequency information inside the mask.  High-pass edge-detecting masks attenuate low-frequency information inside the mask and pass the high-frequency data outside the periphery.
