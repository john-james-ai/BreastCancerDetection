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

Next, we set the context with a brief overview of screen-film mammography, and the process by which the analog images are converted from film to the digital format. Then, we explore the image preprocessing steps in detail.

## Screen-Film Mammography, in Brief
Screen-film mammography (SFM) is the gold standard for breast cancer detection, and has been standard practice as a screening device since the 1970's. The technology has been perfected over the years, and its quality protocols for breast cancer detection and screening are well established. It boasts high spatial resolution, which enables detection of fine structures. Since receiving Food and Drug Administration (FDA) approval in 2000, digital mammography has seen rapid adoption in the United States. Digital mammography (DM) generates radiographic X-ray images in digital format  Like screen-film mammography, digital mammography   the first full-field digital mammography unit Digital mammography, since achieving FDA approval in 2000
Since Digital mammography Recent  SFM is a specialized type medical imaging that uses low-dose X-ray system to see inside the breasts. Images are rendered on film, then converted to a digital format for transmission, storage, and

## Denoise

Mammography is inherently noisy. Unwanted random signals
This subsection aims to explore three overarching questions with respect to denoising screen-film mammography:

1. What is noise?
What is noise, really? Somewhat imprecisely, we might say that noise is any variation in brightness information not part of the original image. But, all biomedical images are imperfect representations of the underlying structure that is being imaged.
Mammography is inherently noisy.

Image acquisition processes can inject unwanted and random signals (noise) into the image, which must be removed or minimized before downstream image processing can take place. So, the primary goal of denoising the image is to remove the unwanted signal or noise, which we denote as $n$, while preserving the details of the original, uncorrupted, signal $s$. More precisely, we aim to find a denoising function $f(x) \approx s$ that takes $x$, a noisy image as input, and returns an approximation of the *true* clean image $s$, as output.

### Noise: An Overview

So, what is image noise? Somewhat imprecisely, we might say that noise is any variation in brightness information not part of the original image. But, all biomedical images are imperfect representations of the underlying structure that is being imaged. Limited resolution (defined by the optics), uneven illumination or background, out-of-focus light, artifacts, and, of course, image noise, contribute to this imperfection. Noise is a variation, but not all variations are noise.

Let's *'describe'* image noise within the context of screen-film mammography, the modality of our CBIS-DDSM dataset, and see if a more precise *'definition'* emerges.

#### Screen-Film Mammography

In screen-film mammography, low-dose ionizing radiation (X-rays) are directed through compressed breast tissue. Compression allows radiologists to achieve high quality imaging, at lower doses of radiation. The X-rays are attenuated by tissues of varying radiographic opacity and are collected by rare-earth phosphor screens. When the X-ray is absorbed at the screen, a process called light scintillation converts high-energy X-rays to photons of visible light. A light-sensitive film emulsion in direct contact with the screen produces a latent image, which is subsequently rendered visible by chemical processing. The result is a 2-dimensional, gray-scale image, in which brightness corresponds to X-ray attenuation probability. We will call this, our true light source $s$.

Next, the continous, analog image is converted into a 2-dimensional digital image by an Analog-to-Digital Converter (ADC). We can think of a digital image as a 2-dimensional function $f(x,y)$, where $x$ and $y$ indcate the location of a pixel within the image, and $f(x,y)$ is a discrete value called the pixel intensity value. The ADC produces this digitized image via a two step process: sampling and quantization. Sampling refers to the digitization of the coordinate values $x$ and $y$; whereas, quantization describes the process of converting the amplitudes to pixel values.



  source $s$. Film in close proximity to the screen captures the light photons, and a grey-scaled image is obtained by exposing the film.  . This .

The types of noise we may encounter in digital mammography can be modeled as additive or multiplicative.

#### Additive Noise Model
 Additive noise is the undesired signal that arises during image acquisition that gets added to an image. Multiplicative noise
Signal processing theory defines an additive noise *model* given by:

```{math}
:label: additive_noise_model
f(x,y)=s(x,y)+n(x,y)
```

where:

- $x$ and $y$ are the coordinates of the pixel to which the noise is applied;
- $f(x,y)$ is the observed noisy image;
- $s(x,y)$ is an unobserved, but deterministic, noise-free image signal which has been corrupted by a noise process;
- $n(x,y)$ is the signal-independent, identically distributed, often zero-mean, random noise with variance $\sigma^2_n$, that is added to the original noise-free image.

In short, the additive model describes an image as the pixel wise sum of an unobserved, noise-free image and random noise signal of the same shape.

Multiplicative noise, by contrast, refers to the unwanted random signal that gets *multiplied* into an image during signal capture, transmission, or other processing. Similarly, we can define the multiplicative noise model as follows:

```{math}
:label: multiplicative_noise_model
f(x,y)=s(x,y)\times n(x,y)
```

where:

- $x$ and $y$ are the coordinates of the pixel to which the noise is applied;
- $f(x,y)$ is the noisy image;
- $s(x,y)$ is the noise-free image;
- $n(x,y)$ refers to signal-dependent, random noise that is multiplied into $s(x,y)$ during image capture, transmission, storage or other processing.

Whereas additive noise is signal independent, multiplicative noise is based on the value of the image pixel; whereby, the amount of noise multiplied into an image pixel is proportional to its value.

Reducing additive noise often involves some linear transformation into a space that separates the signal from the noise.

The various additive and multiplicative noise types extant in mammography include Gaussian Noise, Quantization Noise, Poisson Noise, and Impulse Noise.

### Gaussian Noise

Gaussian noise is ubiquitous in signal processing, telecommunications systems, computer networks, statistical modeling and digital biomedical imaging. Principally, sources of Gaussian noise in digital imagery arise during data acquisition, and may derive from:

- poor illumination during image capture;
- image sensors subject to overheating or other disturbances caused by external factors;
- interference in the transmission channel; or
- random variations in the electrical signal.

Gaussian noise is additive in nature, signal independent, and assumed to be zero-mean. Yet, the defining characteristic of Gaussian noise is that it has a probability density function equal to that of the normal distribution, first introduced by French mathematician Abraham de Moivre in the second edition (1718) of his Doctrine of Chances {cite}`grattan-guinnessLandmarkWritingsWestern2005`, and later attributed to Karl Friedrich Gauss, a German mathematician, for his work connecting the method of least squares to the normal distribution {cite}`stiglerGaussInventionLeast1981`.

Mathematically, Gaussian noise may be expressed by the following bivariate isotropic (circular) Gaussian function.

```{math}
:label: gaussian_pdf
g(x,y)=\frac{1}{2\pi\sigma_x\sigma_y}e^-\frac{[(x-\mu_x)^2+(y-\mu_y)^2]}{2\sigma_x\sigma_y}
```

where:

- $x$ and $y$ are the coordinates of the pixel to which the noise is applied;
- $\mu_x$ and $\mu_y$ are the means in the $x$ and $y$ dimensions, respectively;
- $\sigma_x$ and $\sigma_y$ are the standard deviations in the $x$ and $y$ dimensions, respectively.

{numref}`gaussian_noise` illustrates the additive model in the Gaussian context.

```{figure} ../figures/gaussian_noise.jpg
---
name: gaussian_noise
---
Guassian Noise
```

### Quantization Noise

Quantization noise arises out of the Analog to Digital Conversion (ADC) process. ADC consists of two steps: sampling and quantization. Sampling is the process of digitizing the coordinate values, $x$, and $y$, which defines the spatial resolution, or number of pixels of the digitized image. Quantization is the process of digitizing the amplitude or intensity values and determines the number of grey levels that each pixel can take.

Quantization noise is an unavoidable aspect of ADC. An analog signal is continuous, with infinite accuracy, while the digital signal's accuracy depends upon the quantization resolution, or number of bits in the ADC. A common assumption is that quantization noise is additive, uniformly distributed and signal dependent, unless other noise sources are large enough to cause dithering, the addition of random noise to the pre-quantized signal.

### Poisson Noise

Image sensors measure scene irradiance by counting the number of discrete photons incident on the sensor over a given time interval. Since the detection of individual photons can be treated as independent events that follow a random temporal distribution, photon counting can be modeled as a Poisson process and the number of photons $N$ measured by given sensor element over some time interval $t$ can be described by the standard Poisson distribution:

```{math}
:label: poisson_pdf
Pr(N=k) = \frac{e^{-\lambda t}(\lambda t)^k}{k!}
```

where $\lambda$ is he expected number of photons per unit time interval. The uncertainty described by this distribution is known as photon noise, or Poisson noise.

Since the photon count follows a Poisson distribution, it has the property that the variance, $Var[N]$ is equal to the expectation, $E[N]$. This shows that photon noise is signal dependent and that the standard deviation grows with the square root of the signal.


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
