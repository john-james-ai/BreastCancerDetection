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

The precise and accurate diagnosis of breast cancer, rests upon the discriminatory power of mathematical models designed to detect and classify structural abnormalities in breast tissue from biomedical imaging. Advances in artificial intelligence and computer vision, fueled by an explosion of AI task-specific computational power, have given rise to dense image recognition models capable of distinguishing increasingly complex patterns and structures in biomedical images. Still, the diagnostic performance and clinical applicability of such models rests upon the availability of large datasets containing high-quality, high-resolution images that are clear, sharp and free of noise and artifacts.

However, an exploratory analysis of the CBIS-DDSM dataset exposed several image quality concerns requiring attention. that must be addressed prior to the modeling stage.

- Various artifacts (large texts and annotations) are present within the mammography which resemble the pixel intensities of the regions of interest (ROIs), which can interfere with the ROI extraction process and/or lead to false diagnosis.
- Noise of various types in the images are obstacles to effective feature extraction, image detection, recognition and classification.
- Poor brightness and contrast levels in some mammograms may increase the influence of noise, and/or conceal important and subtle features.
- Malignant tumors are characterized by irregular shape and ambiguous or blurred edges that complicate the ROI segmentation task.
- Dense breast tissue with pixel intensities similar to that of cancerous tissue, may conceal subtle structures of diagnostic importance.
- There is a limited number of mammogram images available for model training.

In this regard, the purpose of this section is to describe the image preprocessing approach, and methods summarized in {numref}`image_prep`.

```{figure} ../figures/ImagePrep.png
---
name: image_prep
---
Image Preprocessing Approach

```

We begin with an evaluation of various denoising algorithms. Once a denoising method and its parameters are selected, we move to the artifact removal stage. Here, we analyze and evaluate several binary masking and thresholding techniques. Once a suitable binary mask is selected, morphological transformations are applied to the binarized images to remove artifacts. Next, the pectoral muscle is removed using various techniques such as Canny Edge Detection, Hough Lines Transformation, and Largest Contour Detection algorithms. To make malignant lesions more conspicuous during model training, we enhance image brightness and contrast using image enhancement methods such as Gamma Correction and Contrast Limited Adaptive Histogram Equalization (CLAHE). Gaussian noise is also added to improve the generalized performance of the neural network and mitigate model overfitting. Our penultimate task addresses the limited number of model training samples by augmenting the data with images that have been rotated and/or flipped along the vertical and horizontal dimensions. Finally, ROI segmentation will apply a pixel intensity threshold to create a binary mask of the lesion ROIs. Denoising, artifact removal, and related image enhancement methods are evaluated in terms of the quality of the images produced. To this end, image quality assessment metrics such as mean squared error (MSE), peak signal-to-noise ratio (PSNR), and structural similarity (SSIM) are applied to processed images during algorithm evaluation and selection stages.

## Denoise

Mammograms are inherently noisy, comprising random variations in image intensity and contrast caused by external disturbances within the image capture and/or transmission processes. Broadly speaking, image noise can be described as being additive or multiplicative. Additive noise is the undesired signal that arises during data acquisition that gets added to an image. Signal processing theory defines an additive noise *model*; whereby, an observed *noisy* image, $f$ is really the sum of an unobserved, noise-free signal $s$ and random, zero-mean, signal independent, independent and identically distributed noise, $n$. Concretely, the additive noise model is given by:

```{math}
:label: additive_noise_model
f(x,y)=s(x,y)+n(x,y)
```

where:

- $x$ and $y$ are the coordinates of the pixel to which the noise is applied;
- $f(x,y)$ is the observed noisy image;
- $s(x,y)$ is an unobserved, but deterministic, noise-free image signal which has been corrupted a noise process;
- $n(x,y)$ is the signal-independent, often zero-mean, random noise with variance $\sigma^2_n$, that is added to the original noise-free image.

Multiplicative noise, by contrast, refers to the unwanted random signal that gets *multiplied* into an image during signal capture, transmission, or other processing. Mathematically, we can define the multiplicative noise model as follows:

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

With that, let's review the types of additive and multiplicative noise most inherent in mammography.

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

As a result of the X-ray system or the digitization hardware, mammograms can be affected by additive white Gaussian noise (AWGN), which has the additional property that the noise
As an illustration, we have f(x,y), the noisy image, s(x,y), the noise-free image and n(x,y) is the signal-independent, random, zero-mean Gaussian noise.

```{figure} ../figures/gaussian_noise.jpg
---
name: gaussian_noise
---
Guassian Noise
```


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
