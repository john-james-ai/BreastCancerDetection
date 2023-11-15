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

Discriminating between benign and malignant lesions in mammograms involves the detection and analysis of structural abnormalities in the breast tissue. Deep learning algorithms, specifically, convolutional neural networks, can extract features from regions of interest (ROIs) based on pixel intensity; however, this task is complicated by:

- the structural complexity of ROIs,
- the presence of artifacts (large texts and annotations) in the mammograms which resemble the pixel intensity of the ROI,
- noise in the form of random variations in pixel intensity that may have been produced during image capture,
- poor brightness and contrast levels in some mammograms,
- dense breast tissue with pixel intensities similar to that of cancerous tissue, and
- the limited number of mammogram images available for model training.

The available literature on machine learning and deep learning as applied to computer-aided detection (CADe) and diagnosis (CADx) systems has established the positive correlation between model performance and the degree to which the above-listed challenges have been addressed via principled and systematic image analysis and preprocessing. In this section, we describe our approach to image preprocessing for the CBIS-DDSM dataset, graphically depicted as follows.

```{figure} ../figures/ImagePrep.png
---
name: image_prep
---
Image preprocessing procedure of this study.
```

We begin with an evaluation of various denoising algorithms. Once a denoising method and its parameters are selected, we move to the artifact removal stage. Here, we analyze and evaluate several binary masking and thresholding techniques. Once a suitable binary mask is selected, morphological transformations are applied to the binarized images to remove artifacts. Next, the pectoral muscle is removed using various techniques such as Canny Edge Detection, Hough Lines Transformation, and Largest Contour Detection algorithms. To make malignant lesions more conspicuous during model training, we enhance image brightness and contrast using image enhancement methods such as Gamma Correction and Contrast Limited Adaptive Histogram Equalization (CLAHE). Gaussian noise is also added to improve the generalized performance of the neural network and mitigate model overfitting. Our penultimate task addresses the limited number of model training samples by augmenting the data with images that have been rotated and/or flipped along the vertical and horizontal dimensions. Finally, ROI segmentation will apply a pixel intensity threshold to create a binary mask of the lesion ROIs. Denoising, artifact removal, and related image enhancement methods are evaluated in terms of the quality of the images produced. To this end, image quality assessment metrics such as mean squared error (MSE), peak signal-to-noise ratio (PSNR), and structural similarity (SSIM) are applied to processed images during algorithm evaluation and selection stages.

## Denoise

Mammograms are inherently noisy, comprising random variations in image intensity and contrast caused by external disturbances of the image capture and/or transmission processes. Broadly speaking, two types of noise models are extant in mammography: additive and multiplicative noise. Mathematically, additive noise is given by:

```{math}
:label: additive_noise_model
N(x,y)=I(x,y)+J(x,y)
```

The multiplicative noise model is given by:

```{math}
:label: multiplicative_noise_model
N(x,y)=I(x,y)\times J(x,y)
```

where $N(x,y)$ is the noisy image $I(x,y)$ is the noise-free image, and $J(x,y)$ is the noise added to the original noise-free image.

In mammography, four classifications or types of noise are extant.

```{table} Types of Noise in Mammogram Images
:name: noise_types

| Noise Type       | Noise Model    |
|------------------|----------------|
| Salt and Pepper  | Additive       |
| Gaussian         | Additive       |
| Poisson          | Additive       |
| Speckle          | Multiplicative |
```

1. **Salt and Pepper Noise**: Also known as spike noise, impulse noise or flat-tail distribution noise.
2. **Speckle Noise**:

noise in mammography  These fluctuations are largely characterized as salt and pepper noise, speckle noise, Gaussian noise, and Poisson noise. Salt and pepper noise, also known as spike noise, impulsive noise or flat-tail distributed noise will appear as black and white dots on the image. Speckle noise is mainly found in radar images whereby the return signal from an object causes random fluctuations within the image. Gaussian noise is additive in nature and tends to follow a Gaussian distribution. Finally, Poisson noise or shot noise appears when there is statistical variation in image brightness, primarily due to characteristics of the capturing device, such as the number of photons used in low-dose X-ray mammography.

And so, denoising aims to decrease

### Filters
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