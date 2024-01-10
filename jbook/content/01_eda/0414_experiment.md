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
(eda614)=
# Denoising Experiment

An experiment was conducted to evaluate the various denoising techniques described above. Eighty-eight (88) images were selected using stratified sampling to ensure all case types and abnormalities were represented. The denoisers evaluated and their parameters are specified below.

| Domain     | Denoiser          | Parameters                                                               |
|------------|-------------------|--------------------------------------------------------------------------|
| Spatial    | MeanFilter        | params={'kernel': [3,5,7]}                                               |
|            | GaussianFilter    | params={'kernel': [3,5,7]}                                               |
|            | MedianFilter      | params={'kernel': [3,5,7]}                                               |
|            | BilateralFilter   | params={'sigma_color_factor': [0.5,1,0, 1.5], 'sigma_space_factor': [1]} |
|            | NonLocalMeans     | params={'h': [3, 5, 10]}                                                 |
| Frequency  | ButterworthFilter | params={'order': [10], 'cutoff': [0.25, 0.5, 0.75]}                      |
|            | WaveletFilter     | params={'wavelet': ['haar']}                                             |

Four noisy images were created from each ground-truth image: a Gaussian noise image, a speckle noise image, a salt-n-pepper noise image, and an impulse noise image. Subjective and objective Image Quality Assessments (IQAs) were conducted for each of the above denoisers on a total of 352 images.

## Subjective Image Quality Assessment

Subjective assessments were performed on a double stimulus continuous quality scale (DSCQS) {cite}`BT500Methodologies`, in which a random sampling of test images was compared to their associated ground truth images for noise reduction and preservation of fine detail in the breast tissue. Each randomly selected ground truth and associated test image was marked with a quality score in [0,100], and then the difference between the quality scores was computed. The DSCQS score was calculated by averaging the quality score differences for all images denoised by a given method. A lower value indicates better image quality and a larger value reflects poorer image quality. The subjective score was finally calculated for each denoiser method as follows:

```{math}
:label: subjective
\text{Subjective Score} = 100-\text{DSCQS Score}
```

### Objective Image Quality Assessment

Objective methods are based on image quality metrics (IQMs) that are designed to estimate the quality of the image automatically, in qualitative terms as observed by human subjects {cite}`afnanSubjectiveAssessmentObjective2023`.  For this effort, three IQMs were used to assess image quality: mean squared error (MSE), peak signal-to-noise ratio (PSNR), and the structural similarity index measure (SSIM).

#### Mean Squared Error (MSE)

MSE, the most common performance criterion for measuring image quality, characterizes the squared error among pixels contained in two images. Mathematically, MSE is defined as follows:

```{math}
:label: mse
\text{MSE} = \frac{1}{mn}\displaystyle\sum_{i=0}^{m-1}\displaystyle\sum_{j=0}^{n-1}(G(i,j)-P(i,j))^2
```

where, **G** is the ground truth image, and **P** is the processed image. The pixels of **G** and **P** are denoted by **m** and **n**. Lastly, the rows and columns of pixels **m**, and **n**, are denoted by **i**, and **j**.
The objective here is to find the *minimum mean-squared error estimator* (MMSE).
Though the use of MSE is widespread, it does have a shortcoming. The main problem with mean-squared error is that the measurement depends upon the image intensity scaling. An MSE of 100 for an 8-bit image with pixel values in the range 0-255 is objectionable, but an MSE of 100 for a 16-bit image with pixel values in [0,65536] would be barely noticeable.
Peak Signal-to-Noise Ratio (PSNR) avoids this problem by scaling the MSE according to the image range.

#### Peak Signal-to-Noise Ratio (PSNR)

PSNR expresses the ratio between a signal’s maximum possible value (power) and the power of the corrupting noise that affects the quality of its representation.  The higher the PSNR, the smaller the error, and the better the quality of the image vis-a-vis the original.

Mathematically, PSNR is calculated with equation {eq}`psnr`.

```{math}
:label: psnr
\text{PSNR} = 20\text{log}_{10}\Bigg(\frac{(MAX)}{\sqrt{MSE}}\Bigg)
```

where MAX is the maximum pixel value contained in the image and PSNR is measured in decibels (dB). An acceptable PSNR for an 8-bit image would be in the range of 30 to 50 dB {cite}`beeravoluPreprocessingBreastCancer2021`.

#### Structural Similarity Index Measure (SSIM)

The Structural Similarity Index Measure (SSIM) is a method for measuring the structural similarity between two images. Since its introduction in 2004 {cite}`bakurovStructuralSimilarityIndex2022`, the SSIM has become one of the most popular full-reference image quality assessment (FR-IQA) measures (over 47,960 citations), owing its success to its mathematical simplicity, low computational complexity, and implicit incorporation of the Human Visual System’s (HVS) characteristics. The incorporation of these characteristics has resulted in a metric with a better correlation with subjective evaluation provided by human observers. Consequently, SSIM has been used as a proxy evaluation for human assessment in a range of image processing and computer vision applications.

SSIM separately measures local brightness (a.k.a. luminance), contrast, and structure of both images and then aggregates all local assessments to obtain the overall measure {cite}`bakurovStructuralSimilarityIndex2022`.

Unlike the MSE-based measures that compare images on a pixel-by-pixel basis, the SSIM operates on patches obtained from a sliding window.  This technique better models the HVS because our eyes can discern differences in local information in a specific area of two images.

Formally, the SSIM compares a reference image and a potentially corrupt image based on three independent components: luminance, contrast, and structure.

##### SSIM Luminance Component

Each image’s patch average $\mu$ represents the luminance information. Hence, the comparison is given by:

```{math}
:label: luminance
l(x,y) = \frac{2\mu_x\mu_y+C_1}{\mu_x^2 + \mu_y^2 + C_1},
```

where $C_1$ is a small quantity introduced for numerical stability. (Stand by, we’ll define that quantity at the end.)

##### SSIM Contrast Component

Contrast is defined in terms of the standard deviation, and the comparison is given by:

```{math}
:label: contrast
c(x,y) = \frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2 + \sigma_y^2 + C_2},
```

##### SSIM Structure Component

The structure element is represented by the inner product of the mean and standard deviation of each image. The comparison is given by:

```{math}
:label: structure
s(x,y) = \frac{\sigma_{xy}+C_3}{\sigma_x \sigma_y + C_3}
```

where:

```{math}
:label: structure_var
\sigma_{xy} = \frac{1}{N-1}\displaystyle\sum_{i=1}^N (x_i-\mu_x)(y_i-\mu_y).
```

Finally, the three components are combined into a unique expression that is weighted with exponents $\alpha$, $\beta$, and $\gamma$:

```{math}
:label: ssim
SSIM(x,y) = [l(x,y)]^\alpha \cdot [c(x,y)]^\beta \cdot [s(x,y)]^\gamma
```

The expressions refer to constants $C_1$, $C_2$, and $C_3$ that are introduced for numerical stability. These three quantities are functions of the dynamic range of the pixel values L (L=255 for 8-bit gray-scale images) and two scalar constants $K_1 \ll 1$ and $K_2 \ll 1$. Traditionally, $K_1$  and $K_2$ are equal to 0.01, and 0.03, respectively. $C_1=(K_1,L)^2, C_2=(K_2,L)^2,  C_3=\frac{C_2}{2}$. In the original paper {cite}`wangImageQualityAssessment2004`, $\alpha= \beta = \gamma = 1$.

## Results of the Denoising Experiment

A total of 44 subjective assessments and 352 objective assessments were conducted using the IQMs described above. The Median filter with a kernel size of 3 produced the overall best quality in terms of noise reduction and preservation of detail in the breast tissue. The Gaussian filter was a close second; however, it did not preserve the detail in the breast tissue as well as the Median filter did. The Frequency Domain filters were effective on salt-n-pepper and speckle noise; however, the overall blurring effect was objectionable.
