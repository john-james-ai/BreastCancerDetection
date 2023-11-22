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
# Image Preprocessing Overview

Precise and accurate diagnosis of breast cancer rests upon the discriminatory power of mathematical models designed to detect and classify structural abnormalities in breast tissue from biomedical imaging. Advances in artificial intelligence and computer vision, fueled by an explosion in AI task-specific computational power, have given rise to dense image recognition models capable of distinguishing increasingly complex patterns and structures in biomedical images. Still, the diagnostic performance and clinical applicability of such models rests upon the availability of large datasets containing high-quality, high-resolution images that are clear, sharp, and free of noise and artifacts.

Exploratory analysis of the CBIS-DDSM mammograph illuminated several issues that compromise the discriminatory power of image detection and recognition models.

- Various artifacts (large texts and annotations) are present within the mammography that resemble the pixel intensities of the regions of interest (ROIs), which can interfere with the ROI extraction process and/or lead to false diagnosis.
- Noise of various types in the images is an obstacle to effective feature extraction, image detection, recognition, and classification.
- Poor brightness and contrast levels in some mammograms may increase the influence of noise, and/or conceal important and subtle features.
- Malignant tumors are characterized by irregular shapes and ambiguous or blurred edges that complicate the ROI segmentation task.
- Dense breast tissue with pixel intensities similar to that of cancerous tissue, may conceal subtle structures of diagnostic importance.
- Deep learning models for computer vision require large datasets. The CBIS-DDSM has just over 3500 full mammogram images, a relatively small dataset for model training.

Addressing these challenges is fundamentally important to model detection, recognition, and classification performance.

## Image Preprocessing Approach

In this regard, a five-stage image preprocessing approach ({numref}`image_prep`) has been devised to reduce noise in the images, eliminate artifacts, and produce a collection of images for maximally effective computer vision model training and classification.

```{figure} ../../figures/ImagePrep.png
---
name: image_prep
---
Image Preprocessing Approach

```

We begin with an evaluation of various denoising methods commonly applied to mammography. Once a denoising method and its (hyper) parameters are selected, we move to the artifact removal stage. Image binarizing and thresholding methods are evaluated, then morphological transformations are applied to the binarized images to remove artifacts. Next, the pectoral muscle is removed using various techniques such as Canny Edge Detection, Hough Lines Transformation, and Largest Contour Detection algorithms. To make malignant lesions more conspicuous during model training, we enhance image brightness and contrast with Gamma Correction and Contrast Limited Adaptive Histogram Equalization (CLAHE). Additive White Gaussian Noise (AWGN) is also added to improve the generalized performance of the neural network and mitigate model overfitting. Finally, we extract the ROIs using automated pixel intensity thresholding to create a binary mask which is applied to the enhanced images.

## Image Quality Assessment (IQA)

Image Quality Assessment (IQA) is the process of evaluating the extent to which an image preserves the information and appearance of the original image.  During the preprocessing stage, we will be evaluating various image enhancement methods in terms of the quality of the images they render. In this regard, our IQA will be a combination of subjective and objective assessment methods.

### Subjective Image Quality Assessment

Subjective methods are based on human subjects’ opinions of image quality {cite}`linLargeScaleCrowdsourcedSubjective2022`. Though time-consuming, expensive, and difficult to scale in real-time,
subjective methods are considered the most reliable methods for assessing image quality because they rely on the opinions of human subjects, who represent the ultimate users of the digital media application {cite}`afnanSubjectiveAssessmentObjective2023`.

Subjective assessments will be performed on a double stimulus comparison scale (DSCS), in which a random selection of test images is compared to its associated ground truth images. In addition,
images from the various image enhancement methods are visually evaluated and scored in terms of relative image quality.

### Objective Image Quality Assessment

Objective methods are based on image quality metrics (IQMs) that are designed to estimate the quality of the image automatically, in qualitative terms as observed by human subjects {cite}`afnanSubjectiveAssessmentObjective2023`. Image processing algorithms will be ranked by calculating these metrics to select the algorithm that produces the highest-quality images.

For this effort, three IQMs will be used to assess image quality: mean squared error (MSE), peak signal-to-noise ratio (PSNR), and the structural similarity index measure (SSIM).

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
\text{PSNR} = 20\text{log}_10\Bigg(\frac{(MAX)}{\sqrt{MSE}}\Bigg)
```

where MAX is the maximum pixel value contained in the image and PSNR is measured in decibels (dB). An acceptable PSNR for an 8-bit image would be in the range of 30 to 50 dB {cite}``beeravoluPreprocessingBreastCancer2021`.

#### Structural Similarity Index Measure (SSIM)

The Structural Similarity Index Measure (SSIM) is a method for measuring the structural similarity between two images. Since its introduction in 2004 {cite}`bakurovStructuralSimilarityIndex2022`, the SSIM has become one of the most popular full-reference image quality assessment (FR-IQA) measures (over 47,960 citations), owing its success to its mathematical simplicity, low computational complexity, and implicit incorporation of the Human Visual System’s (HVS) characteristics. The incorporation of this characteristic has resulted in a better correlation with subjective evaluation provided by human observers. Consequently, SSIM has been used as a proxy evaluation for human assessment in a range of image processing and computer vision applications.

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
c(x,y) = \frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2 + \sugna_y^2 + C_2},
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
SSIM(x,y) = [l(x,y)\^\alpha \cdot [c(x,y)]^\beta \cdot [s(x,y)]^\gamma
```

The expressions refer to constants $C_1$, $C_2$, and $C_3$ that are introduced for numerical stability. These three quantities are functions of the dynamic range of the pixel values L (L=255 for 8-bit gray-scale images) and two scalar constants $K_1 \ll 1$ and $K_2 \ll 1$. Traditionally, $K_1$  and $K_2$ are equal to 0.01, and 0.03, respectively. $C_1=(K_1,L)^2, C_2=(K_2,L)^2,  C_3=\frac{C_2}{2}$. In the original paper {cite}`wangImageQualityAssessment2004`, $\alpha= \beta = \gamma = 1$.
