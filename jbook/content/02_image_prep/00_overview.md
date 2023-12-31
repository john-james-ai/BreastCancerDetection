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
