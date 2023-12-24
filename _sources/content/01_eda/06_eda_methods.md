---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
---
(eda6)=

# EDA Part 6: Algorithm Selection

Image data processing will involve denoising the images, artifact and pectoral muscle removal, image enhancement, and augmentation techniques in advance of statistical modeling. Model performance will, in part, rest on the selection of these algorithms and the degree to which their parameters have been optimized for the dataset.

The goal of this section is to select and optimize the algorithms that will be used during the preprocessing stage. Specifically, our focus will be on selecting a denoiser algorithm and a binary threshold algorithm for breast segmentation.
In section {numref}`eda61`, we will review the types of noise inherent in digital mammography, review the most widely used denoising methods, and then evaluate, select, and parameterize the denoising algorithm that will be used downstream. Next, {numref}`eda52` will review, select, and tune a binary threshold algorithm for breast segmentation.
