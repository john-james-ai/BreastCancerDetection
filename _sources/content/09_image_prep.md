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

Discriminating between benign and malignant lesions in mammograms involves the detection and analysis of different structural changes in the breast tissue. Deep learning algorithms, specifically, convolutional neural networks, can extract features from regions of interest (ROIs) based on pixel intensity; however, this task is complicated by:

- the structural complexity of ROIs,
- presence of artifacts (large texts and annotations) in the mammograms which resemble the pixel intensity of the ROI,
- noise in the form of random variations in pixel intensity that may have been produced during image capture,
- poor brightness and contrast levels in some mammograms,
- dense breast tissue with pixel intensities similar to that of cancerous tissue, and
- the limited number of mammogram images available for model training.

The performance of a breast cancer detection and classification model rests upon the degree to which these issues are addressed during the image preprocessing stage.  This section describes the image preprocessing approach for the CBIS-DDSM dataset in terms of the methods employed and their comparative performance evaluation experiments. The following figure depicts the image preprocessing procedure.

```{figure} ../figures/ImagePrep.png
---
name: image_prep
---
Image preprocessing procedure of this study.
```
