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

# Exploratory Data Analysis
## CBIS-DDSM Dataset
Developed in 1997 through a grant from the DOD Breast Cancer Research Program and the US Army Research and Material Command, the Digital Database for Screening Mammography (DDSM) {cite}`USFDigitalMammography` is a collection of 2620 cases obtained by patient consent from Massachusetts General Hospital, Wake Forest University School of Medicine, Sacred Heart Hospital, and Washington University of St. Louis School of Medicine. Its cases are annotated with ROIs for calcifications and masses, they include Breast Imaging Reporting and Data System (BI-RADS) descriptors for mass shape, mass margin, calcification type, calcification distribution, and breast density, and an overall BI-RADS assessment from 0 to 5, and a rating of the subtlety of the abnormality from 1 to 5.

The DDSM, a powerful and extensively used resource for imaging research, presented certain challenges in terms of accessibility and practical utility. For instance, the original DDSM is saved in a non-standard compression format no longer supported in modern computer systems. Regions of interest mark the general area of legions; but, lack the specificity of precise image segmentation, requiring researchers to implement segmentation algorithms for accurate feature extraction.

This Curated Breast Imaging Subset of DDSM (CBIS-DDSM) {cite}`leeCuratedMammographyData2017` is an updated and standardized version of the original DDSM, developed to address some of the challenges of the DDSM. The original DDSM images were distributed as lossless JPEG files (LJPEG); an obsolete image format. Raw pixel data were converted into 64-bit optical density values, then re-mapped to 16-bit grayscale TIFF format and finally converted to DICOM format for the data repository. Linux bash and C tools were re-implemented in Python to support cross-platform accessibility. A set of convenience images were cropped around the region of interest for researchers analyzing only the abnormalities and not the full mammogram image. Metadata, including patient age, date of the study, date of digitization, the dense tissue category, the scanner used to digitize, and the resolution of the image, stored in .ics files were extracted and compiled into a single CSV file. Three-hundred thirty-nine images deemed to have annotations of suspicious lesions that could not be seen, were removed from the dataset.  Segmentation was performed for the mass images and the data were split into train (80%) and test (20%) sets for method evaluation purposes.

CBIS-DDSM was obtained from the Cancer Imaging Archive {cite}`sawyer-leeCuratedBreastImaging2016` and is summarized below:

| Collection Statistics  |        |
|------------------------|--------|
| Image Size (GB)        | 163.6  |
| Modalities             | MG     |
| Number of Images       | 10239  |
| Number of Participants | 1,566* |
| Number of Series       | 6775   |
| Number of Studies      | 6775   |

Note, the data are structured such that a single participant has multiple patient IDs, where each id corresponds to a scan.

The following files contain the mammography and ROIs for the cases with calcifications.

| Type   | Filename                              | Format |
|--------|---------------------------------------|--------|
| Images | Calc-Test Full Mammogram Images       | DICOM  |
| Images | Calc-Test ROI and Cropped Images      | DICOM  |
| Images | Calc-Training Full Mammogram Images   | DICOM  |
| Images | Calc-Training ROI and Cropped Images  | DICOM  |

The following files contain the mammography and ROIs for the mass cases.

| Type   | Filename                              | Format |
|--------|---------------------------------------|--------|
| Images | Mass-Test Full Mammogram Images       | DICOM  |
| Images | Mass-Test ROI and Cropped Images      | DICOM  |
| Images | Mass-Training Full Mammogram Images   | DICOM  |
| Images | Mass-Training ROI and Cropped Images  | DICOM  |

There are separate metadata files for the train and test calcification and mass cases.

| Type     | Filename                   | Format |
|----------|----------------------------|--------|
| Metadata | Calc-Test-Description      | csv    |
| Metadata | Calc-Training-Description  | csv    |
| Metadata | Mass-Test-Description      | csv    |
| Metadata | Mass-Training-Description  | csv    |

Metadata for each abnormality above contains:
- Patient ID: the first 7 characters of images in the case file
- Density category
- Breast: Left or Right
- View: CC or MLO
- Number of abnormality for the image (This is necessary as there are some cases containing multiple abnormalities.
- Mass shape (when applicable)
- Mass margin (when applicable)
- Calcification type (when applicable)
- Calcification distribution (when applicable)
- BI-RADS assessment
- Pathology: Benign, Benign without call-back, or Malignant
- Subtlety rating: Radiologists’ rating of difficulty in viewing the abnormality in the image
- Path to image files

```{code-cell} ipython3
import warnings
warnings.filterwarnings("ignore")
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from PIL import Image
from bcd.data.dataset import CBISMeta
```

```{code-cell} ipython3
fpc = "data/raw/meta/calc_case_description_train_set.csv"
fpm = "data/raw/meta/mass_case_description_train_set.csv"
```

## Calcifications

```{code-cell} ipython3
dfc = pd.read_csv(fpc)
dsc = CBISMeta(df=dfc)
```

```{code-cell} ipython3
dsc.head()
```

```{code-cell} ipython3
dsc.info
```
