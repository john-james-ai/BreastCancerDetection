---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: bcd
  language: python
  name: python3
---

(eda3)=

# EDA Part 3: Data Preparation

In the prior section, we obtained the CBIS-DDSM data from [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/image_viewpage.action?pageId=22516629) (TCIA). This brief section aims to convert the data to a form well suited for exploratory data analyses.

| # | Dataset              | Description                                                          |
| - | -------------------- | -------------------------------------------------------------------- |
| 1 | calc_cases.csv       | Calcification cases from the training and test sets.                 |
| 2 | mass_cases.csv       | Mass cases from the training and test sets.                          |
| 3 | case_series_xref.csv | Case / Series Cross-Reference                                        |
| 4 | dicom.csv            | DICOM dataset containing resolution and pixel properties each image. |

First, we'll build the case datasets (1,2,3), then we'll construct the DICOM image metadata and quality assessment dataset (4).

```{code-cell}
:tags: [hide-input]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))

import pandas as  pd

from bcd.explore.prep.case import CasePrep
from bcd.explore.prep.dicom import DicomPrep
```

## Build Case Dataset

The data are currently split into train and test sets by BI-RADS category. While convenient for modeling, this is rather cumbersome for the exploratory data analysis which considers the entire dataset, before making any assumptions. Our first task is to combine the calcification and mass train and test sets into a single case dataset, facilitating a top-down, comprehensive image_view of all the data for analysis.

The target variable, pathology, has three values: 'MALIGNANT', 'BENIGN', and 'BENIGN_WITHOUT_CALLBACK. The latter indicates that some aspect of the case as worth tracking; but, no follow-up is required. For our purposes, that is a distinction without a difference. Nonetheless, rather than modifying the variable and lose potentially useful information, we'll create a new target variable, 'cancer', which will be True if pathology is 'MALIGNANT', and False otherwise.

The CBIS-DDSM creators define a case as a particular abnormality as seen in the cranial caudal (CC) or mediolateral oblique (MLO) image_views; yet, the dataset lacks a unique case identifier. Consequently, five different variables are required to map metadata with their associated image. To enable direct matching between metadata and DICOM images, each case will be assigned a unique identifier, which will be cross-referenced to a full mammogram, ROI mask, or cropped image series dataset.

```{code-cell}
:tags: [hide-cell]

CALC_TRAIN_FP = "data/meta/0_raw/calc_train.csv"
CALC_TEST_FP = "data/meta/0_raw/calc_test.csv"
MASS_TRAIN_FP = "data/meta/0_raw/mass_train.csv"
MASS_TEST_FP = "data/meta/0_raw/mass_test.csv"
CASE_FP = "data/meta/1_staged/cases.csv"
CASE_SERIES_XREF_FP = "data/meta/1_staged/case_series_xref.csv"
```

```{code-cell}
case = CasePrep()
df, df_xref = case.prep(calc_train_fp=CALC_TRAIN_FP, calc_test_fp=CALC_TEST_FP, mass_train_fp=MASS_TRAIN_FP, mass_test_fp=MASS_TEST_FP, case_fp=CASE_FP, case_series_fp=CASE_SERIES_XREF_FP, result=True, force=False)
print(f"The Case Dataset has been created with {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"The Case/Series XRef Dataset has been created with {df_xref.shape[0]} rows and {df_xref.shape[1]} columns.")
```

## Build DICOM Dataset

Developing an image preprocessing approach requires visibility into image metadata, currently buried in individual DICOM files. Our first task is to extract these data and store them in a more accessible format. Note: There are 10,239 images in the dataset according to [TCIA](https://wiki.cancerimagingarchive.net/pages/image_viewpage.action?pageId=22516629#2251662935562334b1e043a3a0512554ef512cad). The DICOM image file for mmg_id 'P_01382_LEFT_mass_MLO_1' was corrupt, and could not be read. Therefore, we will have a total of 10,238 images.

```{code-cell}
LOCATION = "data/image/0_raw"
DICOM_FP = "data/meta/1_staged/dicom.csv"
SKIP_LIST = ["Mass-Training_P_01382_LEFT_MLO/07-20-2016-DDSM-93921/1.000000-full mammogram images-05891/1-1.dcm"]
```

```{code-cell}
dicom = DicomPrep()
dfd = dicom.prep(location=LOCATION, dicom_fp=DICOM_FP, skip_list=SKIP_LIST, result=True, force=False)
dfd = dicom.add_series_description(dicom_fp=DICOM_FP, series_fp=CASE_SERIES_XREF_FP)
print(f"The DICOM Dataset has been created with {dfd.shape[0]} rows and {dfd.shape[1]} columns.")
```
