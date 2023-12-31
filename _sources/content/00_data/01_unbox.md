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

# Unboxing CBIS-DDSM - A Structural Analysis

Before conducting the exploratory analyses, we'll unbox the data to ensure that the structure supports the analyses and that ...*' we got what we paid for*. In the next section, we'll perform an analysis of the data quality along dimensions of validity, completeness, consistency, and uniqueness. The following exploratory analysis will harvest insight from the data. 

Our aim here is to get a general sense of the data *structure* and to make any structural changes necessary to facilitate the next stages of the analysis. First, we'll examine the case training sets, then the series metadata, and finally, we'll discover the structure of the DICOM metadata.

```{code-cell} ipython3
:tags: [hide-input]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))
from myst_nb import glue
import pandas as pd
import pydicom

fp_calc_train = "data/meta/0_raw/calc_case_description_train_set.csv"
fp_mass_train = "data/meta/0_raw/mass_case_description_train_set.csv"
fp_metadata = "data/meta/0_raw/metadata.csv"
```

## Case Data

### Calcification Training Cases

```{code-cell} ipython3
:tags: [hide-input]

df = pd.read_csv(fp_calc_train)
df.info()
```

We have 1546 observations and 14 columns *(with spaces in the column names, which is somewhat vexing)* in the calcification training set. Let's take a look at the data.

```{code-cell} ipython3
df.sample(n=5, random_state=57)
```

Ok, let's take a look at the mass training set.

### Mass Train Cases

```{code-cell} ipython3
:tags: [hide-input]

df = pd.read_csv(fp_mass_train)
df.info()
```

And the data...

```{code-cell} ipython3
:tags: [hide-input]

df.sample(n=5, random_state=240)
```

Ok, let's inspect the series metadata.

+++

## Series Metadata

The series metadata contains study and series information for the DICOM files that were downloaded from the TCIA.

```{code-cell} ipython3
:tags: [hide-input]

df = pd.read_csv(fp_metadata)
df.info()
```

```{code-cell} ipython3
:tags: [hide-input]

df.sample(n=5, random_state=55)
```

Very alright! Now the DICOM image metadata.

## DICOM Image Metadata

```{code-cell} ipython3
fp = "data/image/0_raw/Calc-Training_P_01823_LEFT_MLO/08-07-2016-DDSM-99626/1.000000-full mammogram images-81312/1-1.dcm"
pydicom.dcmread(fp)
```

## Summary of Structural Concerns

### Cases

1. The data are currently split into train and test sets by BI-RADS category. While convenient for modeling, this is rather cumbersome for the exploratory data analysis which considers the entire dataset, before making any assumptions. Combining the calcification and mass train and test sets into a single case dataset will facilitate a top-down, comprehensive view of all the data for analysis.
2. Our binary classification target variable can take one of three values: 'MALIGNANT', 'BENIGN', and 'BENIGN_WITHOUT_CALLBACK. The latter indicates that some aspect of the case is worth tracking; but, no follow-up is required. For our purposes, that is a distinction without a difference. Nonetheless, rather than modifying the variable and losing potentially useful information, we'll create a new target variable, 'cancer', which will be True if the pathology is 'MALIGNANT', and False otherwise.
3. The CBIS-DDSM creators define a case as a particular abnormality as seen in the cranial-caudal (CC) or mediolateral oblique (MLO) image views; yet, there is no way to uniquely identify a case without parsing patient_id, abnormality type, the file set, left or right breast, and image_view. A unique case_id variable will uniquely identify each case, defined as a specific abnormality viewed by either a CC or MLO view. Distinguishing a view as a separate case allows us to link more precisely to individual  DICOM full mammogram images. 
4. Finally, there are inconsistencies in the variable names and the file paths are not valid, although the file paths contain series_uid and study_uid, which may serve as another link to the DICOM dataset.

### Series Metadata

1. The most important variables in series metadata dataset are:

```{table} Series Metadata
:name: series_metadata
| Field             | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| series_uid        | DICOM Series Instance UID: Tag (0020, 000E)                        |
| subject_id        | Abnormality type, fileset, patient_id, breast side, and image view |
| study_uid         | DICOM Study Instance UID: Tag (0020,000D)                          |
| study_description | full mammography images, cropped images, or ROI images             |
| number_of_images  | Number of images at the file location                              |
| file_location     | Relative directory path to image(s)                                |
```

2. The subject_id {numref}`series_metadata` can be used to map the DICOM images to cases. Still, since a file location can have more than one image, there is no way to uniquely identify an image. This is less of a problem for full mammogram images, as there is only one per file location; however, we may have two cropped or ROI images per location. A unique identifier per image will simplify the process of image-to-case mapping.

### DICOM Image Metadata

The DICOM structure has several variables of interest:
1.  Series description
2.  Patient's Name / Patient ID
3.  Study Instance UID
4.  Series Instance UID
5.  Patient Orientation
6.  Samples per Pixel
7.  Rows and Columns
8.  Bits Allocated and Stored
9.  Smallest and Largest Image Pixel Value
10. Pixel Data

Two additional variables not included in the DICOM structure would be of interest: Mean Pixel Value and Standard Deviation of the Pixel Values. For ease of accessibility, these data can be extracted into a format that supports DataFrames.

Finally, images do not have a single identifier that uniquely defines each image. This is especially true for ROI and cropped images which may share a common study_uid and series_uid. 

That summarizes the structural concerns from this unboxing effort. Next, we'll implement the structural changes raised in this section.
