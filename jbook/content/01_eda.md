---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Exploratory Data Analysis

**CBIS-DDSM Dataset**

Developed in 1997 through a grant from the DOD Breast Cancer Research Program and the US Army Research and Material Command, the Digital Database for Screening Mammography (DDSM) {cite}`USFDigitalMammography` is a collection of 2620 cases obtained by patient consent from Massachusetts General Hospital, Wake Forest University School of Medicine, Sacred Heart Hospital, and Washington University of St. Louis School of Medicine. Its cases are annotated with ROIs for calcifications and masses, they include Breast Imaging Reporting and Data System (BI-RADS) descriptors for mass shape, mass margin, calcification type, calcification distribution, and breast density. They also include overall BI-RADS assessments from 0 to 5 and ratings of the subtlety of the abnormalities from 1 to 5.

The DDSM, a powerful and extensively used resource for imaging research, presented certain challenges that limited its accessibility and utility.  For instance, the original DDSM was saved in lossless JPEG files (LJPEG); a non-standard, and obsolete compression format. Regions of interest marked the general area of legions; but, lacked the specificity of precise image segmentation, requiring researchers to implement segmentation algorithms for accurate feature extraction. Some annotations marked legions that could not be seen. Further, the metadata processing and image correction routines distributed with the dataset were a set of bash and C routines that were difficult to refactor.
This Curated Breast Imaging Subset of DDSM (CBIS-DDSM) {cite}`leeCuratedMammographyData2017` is an updated and standardized version of the original DDSM, developed to address some of the challenges. In particular, the questionable annotations were reviewed by a trained mammographer, and 254 images were removed from the dataset. The images were converted from PVRG-JPEG to 8-bit raw binary bitmaps. Python tools were developed to convert the 8-bit images to 16-bit grayscale TIFF files, which were later converted to DICOM. The bash and C preprocessing tools were re-implemented in Python to be cross-platform. Convenience images were cropped around the region of interest for researchers analyzing only the abnormalities. Precise segmentation was applied to the calcification images providing much more accurate regions of interest, and the data were split into training and test sets, based on the BIRADS category to support method evaluation and reproducibility.

CBIS-DDSM was obtained from the Cancer Imaging Archive {cite}`sawyer-leeCuratedBreastImaging2016` and is summarized below:

| Collection Statistics  |        |
| ---------------------- | ------ |
| Image Size (GB)        | 163.6  |
| Modalities             | MG     |
| Number of Images       | 10239  |
| Number of Participants | 1,566* |
| Number of Series       | 6775   |
| Number of Studies      | 6775   |

Note, the data are structured such that a single participant has multiple patient IDs, where each id corresponds to a scan.

The following files contain the mammography and ROIs for the cases with calcifications.

| Type   | Filename                             | Format |
| ------ | ------------------------------------ | ------ |
| Images | Calc-Test Full Mammogram Images      | DICOM  |
| Images | Calc-Test ROI and Cropped Images     | DICOM  |
| Images | Calc-Training Full Mammogram Images  | DICOM  |
| Images | Calc-Training ROI and Cropped Images | DICOM  |

The following files contain the mammography and ROIs for the mass cases.

| Type   | Filename                             | Format |
| ------ | ------------------------------------ | ------ |
| Images | Mass-Test Full Mammogram Images      | DICOM  |
| Images | Mass-Test ROI and Cropped Images     | DICOM  |
| Images | Mass-Training Full Mammogram Images  | DICOM  |
| Images | Mass-Training ROI and Cropped Images | DICOM  |

There are separate metadata files for the train and test calcification and mass cases.

| Type     | Filename                  | Format |
| -------- | ------------------------- | ------ |
| Metadata | Calc-Test-Description     | csv    |
| Metadata | Calc-Training-Description | csv    |
| Metadata | Mass-Test-Description     | csv    |
| Metadata | Mass-Training-Description | csv    |

Metadata for each abnormality above contains:

- Patient ID: the first 7 characters of images in the case file
- Density category
- Breast: Left or Right
- View: CC or MLO
- Number of abnormality for the image (This is necessary as there are some cases containing multiple abnormalities.)
- Mass shape (when applicable)
- Mass margin (when applicable)
- Calcification type (when applicable)
- Calcification distribution (when applicable)
- BI-RADS assessment
- Pathology: Benign, Benign without call-back, or Malignant
- Subtlety rating: Radiologists’ rating of difficulty in viewing the abnormality in the image
- Path to image files

```{code-cell} ipython3
:tags: [hide-cell]

import sys
import os
os.chdir(os.path.abspath(os.path.join("../..")))
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from PIL import Image
from bcd.data.dataset import CBISMeta
sns.set_style('whitegrid')
sns.set_palette("Blues_r")
```

We'll begin our exploration with questions to guide the process.

1. Density is considered a potential signal of pathology. How does density relate?
2. For calcifications, what does the calcification type indicate with respect to the BIRADS assessment, and pathology?
3. What does the distribution of calcification indicate?
4. How does mass shape relate to pathology?
5. Are there interactions among density, shape, and type features?
6. What distinguishes the subtle cases from others?

We'll approach calcification cases first, then move to the mass cases.

## Calcification Cases

```{code-cell} ipython3
fp = "/home/john/projects/bcd/data/raw/meta/calc_case_description_train_set.csv"
df = pd.read_csv(fp)
ds = CBISMeta(df=df)
```

```{code-cell} ipython3
ds.head()
```

```{code-cell} ipython3
ds.info
```

**Initial observations:**
1. We have 1546 calcification images in the dataset.
2. There 602 unique patient ids in which multiple patient ids per patient are extant.
3. With the exception of calcification type and distribution, we have no null values.
4. There are 45 calcification types and 9 calcification distribution values.

### Univariate Analysis

+++

#### Patient Id

```{code-cell} ipython3
print(f"The total number of unique ids {df['patient_id'].nunique()}.")
print(f"The median number of images per patient id is {df['patient_id'].value_counts().median()}")
print(f"The average number of images per patient id is {round(df['patient_id'].value_counts().mean(),2)}")
print(f"The range of images to patient_ids is {df['patient_id'].value_counts().min()} to {df['patient_id'].value_counts().max()} ")
```

#### Breast Density

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12,4))
ds.plot.countplot(x=df['breast density'], ax=ax, title ="Distribution of Breast Density")
```

#### Left Right

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12,4))
ds.plot.countplot(x=df['left or right breast'], ax=ax, title ="Distribution of View")
```

#### Image View

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12,4))
ds.plot.countplot(x=df['image view'], ax=ax, title ="Distribution of Image View")
```

#### Abnormality Id

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12,4))
ds.plot.countplot(x=df['abnormality id'], ax=ax, title ="Distribution of Abnormaility ID")
```

Most patients have only one abnormality.

#### Calcification Type

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8,8))
ds.plot.countplot(y=df['calc type'], ax=ax, title ="Distribution of Calcification Type")
```

#### Calcification Distribution

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12,4))
ds.plot.countplot(y=df['calc distribution'], ax=ax, title ="Distribution of Calcification Distribution")
```

#### BIRADS Assessment

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12,4))
ds.plot.countplot(x=df['assessment'], ax=ax, title ="Distribution of BIRADS Assessment")
```

#### Pathology

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12,4))
ds.plot.countplot(x=df['pathology'], ax=ax, title ="Distribution of Pathology")
```

#### Sublety

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12,4))
ds.plot.countplot(x=df['subtlety'], ax=ax, title ="Distribution of Subtlety")
```

### Bivariate Analysis
At this stage, we are examining bivariate relationships between the variables and between the variables and the target, pathology.
