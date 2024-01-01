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
fp = "data/image/0_raw/Mass-Training_P_01981_RIGHT_CC/07-20-2016-DDSM-94258/1.000000-full mammogram images-07312/1-1.dcm"
pydicom.dcmread(fp)
```

## Summary of Structural Concerns

1. The data are currently split into train and test sets by BI-RADS category. While convenient for modeling, this is rather cumbersome for the exploratory data analysis which considers the entire dataset, before making any assumptions. As such, bringing the calcification and mass train and test sets together with the associated full mammogram image metadata into a single dataset will facilitate a top-down, comprehensive view of all the data for analysis.
2. Our binary classification target variable can take one of three values: 'MALIGNANT', 'BENIGN', and 'BENIGN_WITHOUT_CALLBACK. The latter indicates that some aspect of the case is worth tracking; but, no follow-up is required. For our purposes, that is a distinction without a difference. Nonetheless, rather than modifying the variable and losing potentially useful information, we'll create a new target variable, 'cancer', which will be True if the pathology is 'MALIGNANT', and False otherwise.
3. The CBIS-DDSM creators define a case as a particular abnormality as seen in the cranial-caudal (CC) or mediolateral oblique (MLO) image views; yet, there is no formal identification of a unique *mammogram*. A mammogram is defined here as a CC or MLO image of the left or right breast taken for a specific patient. Such a mammogram identifier would allow us to bring the case and the *full* mammogram imaging data together into a single dataset. ROI masks and cropped images are abnormality-level constructs and can remain as such.

Overall, this is an advocation for a single dataset containing only the information relevant to the analyses and modeling tasks. In the next section, a single task-specific dataset will be harvested from the CBIS-DDSM data.
