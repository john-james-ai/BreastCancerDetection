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

# CBIS-DDSM - Unbox

Before conducting data quality and exploratory analyses, we'll unbox the data to ensure that, well...*' we got what we paid for*. In the next section, we'll conduct an analysis of the data quality along dimensions of validity, completeness, consistency and uniqueness. The exploratory analysis is designed to harvest insight from the data. Our aim here, is to get a general sense of the data *structure* and to make any structural changes necessary to facilitate the next stages of the analysis. First, we'll examine the case training sets, then the series metadata, and finally, we'll discover the structure of the DICOM metadata.

```{code-cell} ipython3
:tags: [hide-input]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))
from myst_nb import glue
import pandas as pd

fp_calc_train = "data/meta/0_raw/calc_case_description_train_set.csv"
fp_mass_train = "data/meta/0_raw/mass_case_description_train_set.csv"
fp_metadata = "data/meta/0_raw/metadata.csv"
```

## Case Data
### Calcification Training Cases

```{code-cell} ipython3
df = pd.read_csv(fp_calc_train)
df.info()
```

We have 1546 observations and 14 columns *(with spaces in the column names, which is somewhat vexing)* in the calcification training set. Let's take a look at the data.

```{code-cell} ipython3
calc_train_sample = df.sample(n=5, random_state=57)
glue('calc_train_sample', calc_train_sample)
```

```{glue:figure} calc_train_sample
---
align: center
name: calc_train_sample_fig
---
Calcification Training Set Samples
```

+++

Ok, let's take a look at the mass training set.

+++

### Mass Train Cases

```{code-cell} ipython3
:tags: [hide-input]

df = pd.read_csv(fp_mass_train)
df.info()
```

And the data...

```{code-cell} ipython3
mass_train_sample = df.sample(n=5, random_state=240)
glue('mass_train_sample', mass_train_sample)
```

```{glue:figure} mass_train_sample
---
align: center
name: mass_train_sample_fig
---
Calcification Training Set Samples
```

+++

Ok, a few things stand out thus far.

1. The case datasets have inconsistent variable names. For instance, the calcification set has 'breast density' and the mass case provides 'breast_density'. Our first task is to ensure that the variables are consistently named across case files.
2. The data are currently split into train and test sets by BI-RADS category. While convenient for modeling, this is rather cumbersome for the exploratory data analysis which considers the entire dataset, before making any assumptions. Our second task is to combine the calcification and mass train and test sets into a single case dataset, facilitating a top-down, comprehensive image view of all the data for analysis.
3. The target variable, pathology, has three values: 'MALIGNANT', 'BENIGN', and 'BENIGN_WITHOUT_CALLBACK. The latter indicates that some aspect of the case is worth tracking; but, no follow-up is required. For our purposes, that is a distinction without a difference. Nonetheless, rather than modifying the variable and losing potentially useful information, we'll create a new target variable, 'cancer', which will be True if the pathology is 'MALIGNANT', and False otherwise.
4. The CBIS-DDSM creators define a case as a particular abnormality as seen in the cranial caudal (CC) or mediolateral oblique (MLO) image_views; yet, the dataset lacks a unique case identifier. Yet, distinguishing cases involves parsing the patient_id, abnormality type, the file set, left or right breast, and  image_view. An alternative is to parse the file paths for study_uid and series_uid in order to map the case to an image. 

+++

## Series Metadata
The series metadata contains study and series information for the DICOM files that were downloaded from the TCIA. 

```{code-cell} ipython3
df = pd.read_csv(fp_metadata)
df.info()
```

```{code-cell} ipython3
series_data = df.sample(n=5, random_state=55)
full_mammogram_image_count_per_location = df.loc[df['series_description'] == 'full mammogram images']['number_of_images'].max()
cropped_image_count_per_location = df.loc[df['series_description'] == 'cropped images']['number_of_images'].max()
roi_image_count_per_location =  df.loc[df['series_description'] == 'ROI mask images']['number_of_images'].max()

glue('full_mammogram_image_count_per_location', full_mammogram_image_count_per_location)
glue('cropped_image_count_per_location', cropped_image_count_per_location)
glue('roi_image_count_per_location', roi_image_count_per_location)
glue('series_data', series_data)
```

```{glue:figure} series_data
---
align: center
name: series_data_fig
---
Series Metadata
```

+++

The most important variables in this dataset are:
- series_uid
- subject_id
- study_uid
- study_description
- number_of_images
- file_location

Still, several observations can be made at this stage:
1. Each full mammogram file location has {glue:}`full_mammogram_image_count_per_location`, but each ROI and cropped image folder may have upwards of two images. Since an observation in this dataset can pertain to one or more images, there is no way to uniquely identify a cropped or ROI image in the dataset.

Let's examine a specific case and see how it is represented in the dataset. Patient P_00112 has five abnormalities. Let's check the case dataset.

```{code-cell} ipython3
df_calc = pd.read_csv(fp_calc_train)
df_calc_P_01740 = df_calc.loc[df_calc['patient_id'] == 'P_01740']
df_calc_P_01740
```

```{code-cell} ipython3
df_series = pd.read_csv(fp_metadata)
df_series_P_01740 = df_series.loc[df_series['subject_id'].str.contains('P_01740')]
df_series_P_01740
```
