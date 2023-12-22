---
jupytext:
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

+++

# Exploratory Data Analysis (EDA) 1.0 - Metadata

In this section, we conduct an exploratory data analysis of the CBIS-DDSM Case and DICOM datasets prepared in prior sections. The purpose of this EDA is three-fold:

1. Discover the relationships among the features
2. Explore the nature of the relationships between the features and the diagnostic target.
3. Identify features that have the greatest influence classification accuracy.

## The Dataset

In this section, we will be analyzing the Case Dataset: calcification and mass datasets containing patient, abnormality, BI-RADS assessment, image image_view, breast density, subtlety, and pathology (diagnosis) information. The dataset dictionary is outlined in {numref}`eda1_case_dataset`.

```{table} Case Dataset Dictionary
:name: eda1_case_dataset

| #  | Variable             | Type        | Description                                                                                                                  |
|----|----------------------|-------------|------------------------------------------------------------------------------------------------------------------------------|
| 1  | patient_id           | Nominal     | Unique identifier for each patient.                                                                                          |
| 2  | breast_density       | Discrete    | BI-RADS overall assessment of the volume of attenuating tissues in the breast.                                             |
| 3  | left_or_right_breast | Nominal     | Which breast was imaged.                                                                                                     |
| 4  | image_view           | Dichotomous | Either cranialcaudal or mediolateral oblique image_view.                                                                            |
| 5  | abnormality_id       | Discrete    | Number of abnormalities for the patient.                                                                                       |
| 6  | abnormality_type     | Dichotomous | BI-RADS category of the abnormality.                                                                                         |
| 7  | calc_type            | Nominal     | Characterization of the type of calcification (where applicable)                                                             |
| 8  | calc_distribution    | Nominal     | The arrangement of the calcifications inside the breast and, relative to the probability of malignancy. (where applicable) |
| 9  | mass_shape           | Nominal     | Shape of the mass                                                                                                            |
| 10 | mass_margins         | Nominal     | Feature that separates the mass from the adjacent breast parenchyma.                                                         |
| 11 | assessment           | Discrete    | Overall BI-RADS assessment of the mammography                                                                                |
| 12 | pathology            | Nominal     | Determination of the malignancy of the case.                                                                                 |
| 13 | subtlety             | Discrete    | Degree of diagnostic difficulty                                                                                              |
| 14 | fileset              | Nominal     | Indicates training or test set.                                                                                              |
| 15 | case_id              | Nominal     | Unique identifier for the case.                                                                                              |
| 16 | cancer               | Dichotomous | Indicates whether the cancer is diagnosed.                                                                                   |
```

## Guiding Questions

```{tip}
“Far better an approximate answer to the right question, which is often vague, than an exact answer to the wrong question, which can always be made precise.” — John Tukey
```

Here, we'll put forward a set of questions to motivate and guide the discovery process.

1. What are the relationships between calcification and mass morphological features and malignancy?
2. To what degree does breast density relate to abnormality types and malignancy?
3. Are certain abnormalities more or less subtle?
4. What are the relative features of importance concerning screening?

## Exploratory Data Analysis Plan

The EDA will be conducted in three primary stages:

1. **Univariate**: Examination of the variables independently
2. **Bivariate**: Evaluate the relations among the features and between the features and the target.
3. **Multivariate**: Discover feature importance w.r.t. screening and diagnosis.

+++ {"tags": ["remove-output", "hide-input"]}

```{code-cell}
:tags: [remove-cell, hide-input]

import sys
import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))
import warnings
warnings.filterwarnings("ignore")
```

```{code-cell}
:tags: [hide-input]

import pandas as pd
from scipy import stats
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from bcd.explore.meta.case import CaseExplorer
from bcd.explore.meta.multivariate.pipeline import PipelineBuilder
from bcd.explore.meta.multivariate.selection import ModelSelector

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
sns.set_style('whitegrid')
sns.set_palette("Blues_r")
```

```{code-cell}
:tags: [remove-cell, hide-input]

CASE_FP = "data/meta/3_cooked/cases.csv"
```

```{code-cell}
:tags: [hide-input]

cases = CaseExplorer(filepath=CASE_FP)
calc = CaseExplorer(df=cases.get_calc_data())
mass = CaseExplorer(df=cases.get_mass_data())
```

+++

## Univariate Analysis

Let's get an overall sense of the data.

```{code-cell}
:tags: [remove-output]

summary = cases.summary()
glue("eda1_summary", summary)
```

```{glue:figure} eda1_summary
---
align: center
name: eda1_summary_df
---
CBIS-DDSM Dataset Summary
```

```{code-cell}
:tags: [hide-input]

st = summary.T
pct_calc = round(st['Calcification Cases'] / st['Cases'] * 100,2).values[0]
pct_mass = round(st['Mass Cases'] / st['Cases'] * 100,2).values[0]
pct_calc_mal = round(st['Calcification Cases - Malignant'] / st['Calcification Cases'] * 100,2).values[0]
pct_calc_bn = round(st['Calcification Cases - Benign'] / st['Calcification Cases'] * 100,2).values[0]
pct_mass_mal = round(st['Mass Cases - Malignant'] / st['Mass Cases'] * 100,2).values[0]
pct_mass_bn = round(st['Mass Cases - Benign'] / st['Mass Cases'] * 100,2).values[0]
cases_per_patient = round(st['Cases'] / st['Patients'],2).values[0]

glue("pct_calc", pct_calc)
glue("pct_mass", pct_mass)
glue("pct_calc_mal", pct_calc_mal)
glue("pct_calc_bn", pct_calc_bn)
glue("pct_mass_mal", pct_mass_mal)
glue("pct_mass_bn", pct_mass_bn)
glue("cases_per_patient", cases_per_patient)
```

From {numref}`eda1_summary_df`, several observations can be made:

1. We have 3566 cases, {glue:}`pct_calc`% are calcification cases and {glue:}`pct_mass`% are mass cases.
2. Of the calcification cases, {glue:}`pct_calc_bn`% are benign and {glue:}`pct_calc_mal`% are malignant.
3. Of the mass cases, {glue:}`pct_mass_bn`% are benign and {glue:}`pct_mass_mal`% are malignant.
4. On average, we have approximately {glue:}`cases_per_patient` cases per patient.

Case, as defined in {cite}`leeCuratedMammographyData2017`, indicates a particular abnormality, seen on the craniocaudal (CC) and/or mediolateral oblique (MLO) views.

+++

Let's take a look at the calcification and mass data.

```{code-cell}
:tags: [hide-input]

cases.get_calc_data().sample(5)
cases.get_mass_data().sample(5)
```

+++

Our univariate analysis will cover:

- Breast Density
- Left or Right Breast
- Image View
- Abnormality Id
- Abnormality Type
- Subtlety
- BI-RADS Assessment
- Calcification Type
- Calcification Distribution
- Mass Shape
- Mass Margins
- Pathology
- Cancer (Target)

