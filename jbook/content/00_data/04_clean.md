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

# Data Cleaning
The prior data quality analysis revealed several data anomalies requiring attention.

1. Cases with invalid values for subtlety and breast density.
2. Missing calcification type, calcification distribution, mass shape, and mass margins data.
3. Categories that have different spelling, but the same meaning.

As such, the data cleaning tasks are detailed in {numref}`data_cleaning_tasks`:

```{table} Data Cleaning Tasks
:name: data_cleaning_tasks
| # | Task                                                                             |
|---|----------------------------------------------------------------------------------|
| 1 | Replace invalid values for breast density with NA for imputation.                |
| 2 | Replace cases with invalid values for subtlety with NA for imputation            |
| 3 | Replace calcification types 'LUCENT_CENTER' with 'LUCENT_CENTERED', and 'PLEOMORPHIC-PLEOMORPHIC', with 'PLEOMORPHIC' |
| 4 | Impute missing values using Multiple Imputation by Chained Equations   (MICE)    |
| 5 | Conduct random inspection of imputations.                                        |
| 6 | Conduct final data quality analysis.                                        |
```

Once the case dataset has been cleaned, the case data will be merged into the DICOM dataset.

```{code-cell} ipython3
import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))

import pandas as pd
import numpy as np

from bcd.data_prep.clean import CBISImputer
from bcd.data.dataset import CBISDataset

pd.options.display.max_rows = 999
```

```{code-cell} ipython3
FP_STAGED = "data/meta/2_staged/cbis.csv"
FP_CLEAN = "data/meta/3_clean/cbis.csv"
```

## Load Data

```{code-cell} ipython3
df = pd.read_csv(FP_STAGED)
df_orig = df.copy()
```

## Breast Density
Replace invalid values for breast density with NA for downstream imputation.

```{code-cell} ipython3
# Set invalid values for breast_density to NA
df['breast_density'] = df['breast_density'].replace(0, np.NAN)
```

## Subtlety
Replace invalid values for subtlety with NA.

```{code-cell} ipython3
# Set case and mass data to NOT APPLICABLE where appropriate.
df['subtlety'] = df['subtlety'].replace(0, np.NAN)
```

## Category Alignment
Align categories that have the same meaning, but differ in spelling only.

```{code-cell} ipython3
df.loc[df['calc_type'] == 'LUCENT_CENTER', 'calc_type'] = 'LUCENT_CENTERED'
df.loc[df['calc_type'] == 'ROUND_AND_REGULAR-LUCENT_CENTER-DYSTROPHIC', 'calc_type'] = 'ROUND_AND_REGULAR-LUCENT_CENTERED-DYSTROPHIC'
df.loc[df['calc_type'] == 'PUNCTATE-LUCENT_CENTER', 'calc_type'] = 'PUNCTATE-LUCENT_CENTERED'
df.loc[df['calc_type'] == 'VASCULAR-COARSE-LUCENT_CENTER-ROUND_AND_REGULAR-PUNCTATE', 'calc_type'] = 'VASCULAR-COARSE-LUCENT_CENTERED-ROUND_AND_REGULAR-PUNCTATE'
df.loc[df['calc_type'] == 'ROUND_AND_REGULAR-LUCENT_CENTER', 'calc_type'] = 'ROUND_AND_REGULAR-LUCENT_CENTERED'
df.loc[df['calc_type'] == 'LUCENT_CENTER-PUNCTATE', 'calc_type'] = 'LUCENT_CENTERED-PUNCTATE'
df.loc[df['calc_type'] == 'COARSE-ROUND_AND_REGULAR-LUCENT_CENTER', 'calc_type'] = 'COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED'
df.loc[df['calc_type'] == 'ROUND_AND_REGULAR-LUCENT_CENTER-PUNCTATE', 'calc_type'] = 'ROUND_AND_REGULAR-LUCENT_CENTERED-PUNCTATE'
df.loc[df['calc_type'] == 'COARSE-LUCENT_CENTER', 'calc_type'] = 'COARSE-LUCENT_CENTERED'
df.loc[df['calc_type'] == 'PLEOMORPHIC-PLEOMORPHIC', 'calc_type'] = 'PLEOMORPHIC'
```

## Impute Missing Values
Multiple Imputation by Chained Equations (MICE) is a robust, informative method of estimating missing values in datasets. The procedure imputes missing data through an iterative series of predictive models which estimate the value of missing data using the other variables in the dataset. For this, we'll use our CaseImputer which wraps scikit-learn's IterativeImputer implementation of MICE.

First, let's capture the missing values as we will inspect them after imputation.

```{code-cell} ipython3
# Grab rows with missing data
null_mask = df.isnull().any(axis=1)
df_missing = df[null_mask]
msg = f"There are {df_missing.shape[0]} rows (approximately {round(df_missing.shape[0] / df_orig.shape[0] * 100,1)}% of the dataset) with missing data in the dataset."
print(msg)
```

```{code-cell} ipython3
imp = CBISImputer(random_state=5)
imp.fit(df=df)
df_clean = imp.transform(df=df)
```

With that, let's save the data.

```{code-cell} ipython3
os.makedirs(os.path.dirname(FP_CLEAN), exist_ok=True)
df_clean.to_csv(FP_CLEAN, index=False)
```

## Random Sample Inspection

+++

Let's take a look at a random sampling of the missing data and compare.

```{code-cell} ipython3
sample_cases = df_missing['mmg_id'].sample(5)
df_missing.loc[df_missing['mmg_id'].isin(sample_cases)]
df_clean.loc[df_clean['mmg_id'].isin(sample_cases)]
```

## Data Quality Analysis 2.0
Ok, let's have another go at the data quality analysis.

+++

### Completeness Analysis

```{code-cell} ipython3
ds = CBISDataset(filepath=FP_CLEAN)
dqcc = ds.dqa.analyze_completeness()
print(dqcc.summary)
```

We're complete.

+++

### Uniqueness Analysis

```{code-cell} ipython3
dqcu = ds.dqa.analyze_uniqueness()
print(dqcu.summary)
```

We're unique.

+++

### Validity Analysis

```{code-cell} ipython3
dqcv = ds.dqa.analyze_validity()
print(dqcv.summary)
```

We're valid. That concludes this data cleaning section.
