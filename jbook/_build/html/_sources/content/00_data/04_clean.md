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

```{code-cell} ipython3
:tags: [hide-cell]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from studioai.preprocessing.encode import RankFrequencyEncoder

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
Multiple Imputation by Chained Equations (MICE) is a robust, informative method of estimating missing values in datasets. The procedure imputes missing data through an iterative series of predictive models which estimate the value of missing data using the other variables in the dataset. For this, we'll use our CBISImputer which wraps scikit-learn's IterativeImputer implementation of MICE.

First, let's capture the missing values as we will inspect them after imputation.

```{code-cell} ipython3
# Grab rows with missing data
null_mask = df.isnull().any(axis=1)
df_missing = df[null_mask]
msg = f"There are {df_missing.shape[0]} rows (approximately {round(df_missing.shape[0] / df_orig.shape[0] * 100,1)}% of the dataset) with missing data in the dataset."
print(msg)
```

```{code-cell} ipython3
:tags: [hide-cell]

# %load -r 37-119 bcd/data_prep/clean.py
class CBISImputer:
    """Imputes the missing values in the case dataset using Multiple Imputation by Chained Equations

    Args:
        max_iter (int): Maximum number of imputation rounds to perform before returning
        the imputations computed during the final round.
        initial_strategy (str): Which strategy to use to initialize the missing values.
            Valid values include: {'mean', 'median', 'most_frequent', 'constant'},
            default=most_frequent'
        random_state (int): The seed of the pseudo random number generator to use.

    """

    def __init__(
        self,
        max_iter: int = 50,
        initial_strategy: str = "most_frequent",
        random_state: int = None,
    ) -> None:
        self._max_iter = max_iter
        self._initial_strategy = initial_strategy
        self._random_state = random_state
        self._encoded_values = {}
        self._dtypes = None
        self._enc = None
        self._imp = None

    def fit(self, df: pd.DataFrame) -> CBISImputer:
        """Fits the data to the imputer

        Instantiates the encoder, encodes the data and creates a
        map of columns to valid encoded values. We capture these
        values in order to map imputed values
        back to valid values before we inverse transform.

        Args:
            df (pd.DataFrame): Imputed DataFrame
        """
        self._dtypes = df.dtypes.astype(str).replace("0", "object").to_dict()
        self._enc = RankFrequencyEncoder()
        df_enc = self._enc.fit_transform(df=df)
        self._extract_encoded_values(df=df_enc)

        # Get complete cases for imputer training (fit)
        df_enc_complete = df_enc.dropna(axis=0)

        self._imp = IterativeImputer(
            max_iter=self._max_iter,
            initial_strategy=self._initial_strategy,
            random_state=self._random_state,
        )
        self._imp.fit(X=df_enc_complete.values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs the imputation and returns the imputed DataFrame

        Args:
            df (pd.DataFrame): Imputed DataFrame

        """
        df_enc = self._enc.transform(df=df)
        imp = self._imp.transform(X=df_enc.values)
        df_imp = pd.DataFrame(data=imp, columns=df.columns)
        df_imp = self._map_imputed_values(df=df_imp)
        df_imp = self._enc.inverse_transform(df=df_imp)
        df_imp = df_imp.astype(self._dtypes)
        return df_imp

    def _extract_encoded_values(self, df: pd.DataFrame) -> None:
        """Creates a dictionary of valid values by column."""
        for col in df.columns:
            valid = df[col].dropna()
            self._encoded_values[col] = valid.unique()

    def _map_imputed_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps values to valid values (used after imputation)"""
        for col in df.columns:
            values = np.array(sorted(self._encoded_values[col]))
            df[col] = df[col].apply(lambda x: values[np.argmin(np.abs(x - values))])
        return df
```

```{code-cell} ipython3
imp = CBISImputer(random_state=5)
_ = imp.fit(df=df)
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
