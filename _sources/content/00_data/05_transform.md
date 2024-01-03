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

# Data Transformation

Our final data preparation task before exploratory data analysis is to prepare a dataset for multivariate analysis.   For multivariate modeling, we will be one-hot encoding the morphological features and normalizing numeric data to values in [0,1]. 

The multivariate analysis will include 12 independent variables: breast_density, laterality, image_view, abnormality_id, abnormality_type,  calc_type, calc_distribution, subtlety, mass_shape, mass_margins, mean_pixel_value, and std_pixel_value. The binary dependent target variable will be cancer. Variables not included in the analysis are pathology and assessment, since both of these variables are essentially proxies for the target.

```{code-cell} ipython3
:tags: [hide-cell]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))

import pandas as pd
import numpy as np

from bcd.data_prep.transform import CBISTransformer
pd.options.display.max_columns = 99
```

```{code-cell} ipython3
FP_CBIS = "data/meta/3_clean/cbis.csv"
FP_CBIS_MODELING_DATA = "data/meta/3_clean/cbis_model_data.csv"
```

```{code-cell} ipython3
:tags: [hide-cell]

# %load -r 34-178 bcd/data_prep/transform.py
# ------------------------------------------------------------------------------------------------ #
#                                   CBIS TRANSFORMER                                               #
# ------------------------------------------------------------------------------------------------ #
MODEL_VARS = [
    "breast_density",
    "laterality",
    "image_view",
    "abnormality_id",
    "abnormality_type",
    "calc_type",
    "calc_distribution",
    "subtlety",
    "mass_shape",
    "mass_margins",
    "cancer",
    "mean_pixel_value",
    "std_pixel_value",
]
CALC_TYPES = [
    "AMORPHOUS",
    "COARSE",
    "DYSTROPHIC",
    "EGGSHELL",
    "FINE_LINEAR_BRANCHING",
    "LARGE_RODLIKE",
    "LUCENT_CENTERED",
    "MILK_OF_CALCIUM",
    "PLEOMORPHIC",
    "PUNCTATE",
    "ROUND_AND_REGULAR",
    "SKIN",
    "VASCULAR",
]
CALC_DISTRIBUTIONS = [
    "CLUSTERED",
    "LINEAR",
    "REGIONAL",
    "DIFFUSELY_SCATTERED",
    "SEGMENTAL",
]
MASS_SHAPES = [
    "IRREGULAR",
    "ARCHITECTURAL_DISTORTION",
    "OVAL",
    "LYMPH_NODE",
    "LOBULATED",
    "FOCAL_ASYMMETRIC_DENSITY",
    "ROUND",
    "ASYMMETRIC_BREAST_TISSUE",
]
MASS_MARGINS = [
    "SPICULATED",
    "ILL_DEFINED",
    "CIRCUMSCRIBED",
    "OBSCURED",
    "MICROLOBULATED",
]

ENC_VARS = {
    "abnormality_type": {"prefix": "AT", "values": ["calcification", "mass"]},
    "laterality": {"prefix": "LR", "values": ["LEFT", "RIGHT"]},
    "image_view": {"prefix": "IV", "values": ["CC", "MLO"]},
    "calc_type": {"prefix": "CT", "values": CALC_TYPES},
    "calc_distribution": {"prefix": "CD", "values": CALC_DISTRIBUTIONS},
    "mass_shape": {"prefix": "MS", "values": MASS_SHAPES},
    "mass_margins": {"prefix": "MM", "values": MASS_MARGINS},
}


# ------------------------------------------------------------------------------------------------ #
class CBISTransformer:
    """Collapses morphological categories and dummy encodes nominal variables.

    The CBIS-DDSM has 45 calcification types, 9 calcification distributions, 20 mass shapes, and
    19 mass margins, many of which are compound categories, in that two or more categories are
    combined. For instance, calcification type 'ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS' indicates
    three different types: 'ROUND_AND_REGULAR', 'PUNCTATE', and 'AMORPHOUS'. Segregating these
    compound categories into separate categories will drastically reduce the number of categories
    to analyze. More importantly, it aligns our data and the analyses with the common morphological
    taxonomy. So, task one is to extract the unary morphological categories from the
    compound classifications.

    Args:
        source_fp (str): Path to source file
        destination_fp (str): Path to destination file
        force (bool): Whether to force execution if the destination file already exists.
            Default = False.
    """

    def __init__(
        self, source_fp: str, destination_fp: str, force: bool = False
    ) -> None:
        self._source_fp = os.path.abspath(source_fp)
        self._destination_fp = os.path.abspath(destination_fp)
        self._force = force
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._logger.setLevel(logging.DEBUG)

    def transform(self) -> pd.DataFrame:
        """Performs the transformation of the data."""
        if not os.path.exists(self._destination_fp) or self._force:
            df = pd.read_csv(self._source_fp)
            # Excluding identify variables
            df_model_vars = df[MODEL_VARS].copy()
            df_model_vars["cancer"] = np.where(
                df_model_vars["cancer"] == True, 1, 0
            )  # noqa

            # One-hot encode variables
            df_enc = self._encode_dataset(df=df_model_vars)
            # Dropping original string variables.
            df_numeric = df_enc.select_dtypes(exclude=["object"])
            # Normalize all values to [0,1]
            df_norm = self._normalize(df=df_numeric)
            self._save(df=df_norm)
            return df_norm
        else:
            return pd.read_csv(self._destination_fp)

    def _encode_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encodes the dataset"""
        for feature, data in ENC_VARS.items():
            for value in data["values"]:
                df = self._encode_column(
                    df=df, prefix=data["prefix"], col=feature, value=value
                )
        return df

    def _encode_column(self, df, prefix, col, value):
        "One-hot encodes column"
        newcol = prefix + "_" + value
        df[newcol] = np.where(df[col].str.contains(value), 1, 0)
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes all values to [0,1]"""
        for col in df.columns:
            df[col] = df[col] / (df[col].abs().max() - df[col].abs().min())
        return df

    def _save(self, df: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(self._destination_fp), exist_ok=True)
        df.to_csv(self._destination_fp, index=False)
```

```{code-cell} ipython3
x4mr = CBISTransformer(source_fp=FP_CBIS, destination_fp=FP_CBIS_MODELING_DATA, force=False)
df = x4mr.transform()
```

Ok, let's check the results.

```{code-cell} ipython3
df.info()
```

We have 43 variables, 37 of which are one-hot encoded.

```{code-cell} ipython3
df.sample(n=5, random_state=22)
```

All values have been normalized and this dataset is ready for modeling. This completes the data transformation section. On to exploratory data analysis...finally!
