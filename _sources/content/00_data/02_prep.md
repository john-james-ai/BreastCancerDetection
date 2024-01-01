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

# Data Preparation Part 1: Structural Upgrade

In the prior section, we identified a few structural concerns worth addressing before any quality or exploratory analysis efforts take place. Here, we apply a few upgrades that should streamline the data quality analysis in terms of structural consistency. 

The upgrades will involve the creation of two new datasets:

```{table} Pre-Analysis Case and DICOM Datasets
| # | Dataset                      | Description                                                                         |
|---|------------------------------|-------------------------------------------------------------------------------------|
| 1 | Case Dataset                 | Master case dataset with training and test data for calcification and mass cases. |
| 2 | DICOM Image Metadata Dataset | Inventory of every image with reference back to the individual case.                |
```
First up? The case dataset.

## Case Dataset Upgrades
1.	Clean up the inconsistency in the variable names,
2.	Combine the training and test sets for masses and calcifications into a single master case file. Provide views for morphology or abnormality type-specific analysis.
3.	Add `mmg_id`, a identifier for each mammogram comprised of <abnormality_type>-<fileset>_<patient_id>_<left_or_right_breast>_<image_view>. 
4.	Add a Boolean ‘cancer’ target variable that is True if the case is Malignant, False otherwise.

```{code-cell} ipython3
:tags: [remove-cell]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))
from typing import Union

import numpy as np
import pandas as pd

from bcd.data_prep.base import DataPrep
```

```{code-cell} ipython3
:tags: [hide-cell]

%load -r 38-163 bcd/data_prep/case.py
# ------------------------------------------------------------------------------------------------ #
class CasePrep(DataPrep):
    """Performs Case metadata preparation."""

    def prep(
        self,
        calc_train_fp: str,
        calc_test_fp: str,
        mass_train_fp: str,
        mass_test_fp: str,
        case_fp: str,
        force: bool = False,
        result: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """Combines training and test cases into a single csv case file.

        Args:
            calc_train_fp, calc_test_fp, mass_train_fp, mass_test_fp (str): The file paths to the
                calcification and mass training and test sets.
            case_fp (str): Path to output calcification and mass datasets.

            force (bool): Whether to force execution if output already exists. Default is False.
            result (bool): Whether the result should be returned. Default is False.

        Returns
            If result is True, the case dataframe is returned.
        """
        case_fp = os.path.abspath(case_fp)

        os.makedirs(os.path.dirname(case_fp), exist_ok=True)

        if force or not os.path.exists(case_fp):
            # Merge all case data into a single DataFrame
            df_cases = self._merge_cases(
                calc_train_fp=calc_train_fp,
                calc_test_fp=calc_test_fp,
                mass_train_fp=mass_train_fp,
                mass_test_fp=mass_test_fp,
            )

            df_cases = self._assign_mmg_id(df=df_cases)

            # Transform 'BENIGN WITHOUT CALLBACK' to 'BENIGN'
            df_cases["cancer"] = np.where(
                df_cases["pathology"] == "MALIGNANT", True, False
            )

            # Save datasets
            df_cases.to_csv(case_fp, index=False)

        if result:
            return df_cases

    def _merge_cases(
        self,
        calc_train_fp: str,
        calc_test_fp: str,
        mass_train_fp: str,
        mass_test_fp: str,
    ) -> pd.DataFrame:
        """Combines mass and calcification train and test files into a single file."""
        # Extracts absolute paths, a pre-emptive measure in case
        # jupyter book can't access the path
        calc_train_fp = os.path.abspath(calc_train_fp)
        calc_test_fp = os.path.abspath(calc_test_fp)
        mass_train_fp = os.path.abspath(mass_train_fp)
        mass_test_fp = os.path.abspath(mass_test_fp)

        df_calc_train = pd.read_csv(calc_train_fp)
        df_calc_test = pd.read_csv(calc_test_fp)
        df_mass_train = pd.read_csv(mass_train_fp)
        df_mass_test = pd.read_csv(mass_test_fp)

        # Add the filesets so that we can distinguish training
        # and test data
        df_calc_train["fileset"] = "training"
        df_calc_test["fileset"] = "test"
        df_mass_train["fileset"] = "training"
        df_mass_test["fileset"] = "test"

        # Replace spaces in column names with underscores.
        df_calc_train = self._format_column_names(df=df_calc_train)
        df_calc_test = self._format_column_names(df=df_calc_test)
        df_mass_train = self._format_column_names(df=df_mass_train)
        df_mass_test = self._format_column_names(df=df_mass_test)

        # Concatenate the files, ensuring that the morphologies are appropriately
        # set to NOT APPLICABLE where needed.
        df = pd.concat(
            [df_calc_train, df_calc_test, df_mass_train, df_mass_test], axis=0
        )
        df.loc[df["abnormality_type"] == "mass", "calc_type"] = "NOT APPLICABLE"
        df.loc[df["abnormality_type"] == "mass", "calc_distribution"] = "NOT APPLICABLE"
        df.loc[
            df["abnormality_type"] == "calcification", "mass_shape"
        ] = "NOT APPLICABLE"
        df.loc[
            df["abnormality_type"] == "calcification", "mass_margins"
        ] = "NOT APPLICABLE"
        return df

    def _assign_mmg_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign a mammogram id to each observation."""
        df["mmg_id"] = (
            df["abnormality_type"].apply(lambda x: x[0:4].capitalize())
            + "-"
            + df["fileset"].apply(lambda x: x.capitalize())
            + "_"
            + df["patient_id"]
            + "_"
            + df["left_or_right_breast"].upper()
            + "_"
            + df["image_view"]
        )

        return df

    def _format_column_names(self, df: pd.DataFrame) -> str:
        """Replaces spaces in column names with underscores."""

        def replace_columns(colname: str) -> str:
            return colname.replace(" ", "_")

        df.columns = df.columns.to_series().apply(replace_columns)
        return df
```

## DICOM Image Metadata

Here, our objective is to extract and augment the DICOM Image metadata from the DICOM files, producing the following dataset. 

```{table} DICOM Image Metadata
:name: dicom_image_metadata
| #  | Variable           | Source    | Description                                                                                                                                                                               |
|----|--------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | subject_id         | extracted | Identifier consisting of   <abnormality_type>-<fileset>_<patient_id>_<left_or_right_breast>_<image_view>[_abnormality_id].   Abnormality id is optional, used for ROI and cropped images. |
| 2  | series_description | extracted | Describes whether the series if full mammogram, ROI, or cropped images.                                                                                                                   |
| 3  | rows               | extracted | Number of rows in pixel array                                                                                                                                                             |
| 4  | columns            | extracted | Number of columns in pixel array                                                                                                                                                          |
| 5  | bits               | extracted | Bit resolution                                                                                                                                                                            |
| 6  | image_id           | generated | Image identifier composed of the subject_id and an optional image number for ROI and cropped images.                                                                                    |
| 7  | file_path           | generated | Path to the file on disk                                                                                                                                                                  |
| 8  | file_size          | generated | File size                                                                                                                                                                                 |
| 9  | size               | generated | Number of elements in pixel array                                                                                                                                                         |
| 10 | min_pixel_value    | generated | Minimum pixel value                                                                                                                                                                       |
| 11 | max_pixel_value    | generated | Maximum pixel value                                                                                                                                                                       |
| 12 | mean_pixel_value   | generated | Average pixel value                                                                                                                                                                       |
| 13 | std_pixel_value    | generated | Standard deviation of pixel values.                                                                                                                                                       |
| 14 | mmg_id             | generated | Foreign mammogram identifier linking to cases.                                                                                                                                            |
``
As {numref}`dicom_image_metadata` shows, subject, series, and basic pixel data are extracted from the DICOM datasets.  We generate the image identifier, the mammogram identifier, the file path, the file size, the array size, as well as descriptive statistics on the image pixels.
