---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
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

1. Clean up the inconsistency in the variable names,
2. Combine the training and test sets for masses and calcifications into a single master case file. Provide views for morphology or abnormality type-specific analysis.
3. Add `mmg_id`, a identifier for each mammogram comprised of "<abnormality_type>-<fileset>_<patient_id>_<left_or_right_breast>_<image_view>".
4. Add a Boolean ‘cancer’ target variable that is True if the case is Malignant, False otherwise.

```{code-cell}
:tags: [hide-cell]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))
data_prep = os.path.join("bcd","data_prep", "case.py")
%load $data_prep
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
