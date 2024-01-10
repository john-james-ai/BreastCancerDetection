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
# Data Preparation

In the prior section, we identified a few structural concerns worth addressing before any quality or exploratory analysis analyses take place. Here, we extract the relevant task-specific information from the CBIS-DDSM case and dicom datasets and integrate the data into a single, combined full mammogram dataset.

Our process will take three steps:

1. Combine the calcification mass training and test sets into a single full mammogram dataset,
2. Add DICOM image file paths to the *series* metadata,
3. Extract the *DICOM* image metadata using the file paths above, and merge them with the case data from #1.

The full dataset will have a few upgrades that will facilitate the analysis, detection, and classification tasks:

1. A mammogram ID, consisting of abnormality type, fileset (train/test), patient_id, breast laterality, and view will uniquely identify each full mammogram image.
2. A Boolean target variable, 'cancer', will be added combining BENIGN and BENIGN_WITHOUT_CALLBACK into a single Boolean value.
3. Pixel statistics such as the minimum, maximum, mean, and standard deviation, will be added to the dataset.

## Case Dataset Integration

The following code cells will integrate all case data into a single file. This doesn't rule out separation of mass and calcification cases as may be needed downstream, but it reduces a certain amount of redundancy and allows us to see both the forest and the trees.

```{code-cell}
:tags: [remove-cell]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))
```

```{code-cell}
:tags: [hide-cell]

from typing import Union
from glob import glob

import dask
import numpy as np
import pandas as pd
import pydicom

from bcd.dal.file import IOService
from bcd.utils.file import getsize
from bcd.utils.profile import profiler
from bcd.data_prep.base import DataPrep
from bcd.data_prep.prep import CasePrep
from bcd.data_prep.prep import SeriesPrep
from bcd.data_prep.prep import CBISPrep
```

```{code-cell}

calc_test = "data/meta/0_raw/calc_case_description_test_set.csv"
calc_train = "data/meta/0_raw/calc_case_description_train_set.csv"
mass_test = "data/meta/0_raw/mass_case_description_test_set.csv"
mass_train = "data/meta/0_raw/mass_case_description_train_set.csv"

case_fp = "data/meta/1_interim/cases.csv"

cp = CasePrep(calc_train_fp=calc_train, calc_test_fp=calc_test, mass_train_fp=mass_train, mass_test_fp=mass_test, case_fp=case_fp, force=False)
cases = cp.prep()
cases.info()
cases.sample(n=5, random_state=55)
```

The dataset above has both mass and calcification training and test data, as well as a mammogram id, 'mmmg_id', and a Boolean target 'cancer'.

+++

## Series Metadata

Next, we add filepaths to the series metadata.

```{code-cell}
fpi = "data/meta/0_raw/metadata.csv"
fpo = "data/meta/3_clean/series.csv"
sp = SeriesPrep(filepath=fpi, series_filepath=fpo, force=False)
series = sp.prep()
series.info()
series.sample(n=5, random_state=55)
```

Full filepaths have been added for all 10,239 images in the CBIS-DDSM.

+++

## DICOM Image Metadata

Finally, we extract the DICOM data described in {numref}`dicom_image_metadata` and merge that with the case data.

```{table}
:name: dicom_image_metadata

| # | Name                       | Description                                                                              |
|---|----------------------------|------------------------------------------------------------------------------------------|
| 1 | bit_depth                  | Number of bits used to define each pixel                                                 |
| 2 | rows                       | Number of pixel rows in the image                                                        |
| 3 | cols                       | Number of pixel columns in the image                                                     |
| 4 | aspect_ratio               | Ratio of width to height in image                                                        |
| 5 | size                       | Product of width and height in image                                                     |
| 6 | min_pixel_value            | Minimum pixel value                                                                      |
| 7 | max_pixel_value            | Maximum pixel value                                                                      |
| 8 | mean_pixel_value           | Average pixel value                                                                      |
| 9 | std_pixel_value            | Standard deviation of pixel values                                                       |

```

```{code-cell}
cases = "data/meta/1_interim/cases.csv"
series = "data/meta/3_clean/series.csv"
cbis = "data/meta/2_staged/cbis.csv"
cp = CBISPrep(case_filepath=cases, series_filepath=series, cbis_filepath=cbis, force=False)
cbis = cp.prep()
cbis.info()
cbis.sample(n=5, random_state=55)
```

We have all case information along with the DICOM image metadata in a single dataset.
