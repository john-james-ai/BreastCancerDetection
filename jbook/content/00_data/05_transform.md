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

Our final data preparation task before exploratory data analysis is to prepare a dataset for multivariate analysis.   For multivariate modeling, we will be one-hot encoding the morphological features and normalizing numeric data to range [0,1]. 

The multivariate analysis will include 12 independent variables: breast_density, laterality, image_view, abnormality_id, abnormality_type, assessment,  calc_type, calc_distribution, subtlety, mass_shape, mass_margins, mean_pixel_value, and std_pixel_value. The binary dependent target variable will be cancer. Pathology will not be included in the multivariate analysis; however, BI-RADS assessment is expected to be a major influence on the target. It would be notable to the degree it is not. 

```{code-cell} ipython3
:tags: [remove-cell]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))
```

```{code-cell} ipython3
:tags: [hide-cell]



import pandas as pd
import numpy as np

from bcd.data_prep.transform import CBISTransformer
pd.options.display.max_columns = 99
```

```{code-cell} ipython3
FP_CLEAN = "data/meta/3_clean/cbis.csv"
FP_COOKED = "data/meta/4_cooked/cbis.csv"
```

```{code-cell} ipython3
x4mr = CBISTransformer(source_fp=FP_CLEAN, destination_fp=FP_COOKED, force=True)
df = x4mr.transform()
```

Ok, let's check the results.

```{code-cell} ipython3
df.info()
```

We have 64 variables, 37 of which are one-hot encoded.

```{code-cell} ipython3
df.sample(n=5, random_state=22)
```

All values have been normalized and this completes the data transformation section. On to exploratory data analysis...finally!
