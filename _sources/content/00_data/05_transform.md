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
import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))

import pandas as pd

from bcd.data_prep.transform import CBISTransformer
pd.options.display.max_columns = 99
```

```{code-cell} ipython3
FP_CBIS = "data/meta/3_clean/cbis.csv"
FP_CBIS_MODELING_DATA = "data/meta/3_clean/cbis_model_data.csv"
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
