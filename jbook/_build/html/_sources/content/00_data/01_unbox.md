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

Before conducting data quality or exploratory analyses, we unbox the data to ensure that, well...*'we got what we paid for'*. Our aim here is to:

1. Ensure that record counts are correct,
2. Variable names are consistent across files,
3. Data structure facilitates the next stage of data quality analysis.

```{code-cell} ipython3
:tags: [hide-input]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))
from myst_nb import glue
import pandas as pd
```

```{code-cell} ipython3
fp_calc_train = "data/meta/0_raw/calc_case_description_train_set.csv"
fp_calc_test = "data/meta/0_raw/calc_case_description_test_set.csv"
fp_mass_train = "data/meta/0_raw/mass_case_description_train_set.csv"
fp_mass_test = "data/meta/0_raw/mass_case_description_test_set.csv"
fp_metadata = "data/meta/0_raw/metadata.csv"
```

## Calcification Cases

+++

### Calcification Train Cases

We expect a total of 602 cases, 329 are benign and 273 are malignant.

```{code-cell} ipython3
:tags: [hide-input]

df = pd.read_csv(fp_calc_train)
df.info()
```

We have 1546 observations and 14 columns *(with spaces in the column names, which is somewhat vexing)* in the calcification training set.

```{code-cell} ipython3
:tags: [hide-input, remove-output]

# Total record count
n_records = len(df)

# Number of cases
n_cases =  df['patient_id'].nunique()

# Pathologies and patient counts.
pathologies = df[['patient_id', 'pathology']].drop_duplicates().groupby(by='pathology').count().reset_index()
pathologies.loc['Total'] = pathologies.sum()
pathologies.loc[pathologies.index[-1], 'pathology'] = ""

# Number of patients with multiple abnormalities and diagnoses
case_pathologies = df[['patient_id', 'pathology']].drop_duplicates().groupby(by='patient_id').count()
n_cases_multiple_pathologies = len(case_pathologies.loc[case_pathologies['pathology']==2])

glue('n_records', n_records)
glue('pathologies', pathologies)
glue('n_cases', n_cases)
glue('n_cases_multiple_pathologies', n_cases_multiple_pathologies)
```

There are a total of {glue:}`n_records` records and {glue:}`n_cases` cases in the calcification training set. The following summarizes the pathologies.

```{glue:figure} pathologies
---
align: center
name: pathologies_fig
---
Patient Count by Pathology
```

{numref}`pathologies_fig` shows 14 more cases than expected. Indeed, there are {glue:}`n_cases_multiple_pathologies` patients with two pathologies, bringing our total to 616.

+++

Several observations can be made:

1. It would appear that we have three values for pathology: MALIGNANT, BENIGN, and BENIGN_WITHOUT_CALLBACK. While changing this to BENIGN might simplify things, we lose information that might be useful in understanding missing values, such as #2.
2. About 2% of the cases in the training set have missing calcification types. One case is illustrated above. Might the missing value be associated with the pathology?
3. A quick check of the file paths reveals, unfortunately, that they are not valid. If the other case files contain invalid file paths, we can safely drop these them from the dataset.
4. One slightly vexing

+++

### Calcification Test Cases
