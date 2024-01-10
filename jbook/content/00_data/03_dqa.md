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

# Data Quality Analysis (DQA)
In the previous section, we began the process of transforming the raw CBIS-DDSM case and DICOM data into a structure and format more suitable for analysis. The purpose of this effort is to identify potential data quality issues, errors, and anomalies requiring further data preprocessing, prior to the analysis stage.

## Approach
Our approach will touch on three dimensions of data quality.

| # | Dimension | Definition | Metric | Metric Definition |
|---|----------------|-------------------------------------|----------------|-------------------------------------|
| 1 | Completeness | Are the data complete? | Completeness Ratio | Ratio of non-null data values / rows over the total number of data values / rows |
| 2 | Uniqueness | Are their duplicate records | Uniqueness Ratio | Ratio of unique data values / rows over total number of data values / rows |
| 3 | Validity   | Are the values consistent with defined rules? | Validity Ratio | Ratio of valid cells / rows over total number of data cells / rows |

Note, accuracy and relevance, two additional dimensions of data quality, are missing from this analysis. An accuracy evaluation requires an external source-of-truth against which, the CBIS-DDSM data are compared. As we lack such a data source, we have no basis for evaluating the accuracy of the CBIS-DDSM data collection. With respect to relevance, duplicate and irrelevant data were removed from the data collection during the previous data preparation exercise.

```{code-cell} ipython3
:tags: [hide-cell]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))

import pandas as pd
from IPython.display import HTML, display_html
import numpy as np

from bcd.dqa.cbis import CBISDQA
from bcd.data.dataset import CBISDataset

pd.set_option('display.max_colwidth', 200)
```

## Preliminaries
As a first step, we'll create the CBIS-DDSM Dataset object and install the DQA module.

```{code-cell} ipython3
filepath = "data/meta/2_staged/cbis.csv"
cbis = CBISDataset(filepath=filepath)
```

## Completeness Analysis

```{code-cell} ipython3
dqc = cbis.dqa.analyze_completeness()
print(dqc.summary, end=" ")
```

Record and data value completeness are 0.85 and 0.99 respectively. Let's take a closer look.

```{code-cell} ipython3
dqc.detail
```

Missingness for calc type, mass_shape, and mass_margins is 1%, < 1%, and 2% respectively. Strictly speaking, we could drop those observations without a considerable loss of data. On the other hand, calc distribution missingness deserves a closer look. Let's examine missingness vis-a-vis the pathology target.

```{code-cell} ipython3
cbis.dqa.get_incomplete_data(subset='calc_distribution')['pathology'].value_counts().to_frame()
```

This is interesting. Over 98% of the records with missing calcification distribution have a pathology of BENIGN_WITHOUT_CALLBACK. This suggests that the calcification distribution data are Missing at Random (MAR). That is, the probability of encountering missing calcification distribution may be systematically related to a benign pathology. As the missingness seems to be related to *observed* data, the multiple imputation by chained equations (MICE) {cite}`schaferMultipleImputationPrimer1999` is a principled method for dealing with such missing data while mitigating data loss and bias in the dataset.

Let's move on to the uniqueness analysis.

+++

## Uniqueness Analysis

```{code-cell} ipython3
dqu = cbis.dqa.analyze_uniqueness()
print(dqu.summary)
```

No duplicate records exist in the data.

+++

### Validity Analysis
Here, we assess the degree to which the datasets contain valid values.

```{code-cell} ipython3
dqv = cbis.dqa.analyze_validity()
print(dqv.summary)
```

Record and data value validity are 0.84 and 0.99, respectively. Note, missing data will have an impact on record validity. Still, let's take a closer look at the invalid data.

```{code-cell} ipython3
dqv.detail
```

The invalidity of calc distribution and mass_margins is explained by the missing data. Approximately 6% of the observations have invalid calcification types. Let's take a look.

```{code-cell} ipython3
cbis.dqa.get_invalid_data(subset='calc_type')['calc_type'].unique()
```

A common thread among these invalid values is the type, 'LUCENT_CENTER', which should be 'LUCENT_CENTERED'. 'PLEOMORPHIC-PLEOMORPHIC', should be 'PLEOMORPHIC' and there are some NaN values extant. We'll make those changes in the next section. Now, the invalid breast density records.

```{code-cell} ipython3
cbis.dqa.get_invalid_data(subset='breast_density')
```

Both observations with breast density values of zero, are for the same patient. Let's see what densities we have for similar malignant calcification cases.

```{code-cell} ipython3
df = cbis.dqa.get_complete_data()
df_similar_type = df.loc[(df['calc_type'] == 'PLEOMORPHIC') & (df['cancer'] == True)]
df_similar_dist = df.loc[(df['calc_distribution'] == 'DIFFUSELY_SCATTERED') & (df['cancer'] == True)]
```

The breast density counts below are for PLEOMORPHIC calcification cases.

```{code-cell} ipython3
df_similar_type['breast_density'].value_counts().to_frame()
```

Statistically, breast densities of 2,3 and 4 are equally probable for malignant PLEOMORPHIC cases. Let's see if calcification distribution is more discriminative.

```{code-cell} ipython3
df_similar_dist['breast_density'].value_counts().to_frame()
```

We only have five DIFFUSELY_SCATTERED calcification cases, two of which are our invalid cases. Two cases have breast densities of 2, and another with a value of 3. Not a statistically significant difference in breast densities for DIFFUSELY_SCATTERED calcification cases. Though these invalid observations are relative, they represent 40% of the DIFFUSELY_SCATTERED calcification cases. We'll attempt to impute these values rather than delete them.

+++

#### Invalid Mass Case Analysis
The invalid mass cases have subtlety values of zero. Let's take a look at the data.

```{code-cell} ipython3
cbis.dqa.get_invalid_data(subset='subtlety')
```

Similarly, these cases are for the same patient. Notably, these cases convey little information: missing mass shape and margins data. In this case deletion would be the best option.

+++

### Summary
A brief data quality analysis of the CBIS-DDSM case and DICOM data examined completeness, uniqueness, and validity. Our data cleaning tasks are as follows:

1. The following observations have zero for breast density, values that will be marked as missing and imputed.
   1. P_01743_RIGHT_calcification_CC_1
   2. P_01743_RIGHT_calcification_MLO_1
2. The following cases have zero subtlety. These values will be marked as missing and imputed.
   1. P_00710_RIGHT_mass_MLO_1
   2. P_00710_RIGHT_mass_CC_1
3. For calcification type, we'll perform the following replacements:
   1. 'LUCENT_CENTERED' for 'LUCENT_CENTER'
   2. 'PLEOMORHIC' for 'PLEOMORHIC-PLEOMORHIC'
4. We will use Multivariate Imputation by Chained Equations (MICE) to predict missing values for the following variables:
   1. calc distribution
   2. mass_margins
   3. mass shape
   4. calc type
   5. breast_density
   6. subtlety
