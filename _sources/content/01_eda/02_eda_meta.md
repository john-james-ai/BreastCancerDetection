---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"tags": ["hide-input"]}

# Exploratory Data Analysis (EDA) 1.0 - Metadata

In this section, we conduct an exploratory data analysis of the CBIS-DDSM Case and DICOM datasets prepared in prior sections. The purpose of this EDA is three-fold:

1. Discover the relationships among the features
2. Explore the nature of the relationships between the features and the diagnostic target.
3. Identify features that have the greatest influence classification accuracy.

## The Dataset

In this section, we will be analyzing the Case Dataset: calcification and mass datasets containing patient, abnormality, BI-RADS assessment, image image_view, breast density, subtlety, and pathology (diagnosis) information. The dataset dictionary is outlined in {numref}`eda1_case_dataset`.

```{table} Case Dataset Dictionary
:name: eda1_case_dataset

| #  | Variable             | Type        | Description                                                                                                                  |
|----|----------------------|-------------|------------------------------------------------------------------------------------------------------------------------------|
| 1  | patient_id           | Nominal     | Unique identifier for each patient.                                                                                          |
| 2  | breast_density       | Discrete    | BI-RADS overall assessment of the volume of attenuating tissues in the breast.                                             |
| 3  | left_or_right_breast | Nominal     | Which breast was imaged.                                                                                                     |
| 4  | image_view           | Dichotomous | Either cranialcaudal or mediolateral oblique image_view.                                                                            |
| 5  | abnormality_id       | Discrete    | Number of abnormalities for the patient.                                                                                       |
| 6  | abnormality_type     | Dichotomous | BI-RADS category of the abnormality.                                                                                         |
| 7  | calc_type            | Nominal     | Characterization of the type of calcification (where applicable)                                                             |
| 8  | calc_distribution    | Nominal     | The arrangement of the calcifications inside the breast and, relative to the probability of malignancy. (where applicable) |
| 9  | mass_shape           | Nominal     | Shape of the mass                                                                                                            |
| 10 | mass_margins         | Nominal     | Feature that separates the mass from the adjacent breast parenchyma.                                                         |
| 11 | assessment           | Discrete    | Overall BI-RADS assessment of the mammography                                                                                |
| 12 | pathology            | Nominal     | Determination of the malignancy of the case.                                                                                 |
| 13 | subtlety             | Discrete    | Degree of diagnostic difficulty                                                                                              |
| 14 | fileset              | Nominal     | Indicates training or test set.                                                                                              |
| 15 | case_id              | Nominal     | Unique identifier for the case.                                                                                              |
| 16 | cancer               | Dichotomous | Indicates whether the cancer is diagnosed.                                                                                   |
```

## Guiding Questions

```{tip}
“Far better an approximate answer to the right question, which is often vague, than an exact answer to the wrong question, which can always be made precise.” — John Tukey
```

Here, we'll put forward a set of questions to motivate and guide the discovery process.

1. What are the relationships between calcification and mass morphological features and malignancy?
2. To what degree does breast density relate to abnormality types and malignancy?
3. Are certain abnormalities more or less subtle?
4. What are the relative features of importance concerning screening?

## Exploratory Data Analysis Plan

The EDA will be conducted in three primary stages:

1. **Univariate**: Examination of the variables independently
2. **Bivariate**: Evaluate the relations among the features and between the features and the target.
3. **Multivariate**: Discover feature importance w.r.t. screening and diagnosis.

+++ {"tags": ["remove-output", "hide-input"]}

```{code-cell}
:tags: [remove-cell, hide-input]

import sys
import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../../..")))
import warnings
warnings.filterwarnings("ignore")
```

```{code-cell}
:tags: [hide-input]

import pandas as pd
from scipy import stats
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from bcd.explore.meta.case import CaseExplorer
from bcd.explore.meta.multivariate.pipeline import PipelineBuilder
from bcd.explore.meta.multivariate.selection import ModelSelector

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
sns.set_style('whitegrid')
sns.set_palette("Blues_r")
```

```{code-cell}
:tags: [remove-cell, hide-input]

CASE_FP = "data/meta/3_cooked/cases.csv"
```

```{code-cell}
:tags: [hide-input]

cases = CaseExplorer(filepath=CASE_FP)
calc = CaseExplorer(df=cases.get_calc_data())
mass = CaseExplorer(df=cases.get_mass_data())
```

+++ {"tags": ["hide-input"]}

## Univariate Analysis

Let's get an overall sense of the data.

```{code-cell}
:tags: [remove-output]

summary = cases.summary()
glue("eda1_summary", summary)
```

```{glue:figure} eda1_summary
---
align: center
name: eda1_summary_df
---
CBIS-DDSM Dataset Summary
```

```{code-cell}
:tags: [hide-input]

st = summary.T
pct_calc = round(st['Calcification Cases'] / st['Cases'] * 100,2).values[0]
pct_mass = round(st['Mass Cases'] / st['Cases'] * 100,2).values[0]
pct_calc_mal = round(st['Calcification Cases - Malignant'] / st['Calcification Cases'] * 100,2).values[0]
pct_calc_bn = round(st['Calcification Cases - Benign'] / st['Calcification Cases'] * 100,2).values[0]
pct_mass_mal = round(st['Mass Cases - Malignant'] / st['Mass Cases'] * 100,2).values[0]
pct_mass_bn = round(st['Mass Cases - Benign'] / st['Mass Cases'] * 100,2).values[0]
cases_per_patient = round(st['Cases'] / st['Patients'],2).values[0]

glue("pct_calc", pct_calc)
glue("pct_mass", pct_mass)
glue("pct_calc_mal", pct_calc_mal)
glue("pct_calc_bn", pct_calc_bn)
glue("pct_mass_mal", pct_mass_mal)
glue("pct_mass_bn", pct_mass_bn)
glue("cases_per_patient", cases_per_patient)
```

From {numref}`eda1_summary_df`, several observations can be made:

1. We have 3566 cases, {glue:}`pct_calc`% are calcification cases and {glue:}`pct_mass`% are mass cases.
2. Of the calcification cases, {glue:}`pct_calc_bn`% are benign and {glue:}`pct_calc_mal`% are malignant.
3. Of the mass cases, {glue:}`pct_mass_bn`% are benign and {glue:}`pct_mass_mal`% are malignant.
4. On average, we have approximately {glue:}`cases_per_patient` cases per patient.

Case, as defined in {cite}`leeCuratedMammographyData2017`, indicates a particular abnormality, seen on the craniocaudal (CC) and/or mediolateral oblique (MLO) views.

+++

Let's take a look at the calcification and mass data.

```{code-cell}
:tags: [hide-input]

cases.get_calc_data().sample(5)
cases.get_mass_data().sample(5)
```

+++

Our univariate analysis will cover:

- Breast Density
- Left or Right Breast
- Image View
- Abnormality Id
- Abnormality Type
- Subtlety
- BI-RADS Assessment
- Calcification Type
- Calcification Distribution
- Mass Shape
- Mass Margins
- Pathology
- Cancer (Target)

+++

### Breast Density

Radiologists classify breast density using a 4-level density scale {cite}`americancollegeofradiologyACRBIRADSAtlas2013`:

1. Almost entirely fatty
2. Scattered areas of fibroglandular density
3. Heterogeneously dense
4. Extremely dense

Note: the corresponding BI-RADS breast density categories are a, b, c, and d (not 1,2,3, and 4 as listed above) so as not to be confused with the BI-RADS assessment categories. Notwithstanding, CBIS-DDSM data encodes these values as ordinal numeric variables.   The following chart illustrates the distribution of BI-RADS breast density categories within the CBIS-DDSM.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
ax = cases.plot.countplot(x='breast_density', ax=ax, title ="Distribution of Breast Density in CBIS-DDSM", plot_counts=True)
glue("eda1_univariate_breast_density", ax)
```

```{glue:figure} eda1_univariate_breast_density
---
align: center
name: eda1_univariate_breast_density_fig
---
Breast Density Distribution in the CBIS-DDSM dataset.
```

+++

### Left or Right Side

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
ax = cases.plot.countplot(x='left_or_right_breast', ax=ax, title ="Distribution of Left/Right Breasts in CBIS-DDSM", plot_counts=True)
glue("eda1_univariate_breast_density", ax)
```

```{glue:figure} eda1_univariate_breast_density
---
align: center
name: eda1_univariate_breast_density_fig
---
Breast Density Distribution in the CBIS-DDSM dataset.
```

The dataset is approximately balanced with respect to left or right breast images.

+++

### Image View

CBIS-DDSM contains digital mammography images in two different image_views: cranial-caudal (CC) and mediolateral oblique (MLO). The CC image_view is taken from above the breast, and best visualizes the subarcolar, central, medial, and posteromedial aspects of the breast. The MLO projection (side-image_view) images the breast in its entirety and best visualizes the posterior and upper-outer quadrants of the breast {cite}`lilleMammographicImagingPractical2019`.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
ax = cases.plot.countplot(x='image_view', ax=ax, title ="Distribution of Image View in CBIS-DDSM", plot_counts=True)
glue("eda1_univariate_view", ax)
```

```{glue:figure} eda1_univariate_view
---
align: center
name: eda1_univariate_view_fig
---
CBIS-DDSM Image Views
```

The proportions of CC and MLO image_views are approximately 47% and 53% respectively.

+++

### Abnormality Id

The abnormality id is a sequence number assigned to each abnormality for a patient.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
ax = cases.plot.countplot(x='abnormality_id', ax=ax, title ="Distribution of Abnormality Counts per Patient in CBIS-DDSM", plot_counts=True)
glue("eda1_univariate_ab_id", ax)
```

```{glue:figure} eda1_univariate_ab_id
---
align: center
name: eda1_univariate_ab_id_fig
---
Distribution of Abnormality Counts per Patient in CBIS-DDSM
```

The vast majority of patients present with a single abnormality; although, a considerable number have two or more.

+++

### Abnormality Type

CBIS-DDSM contains two abnormality types: calcification and mass.

Calcifications, common on mammograms, especially after age 50, are calcium deposits within the breast tissue. Typically benign, calcifications show up as either macrocalcifications or microcalcifications. Macrocalcifications appear as large white dots or dashes which are almost always noncancerous, requiring no further testing or follow-up. Microcalcifications show up as fine, white specks, similar to grains of salt. Usually noncancerous, but certain patterns can be an early sign of cancer.

Masses are also common, particularly among women of reproductive age. For the 25% of women affected by breast disease in their lifetime, the vast majority will present initially with a new breast mass in the primary care setting. Breast masses have a wide range of causes, from physiological adenosis to highly aggressive malignancy.

As shown below, the dataset contains a balance of calcification and mass cases.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
ax = cases.plot.countplot(x='abnormality_type', ax=ax, title ="Distribution of Abnormality Types in CBIS-DDSM", plot_counts=True)
glue("eda1_univariate_ab_type", ax)
```

```{glue:figure} eda1_univariate_ab_type
---
align: center
name: eda1_univariate_ab_type_fig
---
Distribution of Abnormality Types in CBIS-DDSM
```

+++

### Subtlety

Subtlety is a measure of the degree to which a particular case is difficult to diagnose. Values range from 1 (highly subtle) to 5 (obvious).

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
ax = cases.plot.countplot(x='subtlety', ax=ax, title ="Distribution of Subtlety in CBIS-DDSM", plot_counts=True)
glue("eda1_univariate_subtlety", ax)
```

```{glue:figure} eda1_univariate_subtlety
---
align: center
name: eda1_univariate_subtlety_fig
---
Distribution of Subtlety in CBIS-DDSM
```

Approximately 17% of the cases are highly subtle (1,2). A plurality of cases are moderately to slightly subtle and nearly a 1/3rd of the cases are considered obvious.

+++

### BI-RADS Assessment

A BI-RADS assessment is based upon a thorough evaluation of the mammographic features of concern and has the following six categories {cite}`americancollegeofradiologyACRBIRADSAtlas2013`:

```{table} BI-RADS Assessment
:name: eda1_univariate_birads
| Category | Definition                                                                                                                                                         |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0        | Means the findings are unclear. The radiologist will need more images to determine a score                                                                       |
| 1        | Means the findings are negative and the breast tissue appears normal. No masses, calcifications, asymmetry, or other abnormalities have been found.             |
| 2        | Means the findings are benign, which is also negative for cancer. While a   mass, calcification, or other abnormality may have been detected, it’s not cancerous. |
| 3        | Means the findings are probably benign. While a mass, calcification, or other abnormality may have been found, it’s most likely not cancerous.                    |
| 4        | Means cancer is suspected. Four subcategories relate the probability of a malignancy: 4A (2-10%) 4B (10-50%) 4C (50-95%)                       |
| 5        | Means cancer is highly suspected. Findings have a 95% chance or higher of being cancerous.                                                                       |
| 6        | Cancer was previously diagnosed using a biopsy.
                                                                                                                  |
```

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
ax = cases.plot.countplot(x='assessment', ax=ax, title ="Distribution of BI-RADS Assessment in CBIS-DDSM", plot_counts=True)
glue("eda1_univariate_birads_plot", ax)
```

```{glue:figure} eda1_univariate_birads_plot
---
align: center
name: eda1_univariate_birads_plot_fig
---
Distribution of BI-RADS Assessment in CBIS-DDSM
```

+++

### Calcification Type

Calcification type describes the morphology of a case and is the most important factor in the differentiation between benign and malignant cases. There are over 40 different categories of calc_type in the dataset; and, the some of the main categories are {cite}`americancollegeofradiologyACRBIRADSAtlas2013`:

```{table} BI-RADS Assessment
:name: eda1_univariate_calc_type
| #  | Morphology            |                                                                                                                | Assessment     |
|----|-----------------------|----------------------------------------------------------------------------------------------------------------|----------------|
| 1  | Amorphous             | Indistinct calcifications, without clearly defined shape, small and/or   hazy in appearance                    | BI-RADS 4B     |
| 2  | Coarse Heterogeneous  | Irregular, conspicuous calcifications, typically larger than 0.5 mm.                                           | BI-RADS 3      |
| 3  | Dystrophic            | Irregular, 'lava-shaped', larger than 0.5 mm, which develop 3-5 years   after treatment in about 30% of women. | BI-RADS 1 or 2 |
| 4  | Eggshell              | Very thin benign calcifications that appear as calcium                                                         | BI-RADS 1 or 2 |
| 5  | Fine Linear Branching | Thin linear or curvilinear irregular.                                                                          | BI-RADS 4B     |
| 6  | Large Rod-like        | Benign calcifications that form continuous rods that may occassionally be   branching.                         | BI-RADS 1 or 2 |
| 7  | Lucent-Centered       | Round or oval calcifications, typically result of fat necrosis, or   calcified debris in ducts                 | BI-RADS 1 or 2 |
| 8  | Milk of Calcium       | Benign sedimented calcifications in macro or microcysts.                                                       | BI-RADS 1 or 2 |
| 9  | Pleomorphic           | Calcifications varying in size and shape, more conspicuous than amorphic   calcifications.                     | BI-RADS 4B     |
| 10 | Punctate              | Round calcifications 0.5-1 mm in size.                                                                         | BI-RADS 2,3,4  |
| 11 | Skin                  | Skin calcifications, usually lucent-centered deposits                                                          | BI-RADS 1 or 2 |
| 12 | Vascular              | Linear or form parallel tracks, usually associated with blood vessels.                                         | BI-RADS 1 or 2 |
```

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,10))
ax = calc.plot.countplot(y='calc_type', ax=ax, title ="Distribution of Calcification Types in CBIS-DDSM", plot_counts=True, order_by_count=True)
glue("eda1_univariate_calc_type", ax)
```

```{glue:figure} eda1_univariate_calc_type
---
align: center
name: eda1_univariate_calc_type_fig
---
Distribution of Calcification Types in CBIS-DDSM
```

Pleomorphic and amorphous calcifications account for over half of the calcification cases in the dataset. Nearly 75% of the calcification cases are represented by five types.

+++

### Calcification Distribution

Calcification distribution refers to the arrangement of the calcifications inside the breast. BI-RADS describes the following categories of calcification distribution {cite}`americancollegeofradiologyACRBIRADSAtlas2013`:

1. Diffuse or Scattered: Calcifications throughout the whole breast.
2. Regional: Scattered in a larger volume (> 2 cc) of breast tissue and not in the expected ductal distribution.
3. Clustered: Groups of at least 5 calcifications in a small volume of tissue
4. Segmental: Calcium deposits appear in ducts or branches of a segment or lobe.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,6))
ax = calc.plot.countplot(y='calc_distribution', ax=ax, title ="Distribution of Calcification Distributions in CBIS-DDSM", plot_counts=True, order_by_count=True)
glue("eda1_univariate_calc_dist", ax)
```

```{glue:figure} eda1_univariate_calc_dist
---
align: center
name: eda1_univariate_calc_dist_fig
---
Distribution of Calcification Distributions in CBIS-DDSM
```

Over 80% of the calfication cases have either clustered or segmental distributions.

+++

### Mass Shape

The BI-RADS lexicon defines three mass shapes {cite}`americancollegeofradiologyACRBIRADSAtlas2013`:
1. Round
2. Oval
3. Irregular

The CBIS-DDSM; however, includes additional categories that further describe the mass shape, symmetry, and architecture.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,6))
ax = mass.plot.countplot(y='mass_shape', ax=ax, title ="Distribution of Mass Shapes in CBIS-DDSM", plot_counts=True, order_by_count=True)
glue("eda1_univariate_mass_shape", ax)
```

```{glue:figure} eda1_univariate_mass_shape
---
align: center
name: eda1_univariate_mass_shape_fig
---
Distribution of Mass Shapes in CBIS-DDSM
```

+++

### Mass Margins

Mass margins are features that separate the mass from the adjacent breast parenchyma. Mass margins can be {cite}`americancollegeofradiologyACRBIRADSAtlas2013`:

1. Circumscribed: Low probability of malignancy.
2. Obscured: Undetermined likelihood of malignancy.
3. Spiculated: Higher likelihood of malignancy.
4. Microlobulated: Suspicious for breast carcinoma:
5. Ill-Defined: Also called 'indistinct'. Generally suspicious of malignancy.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,6))
ax = mass.plot.countplot(y='mass_margins', ax=ax, title ="Distribution of Mass Margins in CBIS-DDSM", plot_counts=True, order_by_count=True)
glue("eda1_univariate_mass_margins", ax)
```

```{glue:figure} eda1_univariate_mass_margins
---
align: center
name: eda1_univariate_mass_margins_fig
---
Distribution of Mass Margins in CBIS-DDSM
```

Spiculated, circumscribed and ill-defined make up nearly 70% of the mass abnormalities.

+++

### Pathology

The dataset distinguishes three outcomes: malignant, benign, and benign without callback. The latter indicates that the region may be suspicious, and should be monitored, but no further investigation is required.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
ax = cases.plot.countplot(x='pathology', ax=ax, title ="Distribution of Pathology in CBIS-DDSM", plot_counts=True, order_by_count=True)
glue("eda1_univariate_pathology", ax)
```

```{glue:figure} eda1_univariate_pathology
---
align: center
name: eda1_univariate_pathology_fig
---
Distribution of Pathology in CBIS-DDSM
```

The majority of cases are benign; although, benign without callback represents a considerable proportion of the cases.

+++

### Cancer

Here, we collapse BENIGN and BENIGN_WITHOUT_CALLBACK into a single category.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
ax = cases.plot.countplot(x='cancer', ax=ax, title ="Distribution of Cancer Diagnoses in CBIS-DDSM")
glue("eda1_univariate_cancer", ax)
```

```{glue:figure} eda1_univariate_cancer
---
align: center
name: eda1_univariate_cancer_fig
---
Distribution of Cancer Diagnoses in CBIS-DDSM
```

+++

### Summary CBIS-DDSM Case Univariate Analysis

Several observations can be made at this stage.

1. The CBIS-DDSM is well-balanced with respect to breast density, morphology, subtlety, BI-RADS assessment, and pathology.
2. Over 40 calcification types are represented; however, the majority of cases fall into one of five types.
3. Similarly, there are nearly 20 categories of mass margins; yet, but most cases fall into one of the five major classes:
   1. Spiculated
   2. Circumscribed
   3. Obscured
   4. Ill-Defined

Next up? Bivariate analysis.

+++ {"tags": ["hide-input"]}

## Case Bivariate Analysis

This bivariate analysis will comprise a dependency analysis and an inter-dependence analysis.  The former assesses the degree to which a cancer diagnosis depends upon the values of the explanatory variables, such as breast density, type and distribution of calcifications, and the shape and margins of masses. The inter-dependence analysis explores the association between two independent variables.

```{code-cell}
:tags: [hide-input]

df = cases.as_df(categorize_ordinals=True)
```

+++

### Bivariate Target Variable Association Analysis

#### Cancer Diagnosis by Breast Density

```{code-cell}
:tags: [hide-input, remove-output]

p = sns.objects.Plot(df, x='breast_density', color='cancer').add(so.Bar(), so.Count(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Breast Density").layout(size=(12,4), engine='tight')
glue("eda1_bivariate_diagnosis_density", p)
```

```{glue:figure} eda1_bivariate_diagnosis_density
---
align: center
name: eda1_bivariate_diagnosis_density_fig
---
Diagnosis by Breast Density
```

It's rather difficult to assess the degree to which breast density related to a diagnosis. Let's plot the relative proportions.

```{code-cell}
:tags: [hide-input, remove-output]

prop = df[['breast_density', 'cancer']].groupby(by=['breast_density']).value_counts(normalize=True).to_frame().reset_index().sort_values(by=['breast_density','cancer'])
p = sns.objects.Plot(prop, x='breast_density', y='proportion', color='cancer').add(so.Bar(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Breast Density").layout(size=(12,4), engine='tight')
glue("eda1_bivariate_diagnosis_density_prop", p)
```

```{glue:figure} eda1_bivariate_diagnosis_density_prop
---
align: center
name: eda1_bivariate_diagnosis_density_prop_fig
---
Diagnosis by Breast Density
```

Breast density is considered a risk factor for breast cancer, as women with dense breasts have a higher risk of breast cancer than women with fatty breasts {cite}`DenseBreastsAnswers2018`. Notwithstanding, the CBIS-DDSM data don't reveal a strong relationship between breast density and diagnosis. Let's see if a test of association supports our inference.

```{code-cell}
:tags: [hide-input]

kt = cases.stats.kendallstau(a='breast_density', b='cancer')
print(kt)
```

The Kendall's Tau test measuring the association between breast density and malignancy indicated a non-significant association of weak effect, ($\phi_\tau$ = 0.01, p = 0.54).

+++

#### Cancer Diagnosis by Breast Side

A 2022 study published in Nature {cite}`abdouLeftSidedBreast2022` suggests that breast cancer is slightly more prevalent on the left side of the body than it is on the right. Do the CBIS-DDSM data support this finding?

```{code-cell}
:tags: [hide-input]

p = sns.objects.Plot(df, x='left_or_right_breast', color='cancer').add(so.Bar(), so.Count(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Breast Side").layout(size=(12,4), engine='tight')
glue("eda1_bivariate_diagnosis_side", p)
```

```{glue:figure} eda1_bivariate_diagnosis_side
---
align: center
name: eda1_bivariate_diagnosis_side_fig
---
Diagnosis by Breast Side
```

If there is a slightly greater risk of cancer in the left breast, it would not be evident in the CBIS-DDSM data.

```{code-cell}
:tags: [hide-input]

cv = cases.stats.cramersv(a='left_or_right_breast', b='cancer')
print(cv)
```

The chi-square test above, indicates a non-significant association of negligible effect between breast and diagnosis, ($X^2$ (1,n=3566)=2.97 p=0.08, $\phi$=.03).

+++

#### Cancer by Image View

A study published in RSNA Journals {cite}`korhonenBreastCancerConspicuity2019` analyzed breast cancer conspicuity by image_view and determined that cancers were more likely to have high conspicuity in the craniocaudal (CC) than the mediolateral oblique (MLO) image_view.  Let's see what our data suggest.

```{code-cell}
:tags: [hide-input]

p = sns.objects.Plot(df, x='image_view', color='cancer').add(so.Bar(), so.Count(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Image View").layout(size=(12,4), engine='tight')
glue("eda1_bivariate_diagnosis_view", p)
```

```{glue:figure} eda1_bivariate_diagnosis_view
---
align: center
name: eda1_bivariate_diagnosis_view_fig
---
Diagnosis by Image View
```

```{code-cell}
:tags: [hide-input, remove-output]

df_cancer_by_view = df[[ 'cancer', 'image_view']].groupby(by=['image_view']).value_counts(normalize=True).to_frame()
glue("eda1_bivariate_cancer_by_view", df_cancer_by_view)
```

```{glue:figure} eda1_bivariate_cancer_by_view
---
align: center
name: eda1_bivariate_cancer_by_view_fig
---
Diagnosis by Image View
```

Both image_views have the same proportion of malignancies suggesting no association between image image_view and the diagnosis.

```{code-cell}
:tags: [hide-input]

cv = cases.stats.cramersv(a='image_view', b='cancer')
print(cv)
```

The chi-square test above, indicates a non-significant association of negligible effect between image image_view and diagnosis, ($X^2$ (1,n=3566)=0.007 p=0.93, $\phi$=.002).

+++

#### Cancer by Abnormality Type

Are masses more or less malignant than calcifications?

```{code-cell}
:tags: [hide-input]

p = sns.objects.Plot(df, x='abnormality_type', color='cancer').add(so.Bar(), so.Count(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Abnormality Type").layout(size=(12,4), engine='tight')
glue("eda1_bivariate_cancer_by_view)
```

```{glue:figure} eda1_bivariate_cancer_by_view
---
align: center
name: eda1_bivariate_cancer_by_view_fig
---
Diagnosis by Abnormality Type
```

```{code-cell}
:tags: [hide-input]

df_cancer_by_ab_type = df[[ 'cancer', 'abnormality_type']].groupby(by=['cancer']).value_counts(normalize=True).to_frame()
glue("eda1_bivariate_cancer_by_ab_type", df_cancer_by_ab_type)
```

```{glue:figure} eda1_bivariate_cancer_by_ab_type
---
align: center
name: eda1_bivariate_cancer_by_ab_type_fig
---
Diagnosis by Abnormality Type
```

These data indicate that the probability of a malignancy is slightly higher for masses than calcifications. Is this significant?

```{code-cell}
:tags: [hide-input]

cv = cases.stats.cramersv(a='abnormality_type', b='cancer')
print(cv)
```

The chi-square test above, indicates a significant association of small effect between abnormality type and diagnosis, ($X^2$ (1,n=3566)=38.85 p<0.01, $\phi$=.10). More malignancies were diagnosed among the mass cases, compared to calcifications (54% vs 46%).

+++

#### BI-RADS Assessment and Cancer

To what degree is there (dis)agreement between the BI-RADS assessment the diagnosis. The BI-RADS assessment is an overall summary of the mammography report and has seven categories.

```{table} BI-RADS Assessments
:name: eda1_birads_assessment

| Label | Description                                                                              | Likelihood of Cancer                                                        |
| ----- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 0     | Incomplete . Need Additional   Imaging Evaluation and/or Prior Mammograms for Comparison | NA                                                                          |
| 1     | Negative Routine mammography                                                             | Essentially 0% likelihood of malignancy                                     |
| 2     | Benign Routine mammography                                                               | Essentially 0% likelihood of malignancy                                     |
| 3     | Probably Benign Short-interval   (6-month)                                               | > 0% but . 2% likelihood of malignancy                                      |
| 4     | Suspicious                                                                               | Low: 2% to ≤ 10%      Moderate: > 10% to ≤ 50%       High: > 50% to < 95% |
| 5     | Highly Suggestive of Malignancy                                                          | > 95% likelihood of malignancy                                              |
| 6     | Known Biopsy-Proven Malignancy                                                           | NA                                                                          |
```

To what degree is there agreement between the BI-RADS assessments and diagnosis.

```{code-cell}
:tags: [hide-input]

sns.objects.Plot(df, x='assessment', color='cancer').add(so.Bar(), so.Count(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by BI-RADS Assessment").layout(size=(12,4), engine='tight')
prop = df[['assessment', 'cancer']].groupby(by=['assessment']).value_counts(normalize=True).to_frame().reset_index().sort_values(by=['assessment','cancer'])
sns.objects.Plot(prop, x='assessment', y='proportion', color='cancer').add(so.Bar(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by BI-RADS Assessment").layout(size=(12,4), engine='tight')
```

```{code-cell}
:tags: [hide-input]

counts = df[[ 'assessment', 'cancer']].groupby(by=['assessment']).value_counts(normalize=False).to_frame().sort_values(by=['assessment', 'cancer'])
prop = df[[ 'assessment', 'cancer']].groupby(by=['assessment']).value_counts(normalize=True).to_frame().sort_values(by=['assessment', 'cancer'])
pd.concat([counts, prop], axis=1)
```

+++ {"tags": ["hide-input"]}

These data do show a relationship between BI-RADS assessment and diagnosis. Let's evaluate the effect size.

```{code-cell}
:tags: [hide-input]

kt = cases.stats.kendallstau(a='assessment', b='cancer')
print(kt)
```

+++ {"tags": ["hide-input"]}

Indeed, the Kendall's Tau test above, indicates a significant association of strong effect between BI-RADS assessment and diagnosis, ($\phi_\tau$=0.60, p<0.01).

+++ {"tags": ["hide-input"]}

Several observations.
1. Incomplete assessments had a malignancy approaching 25%.
2. There were only three BI-RADS Category 1 assessments.
3. Category 2 assessments, were all negative for cancer.
4. Approximately 20% of the BI-RADS category 3, probably benign, were ultimately diagnosed as malignant.
5. Suspicious cases (BI-RADS 4) were nearly 50/50 benign/malignant.
6. Just 2% of the highly suspicious cases were benign.

+++ {"tags": ["hide-input"]}

##### Subtlety and Cancer
Are malignancies more or less conspicuous?

```{code-cell}
:tags: [hide-input]

sns.objects.Plot(df, x='subtlety', color='cancer').add(so.Bar(), so.Count(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Subtlety").layout(size=(12,4), engine='tight')
```

+++ {"tags": ["hide-input"]}

Any association between subtlety and malignancy isn't clear from the above. Let's examine the relative proportions of malignancy vis-a-vis subtlety.

```{code-cell}
:tags: [hide-input]

prop = df[['subtlety', 'cancer']].groupby(by=['subtlety']).value_counts(normalize=True).to_frame().reset_index().sort_values(by=['subtlety','cancer'])
sns.objects.Plot(prop, x='subtlety', y='proportion', color='cancer').add(so.Bar(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Subtlety").layout(size=(12,4), engine='tight')
```

+++ {"tags": ["hide-input"]}

An association seems to be present. Let's look at the numbers.

```{code-cell}
:tags: [hide-input]

counts = df[[ 'subtlety', 'cancer']].groupby(by=['subtlety']).value_counts(normalize=False).to_frame().sort_values(by=['subtlety', 'cancer'])
prop = df[[ 'subtlety', 'cancer']].groupby(by=['subtlety']).value_counts(normalize=True).to_frame().sort_values(by=['subtlety', 'cancer'])
pd.concat([counts, prop], axis=1)
```

+++ {"tags": ["hide-input"]}

Again, it would be difficult to draw an inference of association between subtlety and diagnosis.

```{code-cell}
:tags: [hide-input]

kt = cases.stats.kendallstau(a='subtlety', b='cancer')
print(kt)
```

+++ {"tags": ["hide-input"]}

The Kendall's Tau test measuring the association between subtlety and malignancy indicated a non-significant association of weak effect, $\phi_\tau$ = 0.003, p = 0.86.

+++ {"tags": ["hide-input"]}

##### Calcification Type and Cancer
What is the association between calcification type and malignancy. According to the literature, fine linear branching, and pleomorphic calcifications are of the highest concern, followed by amorphous and coarse heterogenous abnormalities.

```{code-cell}
:tags: [hide-input]

df_calc = calc.as_df()
prop = df_calc[['calc_type', 'cancer']].groupby(by=['calc_type']).value_counts(normalize=True).to_frame().reset_index().sort_values(by=['calc_type','cancer'])
sns.objects.Plot(prop, y='calc_type', x='proportion', color='cancer').add(so.Bar(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Calcification Type").layout(size=(12,12), engine='tight')
```

+++ {"tags": ["remove-output", "hide-input"]}

As this plot suggests, an association between calcification type and diagnosis is extant. Let's check the strength of this association.

```{code-cell}
:tags: [hide-input]

cv = calc.stats.cramersv(a='calc_type', b='cancer')
print(cv)
```

+++ {"tags": ["hide-input"]}

This chi-square test of independence between calcification type and diagnosis indicates a significant association of large effect ($X^2$(1,n=1872)=539.69 p<0.01, $V$=0.54).

+++ {"tags": ["hide-input"]}

The following lists the top 10 most malignant calcification types by proportion in the CBIS-DDSM.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,6))
calc_types = cases.get_most_malignant_calc(x='calc_type', n=10)
calc.plot.barplot(data=calc_types, y='calc_type', x='proportion', title='Malignancy by Calcification Type', ax=ax)
```

+++ {"tags": ["hide-input"]}

##### Calcification Distribution
How do we characterize the association between calcification distribution and malignancy?

```{code-cell}
:tags: [hide-input]

df_calc = calc.as_df()
prop = df_calc[['calc_distribution', 'cancer']].groupby(by=['calc_distribution']).value_counts(normalize=True).to_frame().reset_index().sort_values(by=['calc_distribution','cancer'])
sns.objects.Plot(prop, y='calc_distribution', x='proportion', color='cancer').add(so.Bar(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Calcification Distribution").layout(size=(12,8), engine='tight')
```

+++ {"tags": ["hide-input"]}

Again, we see an association between calcification distribution and malignancy.  Let's check the most malignant calcification distributions.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
calc_types = cases.get_most_malignant_calc(x='calc_distribution', n=5)
calc.plot.barplot(data=calc_types, y='calc_distribution', x='proportion', title='Malignancy by Calcification Distribution', ax=ax)
```

```{code-cell}
:tags: [hide-input]

cv = calc.stats.cramersv(a='calc_distribution', b='cancer')
print(cv)
```

+++ {"tags": ["hide-input"]}

This chi-square test of independence between calcification distribution and diagnosis indicates a moderate and significant association ($X^2$(1,n=1872)=198.56 p<0.01, $V$=0.33).

+++ {"tags": ["hide-input"]}

##### Mass Shape and Cancer
Mass shape and mass margins are the most significant features that indicate whether a mass is benign or malignant {cite}`bassettAbnormalMammogram2003`.

```{code-cell}
:tags: [hide-input]

df_mass = mass.as_df()
prop = df_mass[['mass_shape', 'cancer']].groupby(by=['mass_shape']).value_counts(normalize=True).to_frame().reset_index().sort_values(by=['mass_shape','cancer'])
sns.objects.Plot(prop, y='mass_shape', x='proportion', color='cancer').add(so.Bar(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Mass Shape").layout(size=(12,8), engine='tight')
```

+++ {"tags": ["hide-input"]}

As suggested, an association between mass shape and diagnosis is evident. Which mass shapes are most malignant?

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
mass_shapes = cases.get_most_malignant_mass(x='mass_shape', n=10)
mass.plot.barplot(data=mass_shapes, y='mass_shape', x='proportion', title='Malignancy by Mass Shape', ax=ax)
```

```{code-cell}
:tags: [hide-input]

cv = mass.stats.cramersv(a='mass_shape', b='cancer')
print(cv)
```

+++ {"tags": ["hide-input"]}

This chi-square test of independence between mass shape and diagnosis indicates a large and significant association ($X^2$(1,n=1694)=440.92 p<0.01, $V$=0.51).

+++ {"tags": ["hide-input"]}

##### Mass Margins and Cancer
Mass margins, a feature that separates the mass from the adjacent breast parenchyma, is often the feature which enables differentiation between benign and malignant.

```{code-cell}
:tags: [hide-input]

prop = df_mass[['mass_margins', 'cancer']].groupby(by=['mass_margins']).value_counts(normalize=True).to_frame().reset_index().sort_values(by=['mass_margins','cancer'])
sns.objects.Plot(prop, y='mass_margins', x='proportion', color='cancer').add(so.Bar(), so.Stack()).theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"}).label(title="Diagnosis by Mass Margins").layout(size=(12,8), engine='tight')
```

+++ {"tags": ["hide-input"]}

As expected, mass margins appear determinative. Which mass margins are most concerning?

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
mass_margins = cases.get_most_malignant_mass(x='mass_margins', n=10)
mass.plot.barplot(data=mass_margins, y='mass_margins', x='proportion', title='Malignancy by Mass Margins', ax=ax)
```

```{code-cell}
:tags: [hide-input]

cv = mass.stats.cramersv(a='mass_margins', b='cancer')
print(cv)
```

+++ {"tags": ["hide-input"]}

This chi-square test of independence between mass margins and diagnosis indicates a large and significant association ($X^2$(1,n=1694)=588.62 p<0.01, $V$=0.59).

That concludes the target association section of the bivariate analysis. Let's examine the relationships among the features.

+++ {"tags": ["hide-input"]}

#### Case Bivariate Feature Association Analysis
What does calcification type imply about calcification distribution? To what degree is mass shape and mass margin related? Are certain morphologies more or less subtle? In this section, we examine the strength of associations among the features in the CBIS-DDSM using Cramer's V effect size measurement. For this exercise, we'll interpret the Cramer's V effect sizes as follows:

|      V      |   Effect   |
|:-----------:|:----------:|
|    ≥ 0.5    |    Large   |
| ≥ 0.3 < 0.5 |  Moderate  |
| ≥ 0.1 < 0.3 |    Small   |
|   0 < 0.1   | Negligible |

We'll start with the full dataset, to investigate the relationships between non-morphological features. Then, we'll analyze mass and calcification cases separately to avoid in spurious associations across abnormality types.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(8,8))
cases.plot_feature_associations(ax=ax)
```

+++ {"tags": ["hide-input"]}

We'll ignore abnormality type and the morphology associationss for now. Calcification and mass morphology associations will be analyzed separately. That said, several observations can be made:
1. Abnormality type has a moderate association with BI-RADS assessment.
2. There appears to be weak associations among the non-morphological features.

Let's take a look breast density vis-a-vis abnormality type.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,4))
df = cases.as_df()
df_props = df[['abnormality_type', 'breast_density']].groupby(by=['abnormality_type']).value_counts(normalize=True, sort=False).to_frame().reset_index()
cases.plot.barplot(data=df_props, x='breast_density', y='proportion', hue='abnormality_type', ax=ax)
```

+++ {"tags": ["hide-input"]}

Abnormality types tends to behave similarly across breast density categories, thus the weak association.

+++ {"tags": ["hide-input"]}

##### Calcification Feature Bivariate Analysis
Let's examine the relationships among the features among the calcification cases.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(8,8))
calc.plot_calc_feature_associations(ax=ax)
```

+++ {"tags": ["hide-input"]}

The moderate to strong associations of note are:
- Calcification Type and BI-RADS assessment (0.54)
- Calcification Type and Distribution (0.45)
- Calcification Distribution and BI-RADS assessment (0.39)
- Breast Density and Calcification Type (0.36)
- Calcification Type and Subtlety (0.33)

Let's visualize these relationships and assess their statistical significance.

+++ {"tags": ["hide-input"]}

###### Calcification Type and Assessment

```{code-cell}
:tags: [hide-input]

_ = cases.summarize_morphology_by_feature(morphology='calc_type', by='assessment', figsize=(12,12))
```

+++ {"tags": ["hide-input"]}

Above, we show the proportion of BI-RADS assessments for each calcification type. Pleomorphic, fine linear branching and amorphous calcifications appear to render the highest levels of suspicion with BI-RADS 4 and 5 assessments. Those of intermediate concern are dystrophic and punctate. The remaining are associated with generally benign classifications.

There is a strong association between calcification type and BI-RADS assessment, but is it a statistically significant finding?

```{code-cell}
:tags: [hide-input]

cv = calc.stats.cramersv(a='calc_type', b='assessment')
print(cv)
```

+++ {"tags": ["hide-input"]}

Indeed, the result is signficant ($X^2$(4,n=1872)=2183.62 p<0.01, $V$=0.54).

In the prior section, we separated compound calcification types into separate categories; thereby, reducing the number of calcification types from 40 to 13. Let's examine the relationship between these calcification types and assessment.

+++ {"tags": ["hide-input"]}

###### Calcification Type and Calcification Distribution

```{code-cell}
:tags: [hide-input]

_ = cases.compare_morphology(m1='calc_type', m2='calc_distribution', figsize=(12,12))
```

+++ {"tags": ["hide-input"]}

For each calcification type, we show the calcification distributions by proportion of calcification type cases in which they co-occur. For instance, amorphous types tend to co-occur with clustered distributions. Eggshell calcifications appear exclusively with segmental distributions in the CBIS-DDSM.  Let's check the statistical signficance.

```{code-cell}
:tags: [hide-input]

cv = calc.stats.cramersv(a='calc_type', b='calc_distribution')
print(cv)
```

+++ {"tags": ["hide-input"]}

Again, we observe a statistically significant large association between calcification type and distribution ($X^2$(8,n=1872)=3087.3 p<0.01, $V$=0.45).

+++ {"tags": ["hide-input"]}

###### Calcification Type and Subtlety

```{code-cell}
:tags: [hide-input]

_ = cases.summarize_morphology_by_feature(morphology='calc_type', by='subtlety')
```

+++ {"tags": ["hide-input"]}

Here, we see that dystrophic, course, large_rodlike, lucent_centered, and round and regular calcifications present less conspicuously than the others.

```{code-cell}
:tags: [hide-input]

cv = calc.stats.cramersv(a='calc_type', b='subtlety')
print(cv)
```

+++ {"tags": ["hide-input"]}

Calcification type and subtlety are strongly related ($X^2$(4,n=1872)=793 p<0.01, $V$=0.33).

+++ {"tags": ["hide-input"]}

###### Calcification Distribution and BI-RADS Assessment

```{code-cell}
:tags: [hide-input]

_ = cases.summarize_morphology_by_feature(morphology='calc_distribution', by='assessment', figsize=(12,6))
```

+++ {"tags": ["hide-input"]}

Clustered and linear calcification distributions tend to be suspicious with assessments in the BI-RADS 4 range. Regional distributions can indicate anything from benign to highly suspicious. Diffusely scattered and segmental distributions are primarily considered benign in the CBIS-DDSM.

```{code-cell}
:tags: [hide-input]

cv = calc.stats.cramersv(a='calc_distribution', b='assessment')
print(cv)
```

+++ {"tags": ["hide-input"]}

Calcification distribution and BI-RADS assessment are strongly related ($X^2$(4,n=1872)=1127 p<0.01, $V$=0.39).

+++ {"tags": ["hide-input"]}

###### Calcification Type and Breast Density

```{code-cell}
:tags: [hide-input]

_ = cases.summarize_morphology_by_feature(morphology='calc_type', by='breast_density', figsize=(12,8))
```

+++ {"tags": ["hide-input"]}

Amorphous, course, fine linear branching, lucent centered, and pleomorphic almost follow a normal distribution. Some cases have fatty breasts, some extremely dense; but, most are in the middle categories 2 and 3. Eggshell, large rodlike, milk of calcium, skin, and vascular types stand out as co-occuring with breasts of specific densities.

```{code-cell}
:tags: [hide-input]

cv = calc.stats.cramersv(a='breast_density', b='calc_type')
print(cv)
```

+++ {"tags": ["hide-input"]}

Again, we obsere a strong association between calcification type and breast density ($X^2$(3,n=1872)=709 p<0.01, $V$=0.36).

+++ {"tags": ["hide-input"]}

##### Mass Feature Bivariate Analysis
Next, let's examine feature relationships for mass cases.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(8,8))
mass.plot_mass_feature_associations(ax=ax)
```

+++ {"tags": ["hide-input"]}

Here, we have the following moderate to strong associations within the mass dataset:
- Mass Margin and BI-RADS Assesment (0.40)
- Mass Shape and BI-RADS Assesment (0.37)

It's notable that mass shape and mass margins are weakly associated.

+++ {"tags": ["hide-input"]}

###### Mass Margins and BI-RADS Assessment

```{code-cell}
:tags: [hide-input]

_ = cases.summarize_morphology_by_feature(morphology='mass_margins', by='assessment', figsize=(12,8))
```

+++ {"tags": ["hide-input"]}

Spiculated, ill-defined, and microlobulated masses appear to be of greater concern than obscured, and perhaps circumscribed mass margins.

```{code-cell}
:tags: [hide-input]

cv = mass.stats.cramersv(a='mass_margins', b='assessment')
print(cv)
```

+++ {"tags": ["hide-input"]}

The relationship between mass margins and BI-RADS assessment is strong ($X^2$(5,n=1694)=1358 p<0.01, $V$=0.40).

+++ {"tags": ["hide-input"]}

###### Mass Shape and Assessment

```{code-cell}
:tags: [hide-input]

_ = cases.summarize_morphology_by_feature(morphology='mass_shape', by='assessment', figsize=(12,8))
```

+++ {"tags": ["hide-input"]}

Architectural distortion, irregular, oval, and round shapes tend to render the most concern.

```{code-cell}
:tags: [hide-input]

cv = mass.stats.cramersv(a='mass_shape', b='assessment')
print(cv)
```

+++ {"tags": ["hide-input"]}

Similarly, the relationship between mass shape and BI-RADS assessment is strong ($X^2$(5,n=1694)=1156 p<0.01, $V$=0.37).

+++ {"tags": ["hide-input"]}

#### Summary CBIS-DDSM Case Bivariate Analysis
This concludes the bivariate component of this exploratory data analysis. Let's summarize our observations thus far.

Our bivariate analysis was conducted in two parts:
1. Bivariate Target Variable Association Analysis
2. Bivariate Feature Association Analysis

The former examined the relationships between the features and the target variable, 'cancer' and the latter explored the relationships between feature pairs.

```{code-cell}
:tags: [remove-output, hide-input]

cases.plot_target_associations()
```

+++ {"tags": ["remove-output", "hide-input"]}

##### Bivariate Target Association Analysis Summary
The plot above depicts the Cramer's V associations between the independent variables and our target variable, 'cancer'. The main observations are as follows:
1. There was strong agreement between the BI-RADS assessments and the pathology ($\tau$=0.60, p<0.01); an expected finding as many of the BI-RADS assessments were likely updated as additional information was gathered by the physician {cite}`leeCuratedMammographyData2017`.
2. Mass margins had the strongest association with pathology ($X^2$(1,n=1694)=588.62 p<0.01, $V$=0.59). Of the 19 categories, the margins most associated with malignancy were:
   1. Obscured-Spiculated,
   2. Obscured-Ill-Defined,
   3. Circumscribed-Spiculated,
   4. Microlobulated-Spiculated, and
   5. Microlobulated-Ill-Defined-Spiculated
   Indeed, 100% of the above cases were malignant.
3. Calcification type was also strongly associated with pathology ($X^2$(1,n=1872)=539.69 p<0.01, $V$=0.54). Punctate fine linear branching, punctate amorphous pleomorphic, and amorphous round and regular had malignancy rates of 100% in the dataset. Pleomorphic fine linear branching, fine linear branching, amorphous pleomorphic, and pleomorphic all had malignancy rates exceeding 50%.
4. Mass shape had a large and significant effect on pathology in the dataset ($X^2$(1,n=1694)=440.92 p<0.01, $V$=0.51). Lobulated, round irregular, and irregular architectural distortion were associated with malignancy rates exceeding 80%.
5. Calcification distribution had a moderate effect on pathology.  ($X^2$(1,n=1872)=198.56 p<0.01, $V$=0.33). Linear segmental, clustered linear, and linear calcification distributions were associated with malignancy rates above 50%.
6. Subtlety, abnormality type, breast, breast density, and image image_view were all weakly associated with pathology.

+++ {"tags": ["hide-input"]}

##### Bivariate Feature Association Analysis Summary
Cramer's V was used to measure the degree of association between the features. We observed several strong associations in both calcification and mass cases:
1. Calcification Cases:
   1. There was a strong association (V=0.54) between calcification type and BI-RADS assessment.
   2. Moderate associations were observed between:
      1. breast density and calcification type
      2. calcification type and distribution
      3. calcification type and subtlety
      4. calcification distribution and BI-RADS assessment
   3. The other features were weakly associated.
2. Mass Cases:
   1. No strong associations were observed among the features in the mass dataset.
   2. Moderate associations were observed between:
      1. mass shape and BI-RADS assessment
      2. mass margins and BI-RADS assessment
   3. Other features were weakly associated.

Next, multivariate data analysis.

+++ {"tags": ["hide-input"]}

### Case Data Multivariate Analysis
The purpose of the multivariate analysis is to elucidate features of the data, beyond that which can be derived from the univariate, and bivariate analyses above. Our objectives; therefore, are to:
1. Identify the variables that most impact the diagnosis of cancer,
2. Estimate the importance of each independent variable in explaining the diagnosis.
3. Establish a model to predict a diagnosis, given the independent variables.

 Logistic Regression, Support Vector Machines, and Random Forests classifiers will be trained on the calcification and mass training data. The models will be evaluated using cross-validation, and the best algorithm will make predictions on the entire mass or calcification dataset.

```{code-cell}
:tags: [remove-cell, hide-input]

FORCE_MODEL_FIT = False
```

+++ {"tags": ["hide-input"]}

#### Build Pipelines
Pipelines are built for each classifier.

```{code-cell}
:tags: [hide-input]

pb = PipelineBuilder()
pb.set_jobs(6)
pb.set_standard_scaler()
pb.set_scorer('accuracy')

# Build Logistic Regression Pipeline
params_lr = [{'clf__penalty': ['l1', 'l2'],
		      'clf__C': [1.0, 0.5, 0.1],
		      'clf__solver': ['liblinear']}]
clf = LogisticRegression(random_state=5)
pb.set_classifier(classifier=clf, params=params_lr)
pb.build_gridsearch_cv()
lr = pb.pipeline

# Build SVM Pipeline
clf = SVC(random_state=5)
params_svc = [{'clf__kernel': ['linear'],
		       'clf__C': [1,2,3,4,5, 6, 7, 8, 9, 10]}]
pb.set_classifier(classifier=clf, params=params_svc)
pb.build_gridsearch_cv()
svc = pb.pipeline

# Build Random Forest Pipeline
clf = RandomForestClassifier(random_state=5)
param_range = [1,2,3,4,5]
params_rf = [{'clf__criterion': ['gini', 'entropy'],
		      'clf__min_samples_leaf': param_range,
		      'clf__max_depth': param_range,
		      'clf__min_samples_split': param_range[1:]}]
pb.set_classifier(classifier=clf, params=params_rf)
pb.build_gridsearch_cv()
rf = pb.pipeline
```

+++ {"tags": ["hide-input"]}

#### Calcification Case Multivariate Analysis
##### Get Model Data

```{code-cell}
:tags: [hide-input]

X_train, y_train, X_test, y_test = cases.get_calc_model_data()
X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)
```

+++ {"tags": ["hide-input"]}

##### Build and Execute Model Selection

```{code-cell}
:tags: [hide-input]

BEST_CALC_MODEL_FP = os.path.abspath('model/eda/best_calc_pipeline.pkl')
```

```{code-cell}
:tags: [hide-input]

# Build Model Selector
calc_ms = ModelSelector(filepath=BEST_CALC_MODEL_FP)
calc_ms.add_pipeline(pipeline=lr, name="Logistic Regression")
calc_ms.add_pipeline(pipeline=svc, name="Support Vector Classifier")
calc_ms.add_pipeline(pipeline=rf, name="Random Forest")
```

+++ {"tags": ["hide-input"]}

##### Prediction Results

```{code-cell}
:tags: [hide-input]

calc_ms.run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, force=FORCE_MODEL_FIT)
y_pred = calc_ms.predict(X=X)
calc_ms.score(y_true=y, y_pred=y_pred)
```

+++ {"tags": ["hide-input"]}

Logistic Regression outperformed Support Vector Machine and Random Forest on the calcification dataset. It achieved an overall accuracy of 0.72. The precision (positive class), or the ability of the classifier not to classify false negatives was 0.58. The ability of the classifier to find all positive (malignant) cases was 0.79. The weighted harmonic mean of precision and recall was 0.67.

+++ {"tags": ["hide-input"]}

##### Calcification Variable Importance

+++ {"tags": ["hide-input"]}

The coefficients of the model indicate the importance of the feature to the prediction task. A positive value provides evidence that the instance being classified belongs to the positive or malignant class; whereas, a negative value provides evidence that the instance being classified belongs to the negative or benign class. Let's take a look.

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,8))
title = f"CBIS-DDSM Multivariate Analysis\nCalcification Cases\nFeature Importance"
calc_ms.plot_feature_importance(title=title, ax=ax)
```

+++ {"tags": ["hide-input"]}

We see that calcification types have the greatest impact on the classification task. Further, the feature weight of evidence aligns with current literature on the differential of breast calcifications. For instance, pleomorphic, fine linear, and fine linear branching calcifications are understood to have a higher probability of malignancy. In contrast, lucent centered, and vascular calcifications are typically classified as BI-RADS 1 or 2.

+++ {"tags": ["hide-input"]}

#### Mass Case Multivariate Analysis
##### Get Model Data

```{code-cell}
:tags: [hide-input]

X_train, y_train, X_test, y_test = cases.get_mass_model_data()
X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)
```

+++ {"tags": ["hide-input"]}

##### Build and Execute Model Selection

```{code-cell}
:tags: [hide-input]

BEST_MASS_MODEL_FP = os.path.abspath('model/eda/best_mass_pipeline.pkl')
```

```{code-cell}
:tags: [hide-input]

# Build Model Selector
mass_ms = ModelSelector(filepath=BEST_MASS_MODEL_FP)
mass_ms.add_pipeline(pipeline=lr, name="Logistic Regression")
mass_ms.add_pipeline(pipeline=svc, name="Support Vector Classifier")
mass_ms.add_pipeline(pipeline=rf, name="Random Forest")
```

+++ {"tags": ["hide-input"]}

##### Prediction Results

```{code-cell}
:tags: [hide-input]

mass_ms.run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, force=FORCE_MODEL_FIT)
y_pred = mass_ms.predict(X=X)
mass_ms.score(y_true=y, y_pred=y_pred)
```

+++ {"tags": ["hide-input"]}

Random Forest performed the best on the mass data. It achieved an accuracy of 0.80. Precision and recall were 0.77, and 0.78 respectively.

+++ {"tags": ["hide-input"]}

##### Mass Variable Importance

```{code-cell}
:tags: [remove-output, hide-input]

fig, ax = plt.subplots(figsize=(12,12))
title = f"CBIS-DDSM Multivariate Analysis\nMass Cases\nFeature Importance"
mass_ms.plot_feature_importance(title=title, ax=ax)
```

+++ {"tags": ["hide-input"]}

Random Forest performed best of the three models on the mass data. Unlike Logistic Regression, Random Forest has no coefficients to provide weight of evidence for (or against) the positive class. Rather, Random Forest computes a feature's non-negative importance according to its ability to increase the purity of the leaves in each tree. Pure leaves have data points from only one class. Therefore, we would expect the most important features to be strongly associated with a benign or malignant finding. Let's examine a few of the top features.

Spiculated masses are characterized by lines of varying length and thickness radiating from the margins of the mass and are considered very suspicious for malignancy {cite}`braggOncologicImaging2002`. Circumscribed masses have margins with a sharp demarcation between the lesion and surrounding tissue {cite}`princeMultipleCircumscribedMasses2018`. Most circumscribed masses are benign. Irregular masses, those having margins that are neither round nor oval, tend to imply a more suspicious finding {cite}`princeMultipleIrregularMasses2018`.  An obscured mass is a mass with greater than 25% of its margin hidden by surrounding fibro glandular tissue on the mammography; hence, it cannot be fully assessed. {cite}`elezabyObscuredMass2018`. In practice; however, this term is commonly used when the portion of the margin that is visualized is circumscribed, implying a lower likelihood of malignancy.

+++ {"tags": ["hide-input"]}

## Summary

The purpose of this section was to analyze and explore the CBIS-DDSM metadata.
