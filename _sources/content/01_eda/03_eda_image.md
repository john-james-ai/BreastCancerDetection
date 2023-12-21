---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: bcd
  language: python
  name: python3
---

# Exploratory Data Analysis of Images

Image quality factors such as resolution, the presence of noise and artifacts, contrast, and size affect the visual information contained in mammography, and crucially determine the performance of deep learning classification models.

In this section, we conduct an exploratory data analysis of the CBIS-DDSM full mammogram imaging data to assess data quality and to characterize and parameterize image preprocessing tasks such as denoising, artifact and pectoral muscle removal, contrast enhancement, and data augmentation.

```{code-cell} ipython3
:tags: [remove-input]

import os
if 'jbook' in os.getcwd():
    os.chdir(os.path.abspath(os.path.join("../../..")))

from myst_nb import glue
import warnings
from bcd.explore.image.eda_image import ImageExplorer
warnings.filterwarnings("ignore")
```

## Summary

The CBIS-DDSM contains a total of 3565 full mammography images, not counting a case for which the DICOM file was corrupt {numref}`edai_summary_fig`.

```{code-cell} ipython3
:tags: [remove-input, remove-output]
x = ImageExplorer()
ax, stats = x.summary()
glue("edai_summary_ax", ax)
glue("edai_summary_stats", stats)
```

```{glue:figure} edai_summary_ax
---
align: center
name: edai_summary_fig
---
**CBIS-DDSM Dataset Summary**: A total of 3565 total mammography images with a 80/20 train test split.
```

As indicated in {numref}`edai_summary_stats_fig`, approximately 40% of the cases were malignant.

```{glue:figure} edai_summary_stats
---
align: center
name: edai_summary_stats_fig
---
CBIS-DDSM Dataset Stats
```

Class imbalance can lead to biased predictions. Augmenting the malignant class with various transformations will mitigate bias induced by class imbalance.
