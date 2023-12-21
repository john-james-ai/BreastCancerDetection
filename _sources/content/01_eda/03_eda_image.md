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

```{code-cell} ipython3
:tags: [remove-input, remove-output]
x = ImageExplorer()
ax = x.summary()
glue("edai_summary", ax)
```

```{glue:figure} edai_summary
---
align: center
name: edai_summary_fig
---
CBIS-DDSM Dataset Summary
```
