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

```{code-cell} ipython3
import warnings
from bcd.explore.image.eda_image import ImageExplorer
warnings.filterwarnings("ignore")
```



```{code-cell} ipython3
x = ImageExplorer()
_ = x.summary()
```
