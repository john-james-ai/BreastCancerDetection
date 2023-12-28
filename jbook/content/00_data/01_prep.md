---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
---
(eda2)=

# Data Acquisition

Hosted by [The Cancer Imaging Archive]( https://www.cancerimagingarchive.net/collection/cbis-ddsm/), the CBIS-DDSM dataset contains approximately 163 Gb of images and several metadata files as outlined in {numref}`cbis-ddsm`-acquisition.

```{table}
:name: cbis-ddsm-acquisition
| # | Dataset                    | Filename                            | Format | Size      |
|---|----------------------------|-------------------------------------|--------|-----------|
| 1 | CBIS-DDSM Mammogram Images | <manifest_name>.tcia/CBIS-DDSM/     | DICOM  | 163.51 Gb |
| 2 | Mass Training Description  | mass_case_description_train_set.csv | CSV    | 755 Kb    |
| 3 | Calc Training Description  | calc_case_description_train_set.csv | CSV    | 904 Kb    |
| 4 | Mass Test Description      | mass_case_description_test_set.csv  | CSV    | 212 Kb    |
| 5 | Calc Test Description      | calc_case_description_test_set.csv  | CSV    | 186 Kb    |
| 6 | DICOM Metadata             | metadata.csv                        | CSV    | 2.91 Mb   |
```

