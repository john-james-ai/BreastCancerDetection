---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Deep Learning for Breast Cancer Detection: Modular, Extensible, Reproducible

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://github.com/john-james-ai/BreastCancerDetection)

```{image}
:alt: Deep Learning for Breast Cancer Detection: Modular, Extensible, Reproducible
:align: center
```

## Preface

### Intended Audience

### How to Use This Book

## Local Execution

To run the notebook locally, you will need to clone the repository, set up a virtual environment, and compile the jupyter book.

1. Clone Repository:

The code is [available on GitHub](https://github.com/john-james-ai/BreastCancerDetection) or by following the GitHub icon in the external link at the top of this page. Run the following in your terminal.

```{note}
git clone https://github.com/john-james-ai/BreastCancerDetection
```

2. Create a virtual environment.

Once the download is complete, navigate to the root folder of the repository and create an environment from the `environment.yml` using the following command.

```{note}
conda env create -f environment.yml
```
3. Compile the jupyter book:

Next, compile the jupyter book with the following command.

```{note}
jupyter-book build jbook/
```

## Data

The repository contains all metadata, raw and processed, for every image in the CBIS-DDSM. However, storing the 164 GB CBIS-DDSM *Imaging*  on GitHub was not a particularly serviceable option, so an experimental, 'mini' CBIS-DDSM was created from a 5% stratified random sample of the *full* CBIS-DDSM dataset.

All imaging analyses and exploration were conducted on this experimental dataset, which is also provided in the repository. {numref}`repo_data` outlines the file structure.

```{table} File Organization
: name: repo_data
| Type     | Stage        | Description                                      | Location                                   | Size   |
|----------|--------------|--------------------------------------------------|--------------------------------------------|--------|
| Metadata | Raw          |  Calcification Case Description   Training Data  |  data/calc_case_description_train_set.csv  | 904 KB |
|          |              |  Calcification Case Description   Test Data      |  data/calc_case_description_test_set.csv   | 186 KB |
|          |              |  Mass Case Description Training   Data           |  data/mass_case_description_train_set.csv  | 755 KB |
|          |              |  Mass Case Description Test   Data               |  data/mass_case_description_test_set.csv   | 212 KB |
|          | Final        | CBIS-DDSM Metadata Dataset                       | data/cbis_ddsm.csv                         | 2 MB   |
| Imaging  | Experimental | CBIS-DDSM Imaging Dataset                        | data/1_exp/CBIS-DDSM/                      | 8.2 GB |
```

That said, for those with larger appetites (and an extra 163 GB of available storage capacity), the CBIS-DDSM can be downloaded from [The Cancer Imaging Archive]( https://www.cancerimagingarchive.net/collection/cbis-ddsm/)(TCIA).  The images are downloaded using the [NBIA Data Retriever]( https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide) which must be installed on your local machine.

Please follow the instructions [here](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). Again, the CBIS-DDSM imaging data weighs in at over 163 GB, so download times may be considerable depending on your available internet bandwidth and throttling at TCIA.

```{admonition} File Organization FileManageruration
The file locations are kept in a configuration file, `config.yml`, which is stored in the root directory of the repository. Once you've downloaded the data, check the configuration file to ensure that the file locations on your machine are correctly reflected in the configuration file.
```

## Document Organization

## Acknowledgements

## LIcense

The code and prose in this writing are released under the **MIT License**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The following copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

If you wish to cite this writing, you may use the following:

```{note}
@book{jamesDeepLearningBreast2024,
edition = {1.0},
title = {Deep Learning for Breast Cancer Detection: Modular, Extensible, Reproducible},
url = {https://github.com/john-james-ai/BreastCancerDetection},
author = {James, John},
year = {2024},
}
```

## Dependency Version and Hardware
The essential dependencies, versions, and hardware details are listed below.

```{code-cell} ipython3
%load_ext watermark
%watermark -v -m -p cv2,ipython,ipykernel,joblib,jupyter-book,logmatic,matplotlib,numpy,pandas,python,python-dotenv,pyyaml,requests,scipy,seaborn,skimage,sklearn,studioai,yaml
```
