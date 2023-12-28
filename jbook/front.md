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
To run the notebook locally, you will need to clone the repository, set up a virtual environment, download the data, create a configuration file containing file locations and directories, and build the jupyter book.

1. Clone Repository:

The code is [available on GitHub](https://github.com/john-james-ai/BreastCancerDetection) or by following the GitHub icon in the external link at the top of this page. To run this ebook and its notebooks on your computer, please follow the following steps.

```{note}
git clone https://github.com/john-james-ai/BreastCancerDetection
```
2. Create virtual environment.

Navigate to the root folder of the repository and create an environment from the `environment.yml` using the following command.

```{note}
conda env create -f environment.yml
```

3. Download the CBIS-DDSM

The CBIS-DDSM can be downloaded from [The Cancer Imaging Archive]( https://www.cancerimagingarchive.net/collection/cbis-ddsm/).

- Download the case data and the CBIS-DDSM manifest file into your data directory.
- Then, navigate to [Downloading TCIA Images](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images) and follow the instructions for downloading and installing the NBIA Data Retriever.
- Once you have the data retriever, navigate to the [NBIA Data Retriever Command-Line Interface Guide]( https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide) and follow the instructions for downloading the mammography using the command line. The CBIS-DDSM imaging data comprises over 163 GB, so download times may be considerable depending on your available internet bandwidth.

3. Configuration File

Run the following code cell
4. Compile the book:

```{note}
jupyter-book build jbook/
```

5. Download the CBIS-DDSM dataset and metadata below from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/cbis-ddsm/). Instructions for downloading the dataset can be found at [Downloading TCIA Images](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images). It comprises over 163 GB of data, so downloading the CBIS-DDSM mammography may take a considerable amount of time, depending on your available internet bandwidth.

When complete, you should have a directory CBIS-DDSM-ALL-doi<some_text>.tcia/CBIS-DDSM/ containing all the mammography. Case and DICOM metadata {numref}`cbis-ddsm`(2-6) should also be in your data directory. Feel free to rename or move folders to suit your workflow.

```{table}
:name: cbis-ddsm
| # | Dataset                    | Filename                            | Format | Size      |
|---|----------------------------|-------------------------------------|--------|-----------|
| 1 | CBIS-DDSM Mammogram Images | <manifest_name>.tcia/CBIS-DDSM/     | DICOM  | 163.51 GB |
| 2 | Mass Training Description  | mass_case_description_train_set.csv | CSV    | 755 KB    |
| 3 | Calc Training Description  | calc_case_description_train_set.csv | CSV    | 904 KB    |
| 4 | Mass Test Description      | mass_case_description_test_set.csv  | CSV    | 212 KB    |
| 5 | Calc Test Description      | calc_case_description_test_set.csv  | CSV    | 186 KB    |
| 6 | DICOM Metadata             | metadata.csv                        | CSV    | 2.91 MB   |
```

## Document Organization

## Acknowledgements

## LIcense

The code and prose in this writing are released under the **MIT License**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

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

```{code-cell}
%load_ext watermark
%watermark -v -m -p ipywidgets,matplotlib,numpy,pandas,cv2,skimage,sklearn,seaborn,studioai
```
