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

# EDA Introduction

In this section, we conduct an exploratory data analysis of the CBIS-DDSM dataset with several goals in mind:

1.	Maximize insight into the data and the factors that influence screening results in the CBIS-DDSM dataset.
2.	Assess the quality of the digital mammography in the CBIS-DDSM dataset.
3.	Select optimal methods and parameters for preprocessing	tasks such as denoising and artifact removal. 

The exploratory data analysis is structured as follows:

{numref}`eda_dataset` describes the structure and organization of the CBIS-DDSM dataset. {numref}`eda0` prepares the case and DICOM data for the analysis effort. {numref}`eda1` explores the case metadata for insights into screening and diagnosis of calcification and mass abnormalities in the dataset. {numref}`eda2` examines the quality and characteristics of the CBIS-DDSM images vis-à-vis abnormality type and morphological features of calcifications and masses. Finally, {numref}`eda3` evaluates methods and optimal parameter settings for preprocessing tasks such as denoising and artifact removal.
