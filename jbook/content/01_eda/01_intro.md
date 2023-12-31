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

# EDA Part 1: Introduction

In this section, we conduct an exploratory data analysis of the CBIS-DDSM dataset with several goals in mind:

1. Maximize insight into the data and the factors that influence screening results in the CBIS-DDSM dataset.
2. Assess the quality of the digital mammography in the CBIS-DDSM dataset.
3. Select optimal methods and parameters for preprocessing tasks such as denoising and artifact removal.

The exploratory data analysis will be structured as follows:

{numref}`eda2` explores the case metadata for insights into screening and diagnosis of calcification and mass abnormalities in the dataset. {numref}`eda3` examines the quality and characteristics of the CBIS-DDSM images vis-à-vis abnormality type and morphological features of calcifications and masses. Finally, {numref}`eda4` evaluates methods and optimal parameter settings for preprocessing tasks such as denoising and artifact removal.
