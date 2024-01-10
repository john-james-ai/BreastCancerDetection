---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
---

(eda00)=

# **CBIS-DDSM** Dataset

Developed in 1997 through a grant from the DOD Breast Cancer Research Program and the US Army Research and Material Command, the *original* Digital Database for Screening Mammography (DDSM) {cite}`USFDigitalMammography` is a collection of 2620 cases obtained by patient consent from Massachusetts General Hospital, Wake Forest University School of Medicine, Sacred Heart Hospital, and Washington University of St. Louis School of Medicine. Its cases are annotated with ROIs for calcifications and masses, they include Breast Imaging Reporting and Data System (BI-RADS) descriptors for mass shape, mass margin, calcification type, calcification distribution, and breast density. They also include overall BI-RADS assessments from 0 to 5 and ratings of the subtlety of the abnormalities from 1 to 5.

The DDSM, a powerful and extensively used resource for imaging research, presented certain challenges that limited its accessibility and utility.  For instance, the original DDSM was saved in lossless JPEG files (LJPEG); a non-standard, and obsolete compression format. Regions of interest marked the general area of legions; but, lacked the specificity of precise image segmentation, requiring researchers to implement segmentation algorithms for accurate feature extraction. Some annotations marked legions that could not be seen. Further, the metadata processing and image correction routines distributed with the dataset were a set of bash and C routines that were difficult to refactor.

This Curated Breast Imaging Subset of DDSM (CBIS-DDSM) {cite}`leeCuratedMammographyData2017` is an updated and standardized version of the original DDSM, developed to address some of the challenges. In particular, the questionable annotations were reimage_viewed by a trained mammographer, and 254 images were removed from the dataset. The images were converted from PVRG-JPEG to 8-bit raw binary bitmaps. Python tools were developed to convert the 8-bit images to 16-bit grayscale TIFF files, which were later converted to the Digital Imaging and Communications in Medicine (DICOM) format. The bash and C preprocessing tools were re-implemented in Python to be cross-platform. Convenience images were cropped around the region of interest for researchers analyzing only the abnormalities. Precise segmentation was applied to the calcification images providing much more accurate regions of interest, and the data were split into training and test sets, based on the BI-RADS category to support method evaluation and reproducibility.

The CBIS-DDSM is comprised of:

- **Mammographic Images**: Full mammography, mammogram and cropped images {numref}`cbis_ddsm_images` ,
- **Case Data**: Calcification and mass case data {numref}`cbis_ddsm_cases`
- **Series Data**: Series, study and subject identifiers and image file locations.
- **DICOM Metadata**: Image and modality information in DICOM imaging format.

Next, we summarize the dataset statistics and review the data elements in each dataset.

## Dataset Statistics

In total, the CBIS-DDSM has 10,239 images of 1,644 cases for 1,566 patients.

### Imaging Dataset Statistics

```{table} CBIS-DDSM Images
:name: cbis_ddsm_images

| Collection Statistics  |        |
| ---------------------- | ------ |
| Image Size (GB)        | 163.6  |
| Modalities             | MG     |
| Number of Images       | 10239  |
| Number of Participants | 1,566  |
| Number of Series       | 6775   |
| Number of Studies      | 6775   |
```

### Case Dataset Statistics

```{table} CBIS-DDSM Dataset Statistics
:name: cbis_ddsm_cases
| # | Dataset                   | Filename                            | Format | Size   | Cases | Benign | Benign Abnormalities | Malignant | Malignant Abnormalities |
|---|---------------------------|-------------------------------------|--------|--------|-------|--------|----------------------|-----------|-------------------------|
| 1 | Calc Training Description | calc_case_description_train_set.csv | CSV    | 904 Kb | 602   | 329    | 552                  | 273       | 304                     |
| 2 | Calc Test Description     | calc_case_description_test_set.csv  | CSV    | 186 Kb | 151   | 85     | 112                  | 66        | 77                      |
| 3 | Mass Training Description | mass_case_description_train_set.csv | CSV    | 755 Kb | 691   | 355    | 387                  | 336       | 361                     |
| 4 | Mass Test Description     | mass_case_description_test_set.csv  | CSV    | 212 Kb | 200   | 117    | 135                  | 83        | 87                      |
|   | Total                     |                                     |        |        | 1644  | 886    | 1186                 | 758       | 829                     |
```

Of the 1,644 cases, 54% are benign, and 46% are malignant.

## Dataset Dictionary

### Case Data

Cases, defined as a particular abnormality seen on a cranial caudal (CC) or mediolateral oblique (MLO) image_view, are organized by BI-RADS categories: calcification and mass. Calcification and mass cases were further split into training (80%) and test (20%) sets of various levels of difficulty to support consistent evaluation of computer-aided diagnostics (CADx) systems. Each case description file contains the following variables:

- Patient ID
- Density category
- Breast: Left or Right
- View: CC or MLO
- Number of abnormality from the image.
- Mass shape (when applicable)
- Mass margin (when applicable)
- Calcification type (when applicable)
- Calcification distribution (when applicable)
- BI-RADS assessment
- Pathology: Benign, Benign without call-back, or Malignant
- Subtlety rating: Radiologists’ rating of difficulty in image_viewing the abnormality in the image
- Path to image files

### Series Metadata

A series encapsulates full mammography, ROI mask, or cropped images for a patient and image image_view. The series metadata file, 'metadata.csv', maps series metadata to DICOM image file locations and contains the following attributes:

- Series, subject, and study identifiers
- Data description URL
- Study dates and download timestamps
- Series description, i.e. full mammogram image, ROI mask, or cropped image
- SOP Class name and UID
- The file location and size
- The number of images at the file location

### DICOM Image Data

Each DICOM image file comports with the DICOM standard and contains additional metadata describing the modality, the imaging session and the image.

- SOP Class UID
- SOP Instance UID
- Content Date
- Content Time
- Modality (MG)
- Patient ID
- Body Part Examined
- Study Instance UID
- Series Instance UID
- Series Number
- Instance Number
- Patient Orientation (CC or MLO)
- Rows
- Columns
- Bits Allocated
- Bits Stored
- Smallest Image Pixel Value
- Largest Image Pixel Value

## Dataset Unboxing

Before we launch into a data quality analysis (DQA), let's 'unbox' the case and series datasets, just in case there are aspects to address before DQA.
