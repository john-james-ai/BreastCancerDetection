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
(eda61)=

# Denoise

What is noise? Somewhat imprecisely, we might say that noise is any variation in brightness information not part of the original image. Yet, all biomedical images are imperfect representations of the underlying structure that is being imaged. Limited resolution (defined by the optics), uneven illumination or background, out-of-focus light, artifacts, and, of course, image noise, contribute to this imperfection. For denoising, we distinguish noise from other imperfections, with the following definition:

> Noise is the discrepancy between the true amount of light $s_i$ being measured at pixel $i$, and the corresponding measured pixel value $x_i$.

In the following subsections, we expand on this definition with a review of the most common noise models in digital mammograph. Then, we describe the predominate types of noises encountered in digital mammography.  Next, we introduce the most frequently applied denoising methods in low-dose X-ray mammography. Finally, we conduct an evaluation to determine the denoiser that produces the best results.
