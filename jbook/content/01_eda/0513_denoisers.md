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
# Denoiser Methods

Having defined an image $f(x,y)$ as a function of an original image $s(x,y)$ and a noise component $n(x,y)$ ({eq}`additive_noise_model`, and {eq}`multiplicative_noise_model`), we can define the denoising task as follows:

> Image denoising aims to provide a function $d(x) \approx s$ that takes a noisy image $f$ as input and returns an approximation of the true clean image $\hat{s}$ as output.

Our objective is to create an approximation of the original image, $\hat{s}$ in which:

- flat areas are smooth,
- edges are protected without blurring
- textures are preserved, and
- no new artifacts are generated.

Over the past two decades, a considerable body of research has been devoted to the design, development, and testing of denoising methods for biomedical imaging. While a systematic review of the denoising landscape is well beyond the scope of this effort, we will introduce a few of the most commonly used biomedical image denoising methods, with a focus on applications in mammography. Specifically, we will be evaluating the following five methods:

In general, the denoiser method space is comprised of spatial domain methods and transform domain methods, which we will introduce in the next couple of subsections.
