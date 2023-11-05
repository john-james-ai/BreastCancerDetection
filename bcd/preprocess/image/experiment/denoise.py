#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/preprocess/image/experiment/denoise.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday November 3rd 2023 10:48:10 am                                                #
# Modified   : Saturday November 4th 2023 06:05:35 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Denoise Calibration and implementation Module"""
import json
import multiprocessing
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Tuple
from uuid import uuid4

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# pylint: disable=no-name-in-module, no-member, unnecessary-lambda, invalid-name
from skimage.restoration import calibrate_denoiser, denoise_invariant, estimate_sigma
from tqdm import tqdm


# ------------------------------------------------------------------------------------------------ #
#                                    DENOISER BASE CLASS                                           #
# ------------------------------------------------------------------------------------------------ #
class Denoiser(ABC):
    """Performs Calibration of an image and returns the best J-invariant denoiser and parameters.

    Args:
        denoiser (Callable): The denoiser to be calibrated and applied.
        source (str): The directory containing input images and registry.
            Note, this directory must have a csv file called registry.csv that contains
            the metadata for the images in the directory.
        destination (str): The directory into which, results and denoised
            images will be stored.
        params (dict): Parameters to test. If not provided, they will be estimated
            from the image data.
        n_jobs (int): Number of parallel jobs to run. If None, all cpus will be used.

        Returns:
            Dictionary containing the algorithm name, the minimum loss, and the best parameters.
    """

    stage_id = 1
    stage = "Denoise"

    def __init__(
        self,
        denoiser: Callable,
        source: str,
        destination: str,
        params: dict = None,
        n_jobs: int = None,
    ) -> None:
        self._denoiser = denoiser
        self._params = params
        self._source = source
        self._destination = destination
        self._n_jobs = n_jobs or multiprocessing.cpu_count()

        self._registry_filepath = os.path.join(source, "registry.csv")
        self._results_filepath = os.path.join(destination, "results.csv")

    def run(self) -> dict:
        registry = self._get_revised_registry()

        with joblib.Parallel(n_jobs=self._n_jobs) as parallel:
            results = parallel(
                joblib.delayed(self.denoise_image)(metadata)
                for _, metadata in tqdm(registry.iterrows(), total=len(registry))
            )
        self.save_results(results=results)

    def denoise_image(self, image_metadata: pd.Series) -> None:
        """Denoises using best parameters and saves the results and denoised image."""

        # Read the image
        filepath = os.path.join(self._source, image_metadata["source_image_filepath"])
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

        # Calibrate the denoiser and obtain best parameters and minimum loss
        best_params, min_loss = self.calibrate(image=image)

        # Denoise the image using the J-Invariant version of denoiser with best parameters
        denoised_image = denoise_invariant(
            image, self._denoiser, denoiser_kwargs=best_params, channel_axis=-1
        )

        # Calculate mse, psnr, and structural similarity
        denoised_mse = mse(image, denoised_image)
        denoised_psnr = psnr(image, denoised_image)
        denoised_ssim = ssim(image, denoised_image)
        # Format results
        results = self._format_results(
            image_metadata=image_metadata,
            best_params=best_params,
            denoised_loss=min_loss,
            denoised_mse=denoised_mse,
            denoised_psnr=denoised_psnr,
            denoised_ssim=denoised_ssim,
        )
        # Save the denoised image and return results.
        self.save_image(filename=results["test_image_filepath"], image=denoised_image)

        return results

    def calibrate(self, image: np.ndarray) -> Tuple[dict, float]:
        """Calibrates the denoiser and returns a tuple containing the results

        Args:
            image (np.ndarray): Image in numpy array format.

        Returns:
            Tuple containing
                best parameters (dict): Dictionary of best parameters
                min loss (float): The minimum self-supervised loss.

        """
        params = self._params or self.get_params(image)
        image = image.astype(float)
        _, (params_tested, losses) = calibrate_denoiser(
            image, self._denoiser, denoise_parameters=params, extra_output=True
        )

        best_params = params_tested[np.argmin(losses)]
        min_loss = np.min(losses)
        return (best_params, min_loss)

    @abstractmethod
    def get_params(self, image: np.ndarray) -> dict:
        """Method that returns parameters to be tested.

        Args:
            image (np.ndarray): Image pixel data.
        """

    def _format_results(
        self,
        image_metadata: pd.Series,
        best_params: dict,
        denoised_loss: float,
        denoised_mse: float,
        denoised_psnr: float,
        denoised_ssim: float,
    ) -> dict:
        uid = str(uuid4())
        d = {
            "test_no": image_metadata["test_no"],
            "source_image_uid": image_metadata["source_image_uid"],
            "source_image_filepath": image_metadata["source_image_filepath"],
            "test_image_uid": uid,
            "test_image_filepath": uid + ".png",
            "mode": image_metadata["mode"],
            "stage_id": self.stage_id,
            "stage": self.stage,
            "image_view": image_metadata["image_view"],
            "abnormality_type": image_metadata["abnormality_type"],
            "assessment": image_metadata["assessment"],
            "cancer": image_metadata["cancer"],
            "method": self._denoiser.__name__,
            "params": json.dumps(best_params),
            "loss": denoised_loss,
            "mse": denoised_mse,
            "psnr": denoised_psnr,
            "ssim": denoised_ssim,
            "evaluated": datetime.now(),
        }
        return d

    def save_results(self, results: list) -> None:
        """Converts results to dataframe and stores in csv."""

        os.makedirs(os.path.dirname(self._results_filepath), exist_ok=True)
        results_df = pd.DataFrame(data=results)
        results_df.to_csv(
            self._results_filepath,
            mode="a",
            header=not os.path.exists(self._results_filepath),
            index=False,
        )

    def save_image(self, filename: str, image: np.ndarray) -> None:
        """Save the image in png format."""
        filepath = os.path.join(self._destination, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filename=filepath, img=image)

    def _get_revised_registry(self) -> pd.DataFrame:
        """Conditions the registry to comport with evaluation dataset."""
        registry = pd.read_csv(self._registry_filepath)
        # Add image number
        registry.reset_index(inplace=True, names="test_no")
        # Change uid to source_uid
        registry.rename(columns={"uid": "source_image_uid"}, inplace=True)
        # Add filename
        registry["source_image_filepath"] = registry["filepath"].apply(
            lambda x: os.path.basename(x)
        )
        return registry


# ------------------------------------------------------------------------------------------------ #
#                                        BILATERAL                                                 #
# ------------------------------------------------------------------------------------------------ #
class BilateralDenoiser(Denoiser):
    """Performs Bilateral Denoising"""

    def get_params(self, image: np.ndarray) -> dict:
        size = np.sqrt(np.square(image.shape[0]) + np.square(image.shape[1]))
        # Rule of thumb for sigma domain is 2% of the image diagonal
        theta_domain = (size * 0.02) * np.arange(start=0.6, stop=1.2, step=0.2)
        # Rule of thumb for sigma range is median (or mean) of the image gradients.
        # To compute the gradients, we'll use a kernel size of 3
        scale = 1
        delta = 0
        ddepth = cv2.CV_32F
        ksize = 3
        gX = cv2.Sobel(
            image,
            ddepth=ddepth,
            dx=1,
            dy=0,
            ksize=ksize,
            scale=scale,
            delta=delta,
            borderType=cv2.BORDER_DEFAULT,
        )
        gY = cv2.Sobel(
            image,
            ddepth=ddepth,
            dx=0,
            dy=1,
            ksize=ksize,
            scale=scale,
            delta=delta,
            borderType=cv2.BORDER_DEFAULT,
        )
        g = np.concatenate((gX, gY))
        g_median = np.median(g)
        theta_range = g_median * np.arange(0.4, 1.2, 0.2)
        params = {"sigma_color": theta_range, "sigma_spatial": theta_domain}
        return params


# ------------------------------------------------------------------------------------------------ #
#                                    TOTAL VARIATION                                               #
# ------------------------------------------------------------------------------------------------ #
class TotalVariationDenoisier(Denoiser):
    """Performs Total Variation Denoising using the split Bregman algorithm"""

    def get_params(self, image: np.ndarray) -> dict:
        return {
            "weight": np.arange(3, 7, 0.5),
            "max_num_iter": 100,
            "eps": 0.001,
            "isotropic": True,
        }


# ------------------------------------------------------------------------------------------------ #
#                                       WAVELET                                                    #
# ------------------------------------------------------------------------------------------------ #
class WaveletDenoisier(Denoiser):
    """Performs Wavelet Denoising of an Image"""

    def get_params(self, image: np.ndarray) -> dict:
        # Sigma estimated from the data.
        return {
            "sigma": None,
            "wavelet": np.array(["db1", "db2", "haar", "sym9"]),
            "mode": np.array(["soft", "hard"]),
            "method": np.array(["BayesShrink", "VisuShrink"]),
        }


# ------------------------------------------------------------------------------------------------ #
#                                   NON LOCAL MEANS                                                #
# ------------------------------------------------------------------------------------------------ #
class NLMDenoisier(Denoiser):
    """Performs Non-Local Means Denoising of an Image"""

    def get_params(self, image: np.ndarray) -> dict:
        # Sigma estimated from the data.
        sigma = estimate_sigma(image)
        return {"sigma": np.arange(0.6, 1.4, 0.2) * sigma, "h": np.arange(0.6, 1.2, 0.2) * sigma}
