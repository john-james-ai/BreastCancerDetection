#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Learning for Breast Cancer Detection                                           #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.12                                                                             #
# Filename   : /bcd/data/prep/image/clean.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/BreastCancerDetection                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 16th 2023 08:47:53 pm                                                #
# Modified   : Wednesday October 18th 2023 06:58:46 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module responsible for Image Cleaning Tasks"""
# Standard library imports
import time

# Third-party library imports
import numpy as np
import cv2
from tqdm.notebook import tqdm, trange
from joblib import Parallel, delayed
from skimage.draw import polygon

# Local library imports
from bcd.data.prep.image.base import DICOMTask


# ------------------------------------------------------------------------------------------------ #
class DICOMArtifactRemover(DICOMTask):
    """Performs artifact removal via binary masking and largest contour detection

    Args:
        source_dir (str): Directory containing source images
        destination_dir (str): Directory into which the new images will be stored.
        image_size (tuple): Size of image in pixels. Default = (256,256).
        format (str): Image file format. Default = 'png'
        threshold (int): Global threshold for binarization. Default = 20
        otsu (bool): Indicates whether otsu thresholding will be performed.

    """

    __source_dir = "data/raw/"
    __destination_dir = "data/staged/"

    def __init__(
        self,
        source_dir: str = None,
        destination_dir: str = None,
        image_size: tuple = (256, 256),
        format: str = "png",
        threshold: int = 20,
        otsu: bool = False,
    ) -> None:
        super().__init__()
        self._source_dir = source_dir or self.__source_dir
        self._destination_dir = destination_dir or self.__destination_dir
        self._image_size = image_size
        self._format = format
        self._threshold = threshold
        self._otsu = otsu

    def reset(self) -> None:
        super().reset(self._destination_dir)

    def run(
        self,
        n: int = None,
        fileset: str = "train",
        parallel: bool = True,
        n_jobs: int = 6,
        save: bool = True,
    ) -> None:
        """Performs artifact removal on select images.

        Args:
            n (int): Number of paths to return. If None, all paths matching criteria will be returned.
            fileset (str): Either 'train', or 'test'. Default = 'train'
            parallel (bool): Whether to run the task in parallel. Default = True
            n_jobs (int): Number of jobs to run if parallel is True. Default = 6.
            save (bool): Indicates whether images should be saved.

        """
        start = time.time()

        filepaths = self.get_paths(n=n, fileset=fileset)

        if parallel:
            Parallel(n_jobs=n_jobs)(
                delayed(self.process_image)(filepath, save)
                for filepath in tqdm(filepaths, total=len(filepaths))
            )
        else:
            for i in trange(len(filepaths)):
                self.process_image(filepaths[i], save=save)
        print("Artifact Removal Complete!")
        print("Time =", np.around(time.time() - start, 3), "sec")

    def process_image(self, filepath: str, save: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Masks artifacts from the image

        Args:
            filepath (str): Filepath for the image.

        Returns:
            Tuple of (output image, contour)

        """
        img = self.read_image(filepath=filepath)
        img = self._right_orient_mammogram(img=img)
        img_gray = self._to_grayscale(img=img)
        img_bin = self._binarize(img=img_gray)
        img_contour = self._extract_contour(img=img_bin)
        img_output = self._erase_background(img=img_gray, contour=img_contour)
        img_output = self._crop(img=img_output, contour=img_contour)
        img_output = cv2.resize(img_output, self._image_size)
        if save:
            save_filepath = self.get_path(
                source_filepath=filepath,
                source_dir=self._source_dir,
                destination_dir=self._destination_dir,
                format=self._format,
            )
            self.save_image(img=img_output, filepath=save_filepath)
        return (img_output, img_contour)

    def visualize(self, n: int = 8) -> None:
        """Visualize samples of images processed.

        Args:
            n (int): The number of images to visualize
        """
        super().visualize(directory=self._destination_dir, n=n)

    @staticmethod
    def max_pix_val(dtype):
        """Returns the maximum pixel value given the dtype"""
        if dtype == np.dtype("uint8"):
            maxval = 2**8 - 1
        elif dtype == np.dtype("uint16"):
            maxval = 2**16 - 1
        else:
            raise Exception("Unknown dtype found in input image array")
        return maxval

    def _right_orient_mammogram(self, img: np.ndarray) -> np.ndarray:
        """Orient the image to the right"""
        left_nonzero = cv2.countNonZero(img[:, 0 : int(img.shape[1] / 2)])  # noqa
        right_nonzero = cv2.countNonZero(img[:, int(img.shape[1] / 2) :])  # noqa
        if left_nonzero < right_nonzero:
            img = cv2.flip(img, 1)
        return img

    def _to_grayscale(self, img: np.array) -> np.array:
        # Convert to float to avoid overflow or underflow.
        img = img.astype(float)
        # Rescale to gray scale values between 0-255
        img_gray = (img - img.min()) / (img.max() - img.min()) * 255.0
        # Convert to uint
        img_gray = np.uint8(img_gray)
        return img_gray

    def _binarize(self, img: np.array) -> np.array:
        if self._otsu:
            _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _, img_bin = cv2.threshold(
                img, thresh=self._threshold, maxval=255, type=cv2.THRESH_BINARY
            )
        return img_bin

    def _extract_contour(self, img: np.array) -> np.array:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        return contour

    def _erase_background(self, img: np.array, contour: np.array) -> np.array:
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        output = cv2.bitwise_and(img, mask)
        return output

    def _crop(self, img: np.array, contour: np.array):
        x1, x2 = np.min(contour[:, :, 0]), np.max(contour[:, :, 0])
        y1, y2 = np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
        x1, x2 = int(0.99 * x1), int(1.01 * x2)
        y1, y2 = int(0.99 * y1), int(1.01 * y2)
        return img[y1:y2, x1:x2]


# ------------------------------------------------------------------------------------------------ #
class DICOMPectoralRemover(DICOMTask):
    """Removes the pectoral muscle region from an image.

    Args:
        source_dir (str): Directory containing source images
        destination_dir (str): Directory into which the new images will be stored.
        format (str): Image file format. Default = 'png'
    """

    __source_dir = "data/staged/"
    __destination_dir = "data/clean/"

    def __init__(
        self,
        source_dir: str = None,
        destination_dir: str = None,
        format: str = "png",
    ) -> None:
        super().__init__()
        self._source_dir = source_dir or self.__source_dir
        self._destination_dir = destination_dir or self.__destination_dir
        self._format = format

    def reset(self) -> None:
        super().reset(self._destination_dir)

    def run(
        self,
        n: int = None,
        fileset: str = "train",
        parallel: bool = True,
        n_jobs: int = 6,
        save: bool = True,
    ) -> None:
        """Performs artifact removal on select images.

        Args:
            n (int): Number of paths to return. If None, all paths matching criteria will be returned.
            fileset (str): Either 'train', or 'test'. Default = 'train'
            parallel (bool): Whether to run the task in parallel. Default = True
            n_jobs (int): Number of jobs to run if parallel is True. Default = 6.
            save (bool): Indicates whether images should be saved.

        """
        start = time.time()

        filepaths = self.get_paths(n=n, directory=self._source_dir, fileset=fileset)

        if parallel:
            Parallel(n_jobs=n_jobs)(
                delayed(self.process_image)(filepath, save)
                for filepath in tqdm(filepaths, total=len(filepaths))
            )
        else:
            for i in trange(len(filepaths)):
                self.process_image(filepaths[i], save=save)
        print("Pectoral Muscle Removal Complete!")
        print("Time =", np.around(time.time() - start, 3), "sec")

    def process_image(self, filepath: str, save: bool = False) -> np.ndarray:
        """Removes pectoral muscle from the image

        Args:
            filepath (str): Filepath for the image.

        Returns:
            Image in 2D Numpy array format.

        """
        image = self.read_image(filepath=filepath)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        edges = self._auto_canny(image=image)
        lines = self._extract_hough_lines(edges=edges)
        lines = self._shortlist_lines(lines=lines)
        image = self._remove_pectoral(image=image, lines=lines)
        if save:
            save_filepath = self.get_path(
                source_filepath=filepath,
                source_dir=self._source_dir,
                destination_dir=self._destination_dir,
                format=self._format,
            )
            self.save_image(img=image, filepath=save_filepath)
        return image

    def _auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        # return the edged image
        return edged

    def _shortlist_lines(self, lines):
        MIN_ANGLE = 10
        MAX_ANGLE = 70
        MIN_DIST = 5
        MAX_DIST = 256
        shortlisted_lines = [
            x
            for x in lines
            if (x["rho"] >= MIN_DIST)
            & (x["rho"] <= MAX_DIST)
            & (x["theta"] >= MIN_ANGLE)
            & (x["theta"] <= MAX_ANGLE)
        ]
        return shortlisted_lines

    def _extract_hough_lines(self, edges: np.ndarray) -> list:
        lines = []
        hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        for line in hough_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))
            lines.append(
                {"rho": rho, "theta": np.degrees(theta), "point1": [x1, y1], "point2": [x2, y2]}
            )
        return lines

    def _remove_pectoral(self, image: np.ndarray, lines: list) -> np.ndarray:
        if len(lines) > 0:
            lines.sort(key=lambda x: x["rho"])
            pectoral_line = lines[0]
            d = pectoral_line["rho"]
            theta = np.radians(pectoral_line["theta"])
            x_intercept = d / np.cos(theta)
            y_intercept = d / np.sin(theta)
            rr, cc = polygon([0, 0, y_intercept], [0, x_intercept, 0])
            image[rr, cc] = 0
        return image
