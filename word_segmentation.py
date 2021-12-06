import json
from collections import defaultdict
from typing import List, Dict, Tuple

import cv2
import numpy as np
from path import Path
from sklearn.cluster import DBSCAN

from entities import Word, BoundingBox
from utils import get_image_files, visualize_result, get_image_name


class WordSegmentation:
    def __init__(self,
                 kernel_size: int = 25,
                 sigma: float = 11,
                 theta: float = 5,
                 min_word_area: int = 100,
                 resize_height: int = 1000):
        """
        Constructor
        :param img: Original image
        :param kernel_size: Size of Gaussian kernel (must be an odd number)
        :param sigma: Standard deviation of Gaussian filter (low-pass filter) kernel
        :param theta: Ratio between width / height of words
        :param min_word_area: Minimum allowed area of the bounding box to be considered as a word
        :param resize_height: Resize height
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.theta = theta
        self.min_word_area = min_word_area
        self.resize_height = resize_height

    def load_params(self, file_name):
        with open(f'./models/{file_name}.txt', 'r') as f:
            data = json.load(f)
        self.kernel_size = data['kernel_size']
        self.sigma = data['sigma']
        self.theta = data['theta']
        self.min_word_area = data['min_word_area']
        self.resize_height = data['resize_height']

    def preprocessing(self,
                      img: np.ndarray,
                      resize_height: int) -> Tuple[np.ndarray, float]:
        """
        Convert image to grayscale and resize
        :param img: original image
        :param resize_height: resize height
        :return: pre-processed image
        """
        assert img.ndim in (2, 3)
        assert img.dtype == np.uint8
        assert resize_height > 0
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        current_size = img.shape[0]
        scale_factor = resize_height / current_size
        return cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor), 1 / scale_factor

    @staticmethod
    def create_gaussian_kernel(kernel_size: int,
                               sigma: float,
                               theta: float) -> np.ndarray:
        """
        Prepare Gaussian kernel
        :param kernel_size: Size of Gaussian kernel (must be an odd number)
        :param sigma: Standard deviation of Gaussian filter (low-pass filter) kernel
        :param theta: Ratio between width / height of words
        :return: 2D Gaussian kernel
        """
        # Check odd size kernel
        assert kernel_size % 2

        # Create 2D
        half_kernel_size = kernel_size // 2
        temp_x = temp_y = np.linspace(-half_kernel_size, half_kernel_size, kernel_size)
        x, y = np.meshgrid(temp_x, temp_y)

        # Compute sigma values
        sigma_x = sigma * theta
        sigma_y = sigma

        # Compute x, y and exp matrix and combine them
        exp_matrix = np.exp(-x ** 2 / (2 * sigma_x) - y ** 2 / (2 * sigma_y))
        x_matrix = (x ** 2 - sigma_x ** 2) / (2 * np.math.pi * sigma_x ** 5 * sigma_y)
        y_matrix = (y ** 2 - sigma_y ** 2) / (2 * np.math.pi * sigma_y ** 5 * sigma_x)
        kernel = (x_matrix + y_matrix) * exp_matrix
        kernel = kernel / np.sum(kernel)

        return kernel

    @staticmethod
    def sort_line(words: List[Word]) -> List[Word]:
        """
        Sort words by x-coordinate
        :param words: List of words
        :return: List of sorted words
        """
        return sorted(words, key=lambda word: word.box.x + word.box.w / 2)

    @staticmethod
    def segment_lines(words: List[Word],
                      max_distance: float = 0.7,
                      min_words_per_line: int = 2) -> List[List[Word]]:
        """
        Segment words into multiple lines
        :param words: List of words
        :param max_distance: Maximum allowed Jaccard distance between two words to be considered in a same line
        :param min_words_per_line: Minimum allowed of words per line, if the line contains less than that limit, it is ignored
        :return: List of lines, each line contains list of words
        """
        # Create Jaccard distance matrix between all pairs of words
        words_cnt = len(words)
        distance_matrix = np.ones(shape=(words_cnt, words_cnt))
        for i in range(words_cnt):
            for j in range(words_cnt):
                a, b = words[i].box, words[j].box
                if a.y > b.y + b.h or b.y > a.y + a.h:
                    continue
                intersection = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
                union = a.h + b.h - intersection
                distance_matrix[i, j] = 1 - np.clip(intersection / union if union > 0 else 0, 0, 1)

        # Cluster words into lines
        dbscan = DBSCAN(eps=max_distance, min_samples=min_words_per_line, metric='precomputed').fit(distance_matrix)
        clusters = defaultdict(list)
        for i, cluster_id in enumerate(dbscan.labels_):
            if cluster_id == -1:
                continue
            clusters[cluster_id].append(words[i])
        lines = sorted(clusters.values(), key=lambda line: [word.box.y + word.box.h / 2 for word in line])

        # Sort words in the same line
        result = []
        for line in lines:
            result.append(WordSegmentation.sort_line(line))
        return result

    def segment_words(self,
                      img: np.ndarray,
                      kernel_size: int,
                      sigma: float,
                      theta: float,
                      min_word_area: int) -> List[Word]:
        """
        Word segmentation by using Scale-space technique method
        :param img: Original image
        :param kernel_size: Size of Gaussian kernel (must be an odd number)
        :param sigma: Standard deviation of Gaussian filter (low-pass filter) kernel
        :param theta: Ratio between width / height of words
        :param min_word_area: Minimum allowed area of the bounding box to be considered as a word
        :return: List of lines, each line contains list of words
        """
        assert img.ndim == 2
        assert img.dtype == np.uint8

        # Apply Gaussian filter and binary threshold
        kernel = WordSegmentation.create_gaussian_kernel(kernel_size=kernel_size,
                                                         sigma=sigma,
                                                         theta=theta)
        convolved_img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
        convolved_img = 255 - cv2.threshold(convolved_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Find all connected components of black pixels, each connected component is represented as a word
        words = []
        components = cv2.findContours(convolved_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        for component in components:
            # Ignore small area
            if cv2.contourArea(component) < min_word_area:
                continue
            x, y, h, w = cv2.boundingRect(component)
            cropped_img = img[y:y + h, x:x + w]
            words.append(Word(cropped_img, BoundingBox(x, y, h, w)))
        return words

    def predict(self,
                data: Path,
                is_visualized=False) -> Tuple[Dict[str, List[List[Word]]], Dict[str, float]]:
        """
        Full pipeline of word segmentation algorithm, used for both pages and lines images
        :return: Dictionary of images, each image contains a list of lines, each line contains list of words
        """
        images = get_image_files(data)
        results, rescale_factors = {}, {}
        cnt_img = 0
        for img_name in images:
            cnt_img += 1
            print(f'Processing image {cnt_img}/{len(images)}: {img_name}')
            img, rescale_factor = self.preprocessing(cv2.imread(img_name), self.resize_height)
            words = self.segment_words(img,
                                       kernel_size=self.kernel_size,
                                       sigma=self.sigma,
                                       theta=self.theta,
                                       min_word_area=self.min_word_area)
            lines = self.segment_lines(words)
            img_name = get_image_name(img_name, is_remove_ext=True)
            results[img_name] = lines
            rescale_factors[img_name] = rescale_factor
            if is_visualized:
                visualize_result(img, lines)
        return results, rescale_factors
