import cv2
import matplotlib.pyplot as plt

from typing import List

import numpy as np
from path import Path

from constants import Crop
from entities import Word


def get_image_files(data: Path) -> List[Path]:
    """
    Return all images in a folder
    :param data: Folder's path
    :return: List of images' path
    """
    paths = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        paths += Path(data).files(ext)
    return paths


def get_image_name(data: Path, is_remove_ext: bool = False) -> str:
    """
    Get image name from path
    :param data: Image path
    :param is_remove_ext: Remove extension of file or not
    :return: Image name
    """
    image_name = data.split('/')[-1]
    if is_remove_ext:
        image_name = image_name.split('.')[0]
    return image_name


def visualize_result(img: np.ndarray,
                     lines: List[List[Word]]):
    """
    Visualize results
    :param img: Preprocessed image
    :param lines: List of lines, each line contains list of words
    """
    plt.imshow(img, cmap='gray')
    num_colors = 7
    colors = plt.cm.get_cmap('rainbow', num_colors)
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line):
            xs = [word.box.x, word.box.x, word.box.x + word.box.w, word.box.x + word.box.w, word.box.x]
            ys = [word.box.y, word.box.y + word.box.h, word.box.y + word.box.h, word.box.y, word.box.y]
            plt.plot(xs, ys, c=colors(line_idx % num_colors))
            plt.text(word.box.x, word.box.y, f'{line_idx},{word_idx}', fontsize='x-small')
    plt.show()


def preprocess_dataset(data: Path = Path('./data/forms'), output: Path = Path('./data/preprocessed-forms')):
    """
    Crop IAM dataset header and footer
    :param data: Path to folder contains IAM dataset
    """
    images = Path(data).files()
    cnt_img = 0
    for img_name in images:
        cnt_img += 1
        print(f'Preprocessing image {cnt_img}/{len(images)}: {img_name}')
        img = cv2.imread(img_name)
        height, width, channels = img.shape
        img = img[Crop.OFFSET_U:height - Crop.OFFSET_D, Crop.OFFSET_L:width - Crop.OFFSET_R]
        cv2.imwrite(output + f'/{get_image_name(img_name)}', img)


if __name__ == '__main__':
    preprocess_dataset()
