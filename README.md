# Scale space technique for Word Segmentation

## Installation

Install dependant packages.

```angular2html
pip install requirements.txt
```

## Usage
Run `python main.py` to see how word segmentation works.
List of arguments:
* `data (default='./data/page')`: Path to folder contains list of image
* `kernel_size (default=25)`: Size of Gaussian kernel (must be an odd number)
* `sigma (default=11)`: Standard deviation of Gaussian filter (low-pass filter) kernel
* `theta (default=5)`: Ratio between width / height of words
* `min_word_area (default=100)`: Minimal area for a word
* `size (default=1000)`: Resize factor

Example usage:
```angular2html
python main.py --data ./data/page --size 1000 --theta 5
```