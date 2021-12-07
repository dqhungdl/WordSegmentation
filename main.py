import argparse

from word_segmentation import WordSegmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/page')
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    parser.add_argument('--theta', type=float, default=5)
    parser.add_argument('--min_word_area', type=int, default=100)
    parser.add_argument('--resize_height', type=int, default=1000)
    args = parser.parse_args()

    word_segmentation = WordSegmentation(kernel_size=args.kernel_size,
                                         sigma=args.sigma,
                                         theta=args.theta,
                                         min_word_area=args.min_word_area,
                                         resize_height=args.resize_height)
    word_segmentation.predict(data=args.data,
                              is_visualized=True)


if __name__ == '__main__':
    main()
