import json

from path import Path

from constants import Crop
from entities import BoundingBox
from word_segmentation import WordSegmentation


class Evaluator:
    def __init__(self):
        self.answer = None
        self.predict = None
        self.rescale_factors = None
        self.params = None

    def load_answer(self, path: str = './data/words.txt'):
        """
        Load answer from the dataset
        :param path: Path to dataset
        """
        with open(path) as f:
            lines = f.read().split('\n')[18:]
        self.answer = {}
        for line in lines:
            word_id, status, gray_level, x, y, w, h = line.split(' ')[:7]
            self.answer[word_id] = BoundingBox(int(x), int(y), int(w), int(h))

    def load_predict(self,
                     data,
                     kernel_size: int = 25,
                     sigma: float = 11,
                     theta: float = 5,
                     min_word_area: int = 100,
                     resize_height: int = 0):
        """
        Load predict from our model with pre-defined parameters
        """
        word_segmentation = WordSegmentation(kernel_size=kernel_size,
                                             sigma=sigma,
                                             theta=theta,
                                             min_word_area=min_word_area,
                                             resize_height=resize_height)
        self.predict, self.rescale_factors = word_segmentation.predict(Path(data))

    @staticmethod
    def calculate_iou(box1: BoundingBox,
                      box2: BoundingBox) -> float:
        """
        Calculate Intersection Over Union (IOU) between two bounding boxes
        :param box1: Bounding box 1
        :param box2: Bounding box 2
        :return: IOU score
        """
        intersection = max(min(box1.x + box1.w, box2.x + box2.w) - max(box1.x, box2.x), 0) * \
                       max(min(box1.y + box1.h, box2.y + box2.h) - max(box1.y, box2.h), 0)
        union = box1.area() + box2.area() - intersection
        return intersection / union

    @staticmethod
    def standardize_coordinate(box: BoundingBox,
                               rescale_factor: float) -> BoundingBox:
        """
        # Reverse to original coordinate
        :param box: Bounding box
        :param rescale_factor: Rescale factor
        :return: Original bounding box
        """
        box.x = round(box.x * rescale_factor + Crop.OFFSET_L)
        box.y = round(box.y * rescale_factor + Crop.OFFSET_U)
        box.w = round(box.w * rescale_factor)
        box.h = round(box.h * rescale_factor)
        return box

    def evaluate(self,
                 input_data: str = './data/preprocessed-forms',
                 predict_data: str = './data/words.txt',
                 kernel_size: int = 25,
                 sigma: float = 11,
                 theta: float = 5,
                 min_word_area: int = 100,
                 resize_height: int = 1000):
        self.load_answer(predict_data)
        self.load_predict(input_data,
                          kernel_size=kernel_size,
                          sigma=sigma,
                          theta=theta,
                          min_word_area=min_word_area,
                          resize_height=resize_height)
        self.params = {
            'kernel_size': kernel_size,
            'sigma': sigma,
            'theta': theta,
            'min_word_area': min_word_area,
            'resize_height': resize_height
        }
        # Calculate means of Intersection Over Union (IOU)
        ious = []
        cnt_img = 0
        for img_id, lines in self.predict.items():
            cnt_img += 1
            print(f'Evaluating on image {cnt_img}/{len(self.predict)}: {img_id}')
            sum_iou, cnt_word = 0, 0
            for line_id in range(len(lines)):
                line = lines[line_id]
                for word_id in range(len(line)):
                    word = line[word_id]
                    word.box = self.standardize_coordinate(word.box, self.rescale_factors[img_id])
                    combined_id = '{}-{:02}-{:02}'.format(img_id, line_id, word_id)
                    iou = 0
                    if combined_id in self.answer:
                        iou = self.calculate_iou(word.box, self.answer[combined_id])
                    sum_iou += iou
                    cnt_word += 1
            ious.append(sum_iou / cnt_word)
            print(f'IOU on image {img_id}: {sum_iou / cnt_word}')
        print(f'Mean of IOU: {sum(ious) / len(ious)}')

    def export_params(self, file_name: str = 'optimized_params'):
        """
        Export parameters
        :param file_name: File name
        """
        if file_name is None:
            return
        with open(f'./models/{file_name}.txt', 'w') as f:
            json.dump(self.params, f, indent=4)
        print(f'Exported {file_name}.txt')


if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.evaluate()
    evaluator.export_params()
