import numpy as np


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def processBoxes(image, boxes, labels, obj_thresh, quiet=True):
    processedBoxes = []
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if labels[i] == "car":
                thresh = 0.86
            elif labels[i] == "person":
                thresh = 0.76
            else:
                thresh = obj_thresh
            if box.classes[i] > thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score() * 100, 2)) + '%')
                label = i
            if not quiet: print(label_str)

        if label >= 0:
            className = labels[label]
            confidence = box.get_score()
            xcenter = box.xmin + (box.xmax - box.xmin) / 2
            ycenter = box.ymin + (box.ymax - box.ymin) / 2
            width = (box.xmax - box.xmin)
            height = (box.ymax - box.ymin)

            processedBoxes.append([className, confidence, [xcenter, ycenter, width, height]])

    return processedBoxes
